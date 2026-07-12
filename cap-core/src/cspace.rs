//! The capability space (CSpace) + derivation tree (MDB) + operations (SPEC-CAP §3, §5, §7).
//!
//! P2 uses a **single-level** CSpace: one root `CNode` of `N` slots, a cptr is a slot index
//! (an index path, never a memory pointer — §3), and the MDB is a parent link per slot. All
//! operations are safe code funnelling through `cap-core`'s one `unsafe` fabrication point;
//! a revoked/empty slot resolves to an error, never UB (CAP-REVOKE-1).

use crate::cap::{Cap, CapType, ObjectType, RawCap};
use crate::rights::Rights;
use crate::sched::Budget;

/// A capability pointer: an index into the (single-level, P2) root CNode.
pub type Cptr = usize;

/// MDB parent link, sized to hold any slot index (`Cptr`), so a large `N` can never
/// truncate a parent pointer or collide with `NIL`. `NIL` marks a derivation root.
type SlotIdx = usize;
const NIL: SlotIdx = usize::MAX;

/// Why a capability operation failed. Every operation fails **cleanly** with one of these,
/// never with UB (CAP-REVOKE-1).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CapError {
    /// cptr is not a valid slot index.
    OutOfBounds,
    /// The slot holds no capability (empty, revoked, or deleted).
    EmptySlot,
    /// The capability is not the expected object type.
    WrongType,
    /// The destination slot already holds a capability.
    SlotOccupied,
    /// The operation is not permitted, or would widen rights.
    InsufficientRights,
    /// The `Untyped` has insufficient frames for the requested retype.
    OutOfBudget,
}

/// A capability-table entry: the capability plus its parent in the derivation tree.
/// `parent == NIL` marks a derivation root (the primordial `Untyped`).
#[derive(Clone, Copy)]
struct Cte {
    cap: RawCap,
    parent: SlotIdx,
}

impl Cte {
    const EMPTY: Self = Self {
        cap: RawCap::NULL,
        parent: NIL,
    };
}

/// A single-level capability space: a root CNode of `N` slots and the parent-link MDB over
/// them. `zero` is the object-zeroing hook (CAP-MEM-2 / CAP-REVOKE-2): the kernel zeroes
/// frames through the HHDM, host tests pass a no-op. It receives `(frame_number, frames)`.
pub struct CSpace<const N: usize> {
    slots: [Cte; N],
    zero: fn(u64, u32),
    /// Monotonic id source for objects with no physical `objref` (e.g. `Sched`, which is
    /// pure CPU-time accounting, not frame-backed). Frame-backed objects use their frame
    /// number as `objref`; `Sched` objrefs are drawn from here so each is a distinct identity.
    next_obj: u64,
}

impl<const N: usize> CSpace<N> {
    /// An empty CSpace. Install the primordial capability with [`set_root_untyped`](Self::set_root_untyped).
    #[must_use]
    pub fn new(zero: fn(u64, u32)) -> Self {
        Self {
            slots: [Cte::EMPTY; N],
            zero,
            next_obj: 1,
        }
    }

    /// Install the primordial `Untyped` capability (RETYPE only) at slot 0 — the kernel's
    /// inherent authority over the physical region `[base, base+frames)` at boot, and the
    /// root of every derivation. Called once by the boot path.
    pub fn set_root_untyped(&mut self, base: u64, frames: u32) {
        self.slots[0] = Cte {
            cap: RawCap {
                // RETYPE only — NOT DERIVE. An Untyped must never be COPY/MINT-able: its
                // allocation watermark is per-cap, so a duplicate would fork the budget and
                // alias physical frames (SPEC-CAP §2.1 lists Untyped rights as RETYPE/SPLIT).
                cap_type: CapType::Untyped,
                rights: Rights::RETYPE,
                objref: base,
                size: frames,
                watermark: 0,
                aux: 0,
                badge: 0,
            },
            parent: NIL,
        };
    }

    /// Install a primordial `Sched` capability (CPU-time `budget` per `period`) at `slot` —
    /// the kernel's initial CPU-time authority (SPEC-CAP §6) and the root of the `Sched`
    /// derivation tree. Like [`set_root_untyped`](Self::set_root_untyped), a trusted boot-path
    /// bootstrap; `slot` must be a valid, empty slot. `DERIVE` authorizes SPLIT/DELEGATE; a
    /// `Sched` is never COPY/MINT-able (its per-cap budget would fork — see [`mint`](Self::mint)).
    pub fn set_root_sched(&mut self, slot: Cptr, budget: u32, period: u32) {
        debug_assert!(slot < N, "set_root_sched: slot out of bounds");
        let objref = self.next_obj;
        self.next_obj += 1;
        self.slots[slot] = Cte {
            cap: RawCap {
                cap_type: CapType::Sched,
                rights: Rights::DERIVE,
                objref,
                size: budget,
                watermark: 0, // consumed this period
                aux: period,
                badge: 0,
            },
            parent: NIL,
        };
    }

    // ---- resolution (SPEC-CAP §3) ----

    fn live(&self, c: Cptr) -> Result<RawCap, CapError> {
        if c >= N {
            return Err(CapError::OutOfBounds);
        }
        if self.slots[c].cap.is_null() {
            return Err(CapError::EmptySlot);
        }
        Ok(self.slots[c].cap)
    }

    /// Resolve a cptr to its type-erased capability record. A revoked/deleted cap resolves
    /// to `EmptySlot` — the "fails cleanly" of CAP-REVOKE-1.
    pub fn resolve(&self, c: Cptr) -> Result<RawCap, CapError> {
        self.live(c)
    }

    /// Resolve + type-check + fabricate a typed handle: the single legitimate path from a
    /// stored `RawCap` to a usable `Cap<T>`.
    pub fn get<T: ObjectType>(&self, c: Cptr) -> Result<Cap<T>, CapError> {
        let raw = self.live(c)?;
        if raw.cap_type as u8 != T::TYPE as u8 {
            return Err(CapError::WrongType);
        }
        // SAFETY: `raw` is a legitimately-derived capability read from this CSpace, and its
        // type was just checked to equal `T::TYPE`; this is the authorized fabrication.
        Ok(unsafe { Cap::<T>::fabricate(raw) })
    }

    // ---- derivation (SPEC-CAP §5) ----

    /// RETYPE `count` objects of `new_type` (each `frames_per_obj` frames) out of the
    /// `Untyped` at `ut`, into `dest_start..dest_start+count`. Charges the untyped budget
    /// (§8), zeroes each object before its cap exists (CAP-MEM-2), and records each new cap
    /// as a child of the untyped.
    pub fn retype(
        &mut self,
        ut: Cptr,
        new_type: CapType,
        frames_per_obj: u32,
        count: u32,
        dest_start: Cptr,
    ) -> Result<(), CapError> {
        let u = self.live(ut)?;
        if u.cap_type != CapType::Untyped {
            return Err(CapError::WrongType);
        }
        if !u.rights.contains(Rights::RETYPE) {
            return Err(CapError::InsufficientRights);
        }
        if !is_retypeable(new_type) {
            return Err(CapError::WrongType); // e.g. Null / Untyped / Reply / IrqControl
        }
        if frames_per_obj == 0 || count == 0 {
            return Err(CapError::OutOfBudget); // no zero-size / zero-count objects
        }
        let need = frames_per_obj
            .checked_mul(count)
            .ok_or(CapError::OutOfBudget)?;
        if need > u.size - u.watermark {
            return Err(CapError::OutOfBudget);
        }
        let end = dest_start
            .checked_add(count as usize)
            .ok_or(CapError::OutOfBounds)?;
        if end > N {
            return Err(CapError::OutOfBounds);
        }
        for i in 0..count as usize {
            if !self.slots[dest_start + i].cap.is_null() {
                return Err(CapError::SlotOccupied);
            }
        }

        // All checks passed — commit.
        let base = u.objref + u64::from(u.watermark);
        self.slots[ut].cap.watermark += need;
        let rights = default_rights(new_type);
        for i in 0..count as usize {
            let objref = base + u64::from(i as u32 * frames_per_obj);
            (self.zero)(objref, frames_per_obj); // CAP-MEM-2: zero before the cap exists
            self.slots[dest_start + i] = Cte {
                cap: RawCap {
                    cap_type: new_type,
                    rights,
                    objref,
                    size: frames_per_obj,
                    watermark: 0,
                    aux: 0,
                    badge: 0,
                },
                parent: ut,
            };
        }
        Ok(())
    }

    /// MINT a derived capability from `src` into `dest` with `new_rights` (≤ src rights,
    /// CAP-DERIVE-1) and `badge`. The new cap is a *child* of `src`.
    pub fn mint(
        &mut self,
        src: Cptr,
        dest: Cptr,
        new_rights: Rights,
        badge: u64,
    ) -> Result<(), CapError> {
        let s = self.live(src)?;
        if matches!(
            s.cap_type,
            CapType::Untyped | CapType::Sched | CapType::Reply
        ) {
            // Non-duplicable types. Untyped/Sched carry per-cap allocation state (retype
            // watermark; CPU-time budget), so a COPY/MINT would fork it — forking the frame
            // budget (§8/§2.1) or forging CPU time (DEC-0003-2); sub-allocation is RETYPE /
            // SPLIT, never MINT. `Reply` is single-use and names exactly one blocked caller
            // (CAP-REPLY-1): duplicating it would let a server hoard reply authority and reply
            // to the wrong/old caller — the exact leak the split-out Reply cap exists to prevent.
            return Err(CapError::InsufficientRights);
        }
        if !s.rights.contains(Rights::DERIVE) {
            return Err(CapError::InsufficientRights);
        }
        if !new_rights.subset_of(s.rights) {
            return Err(CapError::InsufficientRights); // widening refused
        }
        self.place(
            dest,
            RawCap {
                rights: new_rights,
                badge,
                ..s
            },
            src,
        )
    }

    /// COPY `src` into `dest` with EQUAL rights (no badge change). A copy is a *sibling* in
    /// the MDB (same parent), so a revoke of an ancestor treats copies alike.
    pub fn copy(&mut self, src: Cptr, dest: Cptr) -> Result<(), CapError> {
        let s = self.live(src)?;
        if matches!(
            s.cap_type,
            CapType::Untyped | CapType::Sched | CapType::Reply
        ) {
            return Err(CapError::InsufficientRights); // Untyped/Sched/Reply are never duplicable (see mint)
        }
        if !s.rights.contains(Rights::DERIVE) {
            return Err(CapError::InsufficientRights);
        }
        // A copy is a sibling (same parent). But a copy of a derivation ROOT (parent NIL) would
        // be a second root that revoke() can never reach, so parent it to the source instead.
        let src_parent = self.slots[src].parent;
        let parent = if src_parent == NIL { src } else { src_parent };
        self.place(dest, s, parent)
    }

    /// MOVE `src` to `dest` (no new authority): the cap and its MDB position move; `src`
    /// empties; any children of `src` are reparented to `dest`.
    pub fn move_cap(&mut self, src: Cptr, dest: Cptr) -> Result<(), CapError> {
        let e = *self.cte_ref(src)?;
        if dest == src {
            return Ok(());
        }
        if dest >= N {
            return Err(CapError::OutOfBounds);
        }
        if !self.slots[dest].cap.is_null() {
            return Err(CapError::SlotOccupied);
        }
        for k in 0..N {
            if self.slots[k].parent == src {
                self.slots[k].parent = dest;
            }
        }
        self.slots[dest] = e;
        self.slots[src] = Cte::EMPTY;
        Ok(())
    }

    // ---- Sched budget derivation (SPEC-CAP §6, ADR-0003 DEC-0003-4) ----

    /// SPLIT a `Sched`: carve `amount` CPU-time budget out of `src` into a **new** child
    /// `Sched` at `dest`. Monotonic — total budget is conserved (`src` loses exactly what
    /// `dest` gains), so a split creates no CPU time. The child is a derivation child of
    /// `src`, so revoking `src` reclaims it. All-or-nothing: if `dest` is occupied/out of
    /// bounds or `amount` exceeds `src`'s available budget, `src` is left untouched.
    pub fn split(&mut self, src: Cptr, dest: Cptr, amount: u32) -> Result<(), CapError> {
        let s = self.live(src)?;
        if s.cap_type != CapType::Sched {
            return Err(CapError::WrongType);
        }
        if !s.rights.contains(Rights::DERIVE) {
            return Err(CapError::InsufficientRights);
        }
        let mut parent = Budget::from_cap(&s);
        let child = parent.split(amount).ok_or(CapError::OutOfBudget)?;
        let objref = self.next_obj;
        let child_cap = RawCap {
            cap_type: CapType::Sched,
            rights: s.rights,
            objref,
            size: child.capacity,
            watermark: child.consumed,
            aux: child.period,
            badge: 0,
        };
        // place() validates dest BEFORE we commit the debit to src (all-or-nothing).
        self.place(dest, child_cap, src)?;
        parent.write_into(&mut self.slots[src].cap);
        self.next_obj += 1;
        Ok(())
    }

    /// DELEGATE: transfer `amount` CPU-time budget from `src` into an existing `Sched` at
    /// `dest`. Monotonic — the total across the two is unchanged. This is the passive-server
    /// primitive (DEC-0003-4): a caller hands budget to a server so the server runs on the
    /// caller's time (P4 wires it to the IPC call path). A self-delegate is a no-op.
    pub fn delegate(&mut self, src: Cptr, dest: Cptr, amount: u32) -> Result<(), CapError> {
        if src == dest {
            return Ok(());
        }
        let s = self.live(src)?;
        let d = self.live(dest)?;
        if s.cap_type != CapType::Sched || d.cap_type != CapType::Sched {
            return Err(CapError::WrongType);
        }
        if !s.rights.contains(Rights::DERIVE) {
            return Err(CapError::InsufficientRights);
        }
        let mut src_b = Budget::from_cap(&s);
        let mut dst_b = Budget::from_cap(&d);
        if !src_b.delegate(&mut dst_b, amount) {
            return Err(CapError::OutOfBudget);
        }
        src_b.write_into(&mut self.slots[src].cap);
        dst_b.write_into(&mut self.slots[dest].cap);
        Ok(())
    }

    // ---- IPC reply authority (SPEC-CAP §2 CAP-REPLY-1, ADR-0004) ----

    /// Mint a single-use `Reply` capability at `dest`, naming the one blocked caller `caller`
    /// (an opaque caller-record id the IPC layer assigns) and stamped with the call's endpoint
    /// `badge`. This is the kernel-only reply-authority path (CAP-REPLY-1): the `Reply` is not
    /// derived from an `Untyped`, carries no rights (possession *is* the authority to reply
    /// once), is non-duplicable (COPY/MINT refuse it), and is consumed by [`consume_reply`]. It
    /// is placed as a derivation root; the IPC layer tears it down on abort/revoke of the call.
    pub fn mint_reply(&mut self, dest: Cptr, caller: u64, badge: u64) -> Result<(), CapError> {
        self.place(
            dest,
            RawCap {
                cap_type: CapType::Reply,
                rights: Rights::empty(),
                objref: caller, // names exactly the one blocked caller
                size: 0,
                watermark: 0,
                aux: 0,
                badge,
            },
            NIL,
        )
    }

    /// Consume the `Reply` at `dest`, returning the caller id it named. This is the single-use
    /// REPLY (CAP-REPLY-1): it empties the slot, so a *second* reply on the same cptr resolves
    /// to `EmptySlot` and fails cleanly. Fails `WrongType` if the slot doesn't hold a `Reply`.
    pub fn consume_reply(&mut self, dest: Cptr) -> Result<u64, CapError> {
        let r = self.live(dest)?;
        if r.cap_type != CapType::Reply {
            return Err(CapError::WrongType);
        }
        let caller = r.objref;
        self.destroy_slot(dest); // single-use: the slot is now empty (a re-reply → EmptySlot)
        Ok(caller)
    }

    // ---- revocation (SPEC-CAP §7) ----

    /// REVOKE: destroy every capability derived from `c` (transitively via the MDB), leaving
    /// `c` intact. Objects with no remaining cap are zeroed (CAP-REVOKE-2); an `Untyped`'s
    /// budget is reclaimed. Revoked slots become empty → later use fails cleanly.
    pub fn revoke(&mut self, c: Cptr) -> Result<(), CapError> {
        self.live(c)?;
        // Destroy descendants leaf-first: a leaf has no children, so removing it can't
        // orphan anything, and each pass removes one node until the subtree is empty.
        while let Some(d) = (0..N).find(|&d| {
            d != c
                && !self.slots[d].cap.is_null()
                && self.is_descendant(d, c)
                && !self.has_children(d)
        }) {
            self.destroy_slot(d);
        }
        if self.slots[c].cap.cap_type == CapType::Untyped {
            self.slots[c].cap.watermark = 0; // reclaim once the retyped children are gone
        }
        Ok(())
    }

    /// DELETE: revoke `c`'s descendants, then destroy `c` itself.
    pub fn delete(&mut self, c: Cptr) -> Result<(), CapError> {
        self.revoke(c)?;
        self.destroy_slot(c);
        Ok(())
    }

    // ---- internals ----

    fn cte_ref(&self, c: Cptr) -> Result<&Cte, CapError> {
        if c >= N {
            return Err(CapError::OutOfBounds);
        }
        if self.slots[c].cap.is_null() {
            return Err(CapError::EmptySlot);
        }
        Ok(&self.slots[c])
    }

    fn place(&mut self, dest: Cptr, cap: RawCap, parent: SlotIdx) -> Result<(), CapError> {
        if dest >= N {
            return Err(CapError::OutOfBounds);
        }
        if !self.slots[dest].cap.is_null() {
            return Err(CapError::SlotOccupied);
        }
        self.slots[dest] = Cte { cap, parent };
        Ok(())
    }

    fn destroy_slot(&mut self, d: Cptr) {
        let cap = self.slots[d].cap;
        if cap.is_null() {
            return;
        }
        let parent = self.slots[d].parent;
        self.slots[d] = Cte::EMPTY;
        // Return a destroyed Sched's remaining budget to its Sched parent — CPU-time is
        // conserved across revoke/delete, not leaked. The parent granted this budget via SPLIT,
        // so reclaiming it on destruction is the CPU-time analogue of an Untyped reclaiming its
        // frames when a retyped child is destroyed. Leaf-first revocation means a subtree's
        // budget flows back up one level per destroyed node until it reaches the revoked root.
        if cap.cap_type == CapType::Sched && parent != NIL {
            let p = &mut self.slots[parent].cap;
            if p.cap_type == CapType::Sched {
                p.size = p.size.saturating_add(cap.size);
            }
        }
        // Zero the object if this was its last capability (CAP-REVOKE-2 no-residual). Untyped
        // memory is zeroed on the next RETYPE (CAP-MEM-2), so it needs no zeroing here; Sched
        // is pure CPU-time accounting with no backing frames (its `size` is a budget, not a
        // frame count); a `Reply` is a kernel record naming a caller (its `objref` is a caller
        // id, not a frame number) — zeroing any of these would scribble memory at a bogus address.
        let another = (0..N).any(|k| {
            let c = self.slots[k].cap;
            !c.is_null() && c.cap_type as u8 == cap.cap_type as u8 && c.objref == cap.objref
        });
        if !another
            && !matches!(
                cap.cap_type,
                CapType::Untyped | CapType::Sched | CapType::Reply
            )
        {
            (self.zero)(cap.objref, cap.size);
        }
    }

    fn is_descendant(&self, start: Cptr, ancestor: Cptr) -> bool {
        let mut d = start;
        for _ in 0..=N {
            let p = self.slots[d].parent;
            if p == NIL {
                return false;
            }
            if p == ancestor {
                return true;
            }
            d = p;
        }
        false // cycle guard (should be unreachable)
    }

    fn has_children(&self, d: Cptr) -> bool {
        (0..N).any(|k| !self.slots[k].cap.is_null() && self.slots[k].parent == d)
    }
}

// ---- GRANT over IPC: cross-CSpace capability transfer (SPEC-CAP §5, ADR-0004 DEC-0004-8) ----

/// Whether a [`grant`] hands the capability off (source loses it) or derives a narrowed child.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GrantMode {
    /// Transfer ownership: the source slot is vacated (single-owner handoff).
    Move,
    /// Derive a child in the destination; the source keeps its capability.
    Mint,
}

/// GRANT a capability over IPC: transfer `src[src_slot]` into `dst[dst_slot]` with `new_rights`
/// (which MUST be a subset of the source's rights — monotonic, CAP-DERIVE-1) and `badge`.
/// `Move` vacates the source (single-owner handoff); `Mint` leaves the source and derives a
/// narrowed child in the destination. **All-or-nothing:** the destination is validated before
/// the source is touched, so a full destination or a rights violation leaves both CSpaces
/// unchanged. Atomicity w.r.t. a concurrent REVOKE is the caller's obligation (the kernel runs
/// this preemption-masked, DEC-0003-7) — within this function no partial state is ever observable.
///
/// Refuses to grant a `Reply` (single-use, bound to one caller — never transferable) and to
/// `Mint`-fork an `Untyped`/`Sched` (their per-cap allocation state can't be duplicated). The
/// granted capability lands as a derivation root in `dst` (cross-CSpace MDB parenting — a
/// granter revoking a `Mint`-granted descendant across CSpaces — lands with real processes, P7).
pub fn grant<const N: usize, const M: usize>(
    src: &mut CSpace<N>,
    src_slot: Cptr,
    dst: &mut CSpace<M>,
    dst_slot: Cptr,
    new_rights: Rights,
    badge: u64,
    mode: GrantMode,
) -> Result<(), CapError> {
    let s = src.live(src_slot)?;
    if s.cap_type == CapType::Reply {
        // A Reply names exactly one blocked caller and is single-use — it is never transferable
        // (CAP-REPLY-1); granting it would be the reply-authority-hoarding leak it prevents.
        return Err(CapError::InsufficientRights);
    }
    if mode == GrantMode::Mint && matches!(s.cap_type, CapType::Untyped | CapType::Sched) {
        return Err(CapError::InsufficientRights); // a MINT-grant would fork per-cap state
    }
    if !s.rights.contains(Rights::GRANT) {
        return Err(CapError::InsufficientRights); // the cap must be transferable over IPC
    }
    if !new_rights.subset_of(s.rights) {
        return Err(CapError::InsufficientRights); // monotonic: rights can narrow, never widen
    }
    if mode == GrantMode::Move && src.has_children(src_slot) {
        // A MOVE vacates the source slot, but its local MDB children live in the SOURCE CSpace
        // and cannot follow across CSpaces — vacating would orphan them (dangling parent links,
        // and a revoke that can no longer reach them: a CAP-REVOKE-1 hazard). Refuse; the granter
        // must revoke/relocate its derivations first, or use a MINT-grant (which keeps the source).
        return Err(CapError::InsufficientRights);
    }
    let granted = RawCap {
        rights: new_rights,
        badge,
        ..s
    };
    // Validate + fill the destination FIRST (all-or-nothing); only then vacate the source.
    dst.place(dst_slot, granted, NIL)?;
    if mode == GrantMode::Move {
        // Vacate WITHOUT zeroing: the object moved to `dst` and is still alive there.
        src.slots[src_slot] = Cte::EMPTY;
    }
    Ok(())
}

/// Object types RETYPE can produce from `Untyped`. Excludes `Null` (the empty marker),
/// `Untyped` (no sub-retype in the single-level model — it would fork the watermark),
/// `Reply`/`IrqControl` (kernel-minted, not retyped — CAP-REPLY-1), and `Sched` (created
/// with an explicit budget+period via [`set_root_sched`](CSpace::set_root_sched) / SPLIT, so
/// a RETYPE — which has no budget argument — could only make a malformed zero-period Sched).
fn is_retypeable(t: CapType) -> bool {
    !matches!(
        t,
        CapType::Null | CapType::Untyped | CapType::Reply | CapType::IrqControl | CapType::Sched
    )
}

/// Rights a freshly-retyped object of `t` gets — the creator owns it fully (MINT narrows).
/// `Frame` gets no EXECUTE by default (W^X: execute is minted explicitly, CAP-MEM-1). A
/// retyped `Untyped` gets RETYPE only (never DERIVE — it must not be COPY/MINT-able).
fn default_rights(t: CapType) -> Rights {
    match t {
        CapType::Untyped => Rights::RETYPE,
        CapType::Frame => {
            Rights::READ | Rights::WRITE | Rights::MAP | Rights::GRANT | Rights::DERIVE
        }
        CapType::CNode => Rights::DERIVE | Rights::GRANT,
        CapType::Endpoint => Rights::SEND | Rights::RECV | Rights::GRANT | Rights::DERIVE,
        CapType::Notification => Rights::SIGNAL | Rights::RECV | Rights::DERIVE,
        _ => Rights::DERIVE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nz(_: u64, _: u32) {}

    /// A CSpace with the primordial untyped (`frames` frames at frame 0) in slot 0.
    fn boot<const N: usize>(frames: u32) -> CSpace<N> {
        let mut cs = CSpace::<N>::new(nz);
        cs.set_root_untyped(0, frames);
        cs
    }

    #[test]
    fn retype_produces_typed_objects_charged_to_budget() {
        let mut cs = boot::<32>(100);
        // 3 frames out of the untyped into slots 1,2,3.
        cs.retype(0, CapType::Frame, 1, 3, 1).unwrap();
        for s in 1..=3 {
            let f = cs.get::<crate::cap::Frame>(s).unwrap();
            assert!(f.allows(Rights::READ | Rights::WRITE));
            assert!(!f.allows(Rights::EXECUTE), "W^X: no execute by default");
        }
        // Budget charged: 100 - 3 consumed.
        assert_eq!(cs.resolve(0).unwrap().watermark, 3);
        assert_eq!(cs.get::<crate::cap::Frame>(1).unwrap().objref(), 0);
        assert_eq!(cs.get::<crate::cap::Frame>(2).unwrap().objref(), 1);
    }

    #[test]
    fn retype_refuses_over_budget_and_wrong_type() {
        let mut cs = boot::<8>(2);
        assert_eq!(
            cs.retype(0, CapType::Frame, 1, 3, 1),
            Err(CapError::OutOfBudget)
        );
        assert_eq!(
            cs.resolve(0).unwrap().watermark,
            0,
            "refused retype charges nothing"
        );
        // slot 5 is empty → retyping from it is EmptySlot; wrong-type from a frame.
        cs.retype(0, CapType::Frame, 1, 1, 1).unwrap();
        assert_eq!(
            cs.retype(1, CapType::Frame, 1, 1, 2),
            Err(CapError::WrongType)
        );
    }

    #[test]
    fn mint_narrows_and_refuses_widening() {
        let mut cs = boot::<16>(10);
        cs.retype(0, CapType::Frame, 1, 1, 1).unwrap(); // slot 1: RW frame
                                                        // Mint a read-only view into slot 2.
        cs.mint(1, 2, Rights::READ, 0xbeef).unwrap();
        let ro = cs.get::<crate::cap::Frame>(2).unwrap();
        assert_eq!(ro.rights(), Rights::READ);
        assert_eq!(ro.badge(), 0xbeef);
        // Widening from the read-only cap is refused.
        assert_eq!(
            cs.mint(2, 3, Rights::READ | Rights::WRITE, 0),
            Err(CapError::InsufficientRights)
        );
    }

    #[test]
    fn copy_keeps_rights_move_relocates() {
        let mut cs = boot::<16>(10);
        cs.retype(0, CapType::Frame, 1, 1, 1).unwrap();
        cs.copy(1, 2).unwrap();
        assert_eq!(cs.resolve(1).unwrap().rights, cs.resolve(2).unwrap().rights);
        cs.move_cap(2, 5).unwrap();
        assert_eq!(cs.resolve(2), Err(CapError::EmptySlot));
        assert!(cs.get::<crate::cap::Frame>(5).is_ok());
    }

    #[test]
    fn revoke_destroys_all_descendants_transitively() {
        let mut cs = boot::<32>(100);
        // untyped(0) -> frames in 1,2,3; mint a child of 1 into 4; copy 2 into 5.
        cs.retype(0, CapType::Frame, 1, 3, 1).unwrap();
        cs.mint(1, 4, Rights::READ, 0).unwrap(); // child of frame 1 (grandchild of untyped)
        cs.copy(2, 5).unwrap(); // sibling of frame 2 (child of untyped)
                                // Revoke the untyped: every descendant (1,2,3,4,5) must be destroyed, untyped kept.
        cs.revoke(0).unwrap();
        for s in 1..=5 {
            assert_eq!(
                cs.resolve(s),
                Err(CapError::EmptySlot),
                "slot {s} not revoked"
            );
        }
        assert!(cs.resolve(0).is_ok(), "the untyped itself survives revoke");
        assert_eq!(
            cs.resolve(0).unwrap().watermark,
            0,
            "untyped budget reclaimed"
        );
    }

    #[test]
    fn revoked_cap_use_fails_cleanly_not_ub() {
        let mut cs = boot::<16>(10);
        cs.retype(0, CapType::Frame, 1, 1, 1).unwrap();
        cs.mint(1, 2, Rights::READ, 0).unwrap();
        cs.revoke(1).unwrap(); // destroys the minted child (2), keeps 1
        assert_eq!(cs.resolve(2), Err(CapError::EmptySlot));
        assert_eq!(
            cs.get::<crate::cap::Frame>(2).err(),
            Some(CapError::EmptySlot)
        );
        assert!(cs.resolve(1).is_ok());
    }

    #[test]
    fn out_of_bounds_and_empty_resolve_cleanly() {
        let cs = boot::<4>(10);
        assert_eq!(cs.resolve(99), Err(CapError::OutOfBounds));
        assert_eq!(cs.resolve(1), Err(CapError::EmptySlot));
        assert_eq!(
            cs.get::<crate::cap::CNode>(0).err(),
            Some(CapError::WrongType)
        );
    }

    #[test]
    fn delete_last_cap_zeroes_object() {
        use core::cell::Cell;
        thread_local!(static ZEROED: Cell<u64> = const { Cell::new(0) });
        fn rec(objref: u64, _n: u32) {
            ZEROED.with(|z| z.set(z.get() + objref + 1));
        }
        let mut cs = CSpace::<8>::new(rec);
        cs.set_root_untyped(10, 100);
        cs.retype(0, CapType::Frame, 1, 1, 1).unwrap(); // frame objref=10 (retype zeroes it, CAP-MEM-2)
        cs.copy(1, 2).unwrap(); // second cap to the same object
        ZEROED.with(|z| z.set(0)); // ignore retype's zeroing; isolate destroy-zeroing (CAP-REVOKE-2)
        cs.delete(1).unwrap(); // not the last cap → no zero yet
        ZEROED.with(|z| assert_eq!(z.get(), 0, "object still referenced; not zeroed"));
        cs.delete(2).unwrap(); // last cap → zero object (objref 10 -> +11)
        ZEROED.with(|z| assert_eq!(z.get(), 11, "last-cap delete zeroes the object"));
    }

    #[test]
    fn untyped_is_never_copyable_or_mintable() {
        // The critical fix: duplicating an Untyped would fork its per-cap watermark, letting
        // two caps retype the SAME physical frames (double-allocation). Both are refused.
        let mut cs = boot::<16>(64);
        assert_eq!(cs.copy(0, 1), Err(CapError::InsufficientRights));
        assert_eq!(
            cs.mint(0, 1, Rights::RETYPE, 0),
            Err(CapError::InsufficientRights)
        );
        // The untyped remains the sole allocator; there is exactly one watermark.
        cs.retype(0, CapType::Frame, 1, 4, 1).unwrap();
        assert_eq!(cs.resolve(0).unwrap().watermark, 4);
        assert_eq!(cs.resolve(1).unwrap().cap_type, CapType::Frame); // a real Frame, not an untyped alias
    }

    #[test]
    fn retype_refuses_degenerate_and_non_retypeable() {
        let mut cs = boot::<8>(64);
        assert_eq!(
            cs.retype(0, CapType::Frame, 0, 4, 1),
            Err(CapError::OutOfBudget)
        ); // zero size
        assert_eq!(
            cs.retype(0, CapType::Frame, 1, 0, 1),
            Err(CapError::OutOfBudget)
        ); // zero count
        assert_eq!(
            cs.retype(0, CapType::Null, 1, 1, 1),
            Err(CapError::WrongType)
        );
        assert_eq!(
            cs.retype(0, CapType::Untyped, 1, 1, 1),
            Err(CapError::WrongType)
        ); // no sub-retype
        assert_eq!(
            cs.retype(0, CapType::Reply, 1, 1, 1),
            Err(CapError::WrongType)
        ); // kernel-minted
        assert_eq!(
            cs.resolve(0).unwrap().watermark,
            0,
            "refused retypes charge nothing"
        );
    }

    // ---- Sched (P3a) ----

    fn budget_of<const N: usize>(cs: &CSpace<N>, c: Cptr) -> Budget {
        Budget::from_cap(&cs.resolve(c).unwrap())
    }

    #[test]
    fn sched_split_conserves_budget_and_scopes_to_revoke() {
        let mut cs = CSpace::<16>::new(nz);
        cs.set_root_sched(0, 100, 50); // root Sched: 100 units / 50 period
        assert_eq!(cs.resolve(0).unwrap().cap_type, CapType::Sched);
        // SPLIT 30 into slot 1, then 20 into slot 2.
        cs.split(0, 1, 30).unwrap();
        cs.split(0, 2, 20).unwrap();
        assert_eq!(budget_of(&cs, 0).capacity, 50, "root debited by 30+20");
        assert_eq!(budget_of(&cs, 1).capacity, 30);
        assert_eq!(budget_of(&cs, 2).capacity, 20);
        assert_eq!(budget_of(&cs, 1).period, 50, "child inherits period");
        // Conservation across the whole tree.
        let total =
            budget_of(&cs, 0).capacity + budget_of(&cs, 1).capacity + budget_of(&cs, 2).capacity;
        assert_eq!(total, 100, "no CPU time created by splitting");
        // Each Sched is a distinct object identity.
        assert_ne!(cs.resolve(0).unwrap().objref, cs.resolve(1).unwrap().objref);
        // REVOKE the root: children destroyed, root kept — and their budget flows BACK to the
        // root (CPU-time conserved across revoke, not leaked).
        cs.revoke(0).unwrap();
        assert_eq!(cs.resolve(1), Err(CapError::EmptySlot));
        assert_eq!(cs.resolve(2), Err(CapError::EmptySlot));
        assert!(cs.resolve(0).is_ok());
        assert_eq!(
            budget_of(&cs, 0).capacity,
            100,
            "revoke reclaims child budget to the root (no CPU time lost)"
        );
    }

    #[test]
    fn sched_revoke_reclaims_budget_transitively() {
        // root → A → B (a 2-level Sched tree). Revoking the root must funnel B's budget up
        // through A back to root, leaf-first, conserving the full 100 units.
        let mut cs = CSpace::<8>::new(nz);
        cs.set_root_sched(0, 100, 10);
        cs.split(0, 1, 60).unwrap(); // A ← 60 (root 40)
        cs.split(1, 2, 25).unwrap(); // B ← 25 from A (A 35)
        assert_eq!(budget_of(&cs, 0).capacity, 40);
        assert_eq!(budget_of(&cs, 1).capacity, 35);
        assert_eq!(budget_of(&cs, 2).capacity, 25);
        cs.revoke(0).unwrap();
        assert_eq!(cs.resolve(1), Err(CapError::EmptySlot));
        assert_eq!(cs.resolve(2), Err(CapError::EmptySlot));
        assert_eq!(
            budget_of(&cs, 0).capacity,
            100,
            "transitive reclaim: B→A→root conserves all CPU time"
        );
    }

    #[test]
    fn sched_split_over_budget_is_all_or_nothing() {
        let mut cs = CSpace::<8>::new(nz);
        cs.set_root_sched(0, 40, 10);
        assert_eq!(cs.split(0, 1, 41), Err(CapError::OutOfBudget));
        assert_eq!(
            budget_of(&cs, 0).capacity,
            40,
            "refused split leaves root untouched"
        );
        assert_eq!(cs.resolve(1), Err(CapError::EmptySlot), "no child placed");
        // A split whose destination is occupied must not debit the source either.
        cs.split(0, 1, 10).unwrap();
        assert_eq!(cs.split(0, 1, 5), Err(CapError::SlotOccupied));
        assert_eq!(
            budget_of(&cs, 0).capacity,
            30,
            "occupied-dest split did not debit source"
        );
    }

    #[test]
    fn sched_delegate_transfers_and_conserves() {
        let mut cs = CSpace::<8>::new(nz);
        cs.set_root_sched(0, 100, 10);
        cs.split(0, 1, 40).unwrap(); // slot1 has 40, root now 60
        cs.delegate(0, 1, 25).unwrap();
        assert_eq!(budget_of(&cs, 0).capacity, 35);
        assert_eq!(budget_of(&cs, 1).capacity, 65);
        assert_eq!(budget_of(&cs, 0).capacity + budget_of(&cs, 1).capacity, 100);
        // Over-available delegation refused, both untouched.
        assert_eq!(cs.delegate(0, 1, 36), Err(CapError::OutOfBudget));
        assert_eq!(budget_of(&cs, 0).capacity, 35);
        assert_eq!(budget_of(&cs, 1).capacity, 65);
        // Self-delegate is a harmless no-op (must NOT double-credit).
        cs.delegate(1, 1, 10).unwrap();
        assert_eq!(
            budget_of(&cs, 1).capacity,
            65,
            "self-delegate created no budget"
        );
    }

    #[test]
    fn sched_is_never_duplicable_or_retypeable() {
        // Like Untyped: duplicating a Sched would fork its per-cap budget (CPU-time forgery).
        let mut cs = CSpace::<8>::new(nz);
        cs.set_root_sched(0, 50, 10);
        assert_eq!(cs.copy(0, 1), Err(CapError::InsufficientRights));
        assert_eq!(
            cs.mint(0, 1, Rights::DERIVE, 0),
            Err(CapError::InsufficientRights)
        );
        // And a Sched cannot be conjured by RETYPE (no budget argument ⇒ malformed).
        let mut cu = CSpace::<8>::new(nz);
        cu.set_root_untyped(0, 64);
        assert_eq!(
            cu.retype(0, CapType::Sched, 1, 1, 1),
            Err(CapError::WrongType)
        );
    }

    #[test]
    fn sched_destroy_does_not_zero_frames() {
        // A Sched has no backing frames; destroying it must NOT invoke the frame-zeroing hook
        // (its `size` is a budget, not a frame count — zeroing would scribble memory).
        use core::cell::Cell;
        thread_local!(static ZEROED: Cell<u32> = const { Cell::new(0) });
        fn rec(_objref: u64, _n: u32) {
            ZEROED.with(|z| z.set(z.get() + 1));
        }
        let mut cs = CSpace::<8>::new(rec);
        cs.set_root_sched(0, 1_000_000, 10); // huge "size" — would be catastrophic if zeroed
        cs.split(0, 1, 500_000).unwrap();
        cs.delete(1).unwrap(); // destroy a Sched
        cs.delete(0).unwrap();
        ZEROED.with(|z| assert_eq!(z.get(), 0, "Sched destruction must never zero frames"));
    }

    // ---- IPC (P4): single-use Reply + cross-CSpace GRANT ----

    #[test]
    fn reply_is_single_use_and_names_the_caller() {
        // CAP-REPLY-1 / AC4.2: the first reply consumes the cap; a second fails cleanly.
        let mut cs = CSpace::<8>::new(nz);
        cs.mint_reply(1, 0xCA11E7, 0xBADD).unwrap();
        assert_eq!(cs.resolve(1).unwrap().cap_type, CapType::Reply);
        assert_eq!(cs.resolve(1).unwrap().badge, 0xBADD);
        assert_eq!(
            cs.consume_reply(1),
            Ok(0xCA11E7),
            "reply returns the named caller"
        );
        // Second reply on the same cptr fails cleanly (slot now empty) — the single-use guarantee.
        assert_eq!(cs.consume_reply(1), Err(CapError::EmptySlot));
        assert_eq!(cs.resolve(1), Err(CapError::EmptySlot));
    }

    #[test]
    fn reply_is_non_duplicable() {
        // A Reply must not be COPY/MINT'd (else a server hoards reply authority — the exact leak
        // CAP-REPLY-1 exists to prevent).
        let mut cs = CSpace::<8>::new(nz);
        cs.mint_reply(1, 7, 0).unwrap();
        assert_eq!(cs.copy(1, 2), Err(CapError::InsufficientRights));
        assert_eq!(
            cs.mint(1, 2, Rights::empty(), 0),
            Err(CapError::InsufficientRights)
        );
        assert!(
            cs.resolve(1).is_ok(),
            "the reply survives the refused duplication"
        );
    }

    #[test]
    fn reply_consume_zeroes_no_frames() {
        // The Reply's objref is a caller id, not a frame — consuming it must not call the zero hook.
        use core::cell::Cell;
        thread_local!(static Z: Cell<u32> = const { Cell::new(0) });
        fn rec(_: u64, _: u32) {
            Z.with(|z| z.set(z.get() + 1));
        }
        let mut cs = CSpace::<8>::new(rec);
        cs.mint_reply(1, 0xFFFF_FFFF, 0).unwrap(); // huge caller id — catastrophic if a frame
        cs.consume_reply(1).unwrap();
        Z.with(|z| assert_eq!(z.get(), 0));
    }

    /// A CSpace with a GRANT-able Frame (retyped from a 64-frame untyped) at slot 1.
    fn boot_with_frame<const N: usize>() -> CSpace<N> {
        let mut cs = CSpace::<N>::new(nz);
        cs.set_root_untyped(0, 64);
        cs.retype(0, CapType::Frame, 1, 1, 1).unwrap(); // slot 1: RW|MAP|GRANT|DERIVE Frame
        cs
    }

    #[test]
    fn grant_move_transfers_and_narrows_monotonically() {
        // AC4.5: GRANT moves a cap into the receiver's CSpace with monotonic (narrowed) rights.
        let mut client = boot_with_frame::<8>();
        let mut server = CSpace::<8>::new(nz);
        let objref = client.resolve(1).unwrap().objref;
        grant(
            &mut client,
            1,
            &mut server,
            3,
            Rights::READ,
            0xC11E,
            GrantMode::Move,
        )
        .unwrap();
        let g = server.resolve(3).unwrap();
        assert_eq!(g.cap_type, CapType::Frame);
        assert_eq!(g.objref, objref, "same object landed in the receiver");
        assert_eq!(g.rights, Rights::READ, "rights narrowed on transfer");
        assert_eq!(g.badge, 0xC11E, "badged for the receiver's accounting");
        assert_eq!(
            client.resolve(1),
            Err(CapError::EmptySlot),
            "MOVE vacated the source"
        );
    }

    #[test]
    fn grant_mint_keeps_source_and_refuses_widening() {
        let mut client = boot_with_frame::<8>();
        let mut server = CSpace::<8>::new(nz);
        grant(
            &mut client,
            1,
            &mut server,
            3,
            Rights::READ | Rights::WRITE,
            0,
            GrantMode::Mint,
        )
        .unwrap();
        assert!(client.resolve(1).is_ok(), "MINT-grant leaves the source");
        assert_eq!(
            server.resolve(3).unwrap().rights,
            Rights::READ | Rights::WRITE
        );
        // Widening beyond the source's rights is refused (monotonic, CAP-DERIVE-1), all-or-nothing.
        assert_eq!(
            grant(
                &mut client,
                1,
                &mut server,
                4,
                Rights::ALL,
                0,
                GrantMode::Move
            ),
            Err(CapError::InsufficientRights)
        );
        assert!(
            client.resolve(1).is_ok(),
            "refused grant left the source untouched"
        );
        assert_eq!(
            server.resolve(4),
            Err(CapError::EmptySlot),
            "refused grant placed nothing"
        );
    }

    #[test]
    fn grant_is_all_or_nothing_on_full_destination() {
        // If the destination slot is occupied, a MOVE-grant must NOT vacate the source — the
        // all-or-nothing property the in-flight-REVOKE atomicity leans on.
        let mut client = boot_with_frame::<8>();
        let mut server = boot_with_frame::<8>(); // server slot 1 occupied
        assert_eq!(
            grant(
                &mut client,
                1,
                &mut server,
                1,
                Rights::READ,
                0,
                GrantMode::Move
            ),
            Err(CapError::SlotOccupied)
        );
        assert!(
            client.resolve(1).is_ok(),
            "occupied-dest grant left the source intact"
        );
    }

    #[test]
    fn grant_refuses_reply_and_ungrantable() {
        let mut a = CSpace::<8>::new(nz);
        let mut b = CSpace::<8>::new(nz);
        // A Reply is never transferable (bound to one caller).
        a.mint_reply(1, 5, 0).unwrap();
        assert_eq!(
            grant(&mut a, 1, &mut b, 2, Rights::empty(), 0, GrantMode::Move),
            Err(CapError::InsufficientRights)
        );
        // A cap without the GRANT right can't be transferred (Untyped has RETYPE only).
        let mut c = CSpace::<8>::new(nz);
        c.set_root_untyped(0, 64);
        assert_eq!(
            grant(&mut c, 0, &mut b, 2, Rights::RETYPE, 0, GrantMode::Move),
            Err(CapError::InsufficientRights)
        );
        assert!(
            c.resolve(0).is_ok(),
            "refused grant left the untyped intact"
        );
    }

    #[test]
    fn grant_then_revoke_source_resolves_cleanly() {
        // In-flight atomicity shape (CAP-REVOKE-1) at the logic level: after a MOVE-grant, the
        // source is empty, so a source-side REVOKE finds nothing to reach and the receiver keeps
        // its cap — whichever order runs, the state is consistent (never a half-transferred cap).
        let mut client = boot_with_frame::<8>();
        let mut server = CSpace::<8>::new(nz);
        grant(
            &mut client,
            1,
            &mut server,
            2,
            Rights::READ,
            0,
            GrantMode::Move,
        )
        .unwrap();
        // A revoke of the client's (now-empty) source slot fails cleanly, not UB.
        assert_eq!(client.revoke(1), Err(CapError::EmptySlot));
        // A revoke of the client's untyped root does NOT reach the moved cap (it left the tree).
        client.revoke(0).unwrap();
        assert!(
            server.resolve(2).is_ok(),
            "the granted (moved) cap survives a source revoke"
        );
    }

    #[test]
    fn grant_move_refuses_a_cap_with_local_children() {
        // MOVE-grant of a cap that has MDB children would orphan them (they live in the source
        // CSpace and can't cross) — refuse it. MINT-grant (source kept) is fine.
        let mut client = boot_with_frame::<8>(); // slot 1 = Frame (RW|MAP|GRANT|DERIVE)
        client.mint(1, 2, Rights::READ, 0).unwrap(); // derive a child of the frame at slot 2
        let mut server = CSpace::<8>::new(nz);
        assert_eq!(
            grant(
                &mut client,
                1,
                &mut server,
                3,
                Rights::READ,
                0,
                GrantMode::Move
            ),
            Err(CapError::InsufficientRights)
        );
        assert!(
            client.resolve(1).is_ok(),
            "refused MOVE left the source + child intact"
        );
        assert!(client.resolve(2).is_ok());
        assert_eq!(
            server.resolve(3),
            Err(CapError::EmptySlot),
            "nothing placed"
        );
        // MINT-grant keeps the source, so no orphaning — allowed.
        grant(
            &mut client,
            1,
            &mut server,
            3,
            Rights::READ,
            0,
            GrantMode::Mint,
        )
        .unwrap();
        assert!(client.resolve(1).is_ok());
    }

    #[test]
    fn revoke_racing_grant_is_atomic_in_both_orderings() {
        // CAP-REVOKE-1 in-flight atomicity — the explicit race test. On a single CPU the masked
        // grant and a REVOKE cannot interleave; this asserts that WHICHEVER wins, the state is
        // consistent (never a half-transferred or duplicated cap).
        // (1) REVOKE-then-GRANT: the source is gone ⇒ the grant fails cleanly, forging nothing.
        {
            let mut client = boot_with_frame::<8>();
            let mut server = CSpace::<8>::new(nz);
            client.revoke(0).unwrap(); // revoke the untyped ⇒ destroys the frame at slot 1
            assert_eq!(client.resolve(1), Err(CapError::EmptySlot));
            assert_eq!(
                grant(
                    &mut client,
                    1,
                    &mut server,
                    2,
                    Rights::READ,
                    0,
                    GrantMode::Move
                ),
                Err(CapError::EmptySlot),
                "grant after revoke finds nothing — no half-transferred cap"
            );
            assert_eq!(server.resolve(2), Err(CapError::EmptySlot));
        }
        // (2) GRANT-then-REVOKE: the MOVE'd cap left the source tree ⇒ a source revoke can't
        // reclaim it, and exactly one consistent copy exists (in the receiver).
        {
            let mut client = boot_with_frame::<8>();
            let mut server = CSpace::<8>::new(nz);
            grant(
                &mut client,
                1,
                &mut server,
                2,
                Rights::READ,
                0,
                GrantMode::Move,
            )
            .unwrap();
            client.revoke(0).unwrap();
            assert!(
                server.resolve(2).is_ok(),
                "granted cap survives a post-grant source revoke"
            );
            assert_eq!(client.resolve(1), Err(CapError::EmptySlot));
        }
    }
}
