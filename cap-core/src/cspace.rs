//! The capability space (CSpace) + derivation tree (MDB) + operations (SPEC-CAP §3, §5, §7).
//!
//! P2 uses a **single-level** CSpace: one root `CNode` of `N` slots, a cptr is a slot index
//! (an index path, never a memory pointer — §3), and the MDB is a parent link per slot. All
//! operations are safe code funnelling through `cap-core`'s one `unsafe` fabrication point;
//! a revoked/empty slot resolves to an error, never UB (CAP-REVOKE-1).

use crate::cap::{Cap, CapType, ObjectType, RawCap};
use crate::rights::Rights;

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
}

impl<const N: usize> CSpace<N> {
    /// An empty CSpace. Install the primordial capability with [`set_root_untyped`](Self::set_root_untyped).
    #[must_use]
    pub fn new(zero: fn(u64, u32)) -> Self {
        Self {
            slots: [Cte::EMPTY; N],
            zero,
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
        if s.cap_type == CapType::Untyped {
            // An Untyped is never duplicable: its per-cap watermark would fork, forking the
            // budget and aliasing frames (§8/§2.1). Sub-allocation is a future SPLIT, not MINT.
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
        if s.cap_type == CapType::Untyped {
            return Err(CapError::InsufficientRights); // Untyped is never duplicable (see mint)
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
        self.slots[d] = Cte::EMPTY;
        // Zero the object if this was its last capability (CAP-REVOKE-2 no-residual). Untyped
        // memory is zeroed on the next RETYPE (CAP-MEM-2), so it needs no zeroing here.
        let another = (0..N).any(|k| {
            let c = self.slots[k].cap;
            !c.is_null() && c.cap_type as u8 == cap.cap_type as u8 && c.objref == cap.objref
        });
        if !another && cap.cap_type != CapType::Untyped {
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

/// Object types RETYPE can produce from `Untyped`. Excludes `Null` (the empty marker),
/// `Untyped` (no sub-retype in the single-level model — it would fork the watermark), and
/// `Reply`/`IrqControl` (kernel-minted, not retyped — CAP-REPLY-1).
fn is_retypeable(t: CapType) -> bool {
    !matches!(
        t,
        CapType::Null | CapType::Untyped | CapType::Reply | CapType::IrqControl
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
}
