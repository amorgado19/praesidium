//! The capability representation and the single typed wrapper (SPEC-CAP §2, §4).
//!
//! Two forms, per SPEC-CAP §4's "two layers that must agree":
//!  - [`RawCap`] — the **runtime** record stored in CSpace slots (object type + rights +
//!    object reference + badge); the ground truth checked on every operation, valid even
//!    for non-Rust userspace.
//!  - [`Cap<T>`] — the **compile-time** typed wrapper the kernel's own Rust code holds:
//!    object-type-safe (you cannot name a resource without the right `Cap<T>`), non-`Copy`
//!    (CAP-RUST-2), and fabricable ONLY via [`Cap::fabricate`] — the sole `unsafe`
//!    capability-fabrication point in the whole kernel (CAP-RUST-1).

use core::marker::PhantomData;

use crate::rights::Rights;

/// The object class a capability names — it fixes *what operations exist at all*
/// (SPEC-CAP §2.1). `Null` marks an empty CSpace slot.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum CapType {
    Null = 0,
    Untyped,
    Frame,
    CNode,
    Endpoint,
    Notification,
    Task,
    Sched,
    Reply,
    AddrSpace,
    VSpace,
    FNode,
    Device,
    IrqControl,
}

/// The type-erased capability record (SPEC-CAP §4 layer 1). Not a pointer: `objref` is an
/// opaque, kernel-assigned object id (e.g. a physical frame number for `Frame`/`Untyped`),
/// never an address a holder can dereference. `size`/`watermark`/`aux` carry the
/// **type-specific** state (their meaning depends on `cap_type`):
///  - `Frame`/`CNode`/`Endpoint`: `size` = frames it occupies.
///  - `Untyped`: `size` = total frames, `watermark` = frames consumed by RETYPE.
///  - `Sched` (P3): `size` = CPU-time **budget** (units/period), `watermark` = units
///    **consumed** this period, `aux` = replenishment **period**. Budget lives *in the cap*
///    (per-cap, like the Untyped watermark), so `Sched` is non-duplicable — a COPY/MINT
///    would fork the budget = CPU-time forgery (DEC-0003-2, mirrors the Untyped invariant).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RawCap {
    pub cap_type: CapType,
    pub rights: Rights,
    /// Object reference (frame number for Frame/Untyped; opaque object id otherwise).
    pub objref: u64,
    /// Type-specific: frames occupied (Frame/CNode/Endpoint), total frames (Untyped), or
    /// CPU-time budget (Sched).
    pub size: u32,
    /// Type-specific: frames consumed by RETYPE (Untyped) or CPU-time consumed this period
    /// (Sched). Zero for other types.
    pub watermark: u32,
    /// Type-specific auxiliary state. Sched: the replenishment period. Unused (0) otherwise.
    pub aux: u32,
    /// Provenance stamp set by MINT (SPEC-CAP §2); distinguishes callers, scopes revoke.
    pub badge: u64,
}

impl RawCap {
    /// The empty-slot record.
    pub const NULL: Self = Self {
        cap_type: CapType::Null,
        rights: Rights::empty(),
        objref: 0,
        size: 0,
        watermark: 0,
        aux: 0,
        badge: 0,
    };

    /// Is this slot empty?
    #[must_use]
    pub fn is_null(&self) -> bool {
        matches!(self.cap_type, CapType::Null)
    }
}

/// Marker trait tying a zero-sized object-type marker to its [`CapType`] tag. Implemented
/// only by the marker types below, so `Cap<T>` can only exist for a real object type.
pub trait ObjectType {
    const TYPE: CapType;
}

/// Declare the zero-sized object-type markers + their `ObjectType` impls.
macro_rules! object_types {
    ($($name:ident => $tag:ident),+ $(,)?) => {
        $(
            /// Object-type marker (see [`CapType`]).
            #[derive(Debug)]
            pub struct $name;
            impl ObjectType for $name {
                const TYPE: CapType = CapType::$tag;
            }
        )+
    };
}

object_types! {
    Untyped => Untyped,
    Frame => Frame,
    CNode => CNode,
    Endpoint => Endpoint,
    Notification => Notification,
    Task => Task,
    Sched => Sched,
    Reply => Reply,
    AddrSpace => AddrSpace,
    VSpace => VSpace,
    FNode => FNode,
    Device => Device,
    IrqControl => IrqControl,
}

/// A typed, non-`Copy` capability handle. Holding a `Cap<Frame>` is the only way to name a
/// frame; its rights are checked at runtime against the inner [`RawCap`]. Constructing one
/// requires [`Cap::fabricate`], the sole `unsafe` fabrication surface (CAP-RUST-1). Not
/// `Copy` (CAP-RUST-2) — duplication goes through COPY/MINT so provenance stays explicit.
pub struct Cap<T: ObjectType> {
    raw: RawCap,
    _marker: PhantomData<T>,
}

impl<T: ObjectType> Cap<T> {
    /// Fabricate a typed capability from a raw record. **This is THE trusted primitive:**
    /// every capability in the kernel ultimately originates here, and this is the only
    /// place `unsafe` constructs a capability (CAP-RUST-1). All derivation
    /// (RETYPE/MINT/COPY) funnels through it inside `cap-core`.
    ///
    /// # Safety
    /// The caller MUST ensure `raw` is a genuine, kernel-authorized capability record —
    /// produced by a legitimate derivation or the primordial bootstrap — and that
    /// `raw.cap_type == T::TYPE`. Fabricating a capability for an object the caller has no
    /// authority over forges authority and breaks the Root Invariant.
    #[must_use]
    pub unsafe fn fabricate(raw: RawCap) -> Self {
        debug_assert!(
            raw.cap_type as u8 == T::TYPE as u8,
            "cap type mismatch in fabricate"
        );
        Self {
            raw,
            _marker: PhantomData,
        }
    }

    /// The rights this capability carries.
    #[must_use]
    pub fn rights(&self) -> Rights {
        self.raw.rights
    }

    /// The object this capability names.
    #[must_use]
    pub fn objref(&self) -> u64 {
        self.raw.objref
    }

    /// The provenance badge (0 if unset).
    #[must_use]
    pub fn badge(&self) -> u64 {
        self.raw.badge
    }

    /// Does this capability permit every right in `r`?
    #[must_use]
    pub fn allows(&self, r: Rights) -> bool {
        self.raw.rights.contains(r)
    }

    /// The type-erased record (the form stored in a CSpace slot). Carrying a `RawCap`
    /// grants no authority on its own — nothing can be *done* with it without fabricating
    /// a `Cap<T>` (which only `cap-core` does).
    #[must_use]
    pub fn to_raw(&self) -> RawCap {
        self.raw
    }
}
