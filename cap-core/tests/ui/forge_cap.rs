//! AC5.1: constructing a `Cap<Frame>` from OUTSIDE cap-core must NOT compile — there is no way to
//! *name* a region without a legitimately-obtained capability. Both forge routes are refused:
//! the `fabricate` primitive is `pub(crate)`, and the struct's fields are private.
use cap_core::cap::{Cap, Frame, RawCap};
use core::marker::PhantomData;

fn main() {
    // Route 1: the sole construction primitive is `pub(crate)` — not reachable from another crate.
    let _via_fabricate: Cap<Frame> = unsafe { Cap::<Frame>::fabricate(RawCap::NULL) };

    // Route 2: the struct's fields are private — a struct literal cannot name them.
    let _via_fields: Cap<Frame> = Cap {
        raw: RawCap::NULL,
        _marker: PhantomData,
    };
}
