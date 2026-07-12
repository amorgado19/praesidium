//! P5 Layer-1 (SASOS type isolation, AC5.1) — the compile-time nameability proof.
//!
//! ADR-0008 Layer 1: within the kernel, cross-component memory access is gated by the capability
//! *types* — a component with no `Cap<Frame,_>` to a region has no *safe* way to name it, because
//! `Cap<T>` cannot be constructed outside `cap-core` (its fields are private and its one
//! `unsafe fn fabricate` is `pub(crate)`). This is CAP-RUST-1 enforced by the type system, not by
//! audit. These compile-fail fixtures mechanically regression-guard that guarantee: if a future
//! refactor ever opened a construction path, the build would break here.
//!
//! `.stderr` snapshots are toolchain-version-sensitive (regenerate with `TRYBUILD=overwrite` on a
//! toolchain bump). Pinned nightly keeps them stable.

#[test]
fn cap_is_unconstructable_outside_cap_core() {
    trybuild::TestCases::new().compile_fail("tests/ui/*.rs");
}
