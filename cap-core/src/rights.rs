//! Capability rights — a runtime bitflags set (SPEC-CAP §4 CAP-RUST-3, P2 decision).
//!
//! Rights are the subset of a type's operations a particular capability permits. They
//! are checked at runtime on every operation and only ever **narrow** along a
//! derivation (CAP-DERIVE-1) — the monotonicity that MINT/COPY enforce via
//! [`Rights::subset_of`].

use bitflags::bitflags;

bitflags! {
    /// The operations a capability permits. Each object type reads the bits relevant to
    /// it (e.g. a `Frame` cap uses READ/WRITE/EXECUTE/MAP/GRANT; an `Untyped` cap uses
    /// RETYPE). Rights are type-agnostic bits; widening one is impossible without MINT.
    #[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
    pub struct Rights: u32 {
        /// Read the object's memory (Frame, FNode).
        const READ = 1 << 0;
        /// Write the object's memory (Frame, FNode).
        const WRITE = 1 << 1;
        /// Map the frame executable (subject to W^X, CAP-MEM-1).
        const EXECUTE = 1 << 2;
        /// Install the frame into an address space (Frame, AddrSpace).
        const MAP = 1 << 3;
        /// Transfer the capability over IPC (Endpoint/Frame grant).
        const GRANT = 1 << 4;
        /// Turn `Untyped` memory into typed objects.
        const RETYPE = 1 << 5;
        /// Send on an endpoint.
        const SEND = 1 << 6;
        /// Receive on an endpoint.
        const RECV = 1 << 7;
        /// Signal a notification (raise its async signal).
        const SIGNAL = 1 << 8;
        /// Derive/duplicate capabilities from this one (MINT/COPY authority).
        const DERIVE = 1 << 9;
        /// Enter an isolation domain (P5, ADR-0008 DEC-0008-5). Held by a principal that may
        /// *run in* a domain — deliberately distinct from a `VSpace`'s MAP_TABLE (editing a
        /// domain's translation), so a principal can hold ENTER without ever being able to remap
        /// its own domain (which would be an isolation escape). Domain entry is never ambient.
        const ENTER = 1 << 10;
        /// Wait on a `Notification` (block until its async signal is raised — SPEC-CAP §2). Distinct
        /// from `RECV` (endpoint receive): a notification carries no payload, and the two authorities
        /// name different object classes.
        const WAIT = 1 << 11;
    }
}

impl Rights {
    /// An "owner" capability with every right (the primordial `Untyped` gets this).
    pub const ALL: Self = Self::all();

    /// Is `self` no wider than `parent`? MINT/COPY refuse a derivation that fails this
    /// (CAP-DERIVE-1: rights are non-increasing along a derivation).
    #[must_use]
    pub fn subset_of(self, parent: Rights) -> bool {
        parent.contains(self)
    }
}
