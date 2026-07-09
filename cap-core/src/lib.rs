//! # cap-core — the capability trust root (SPEC-CAP CAP-RUST-1)
//!
//! The runtime + compile-time layers of the Praesidium capability model: the capability
//! record ([`RawCap`]), the typed wrapper ([`Cap<T>`]) with the **single** `unsafe`
//! fabrication point, capability rights ([`Rights`]), and the capability space
//! ([`CSpace`]) with its derivation tree (MDB) and the RETYPE / MINT / COPY / MOVE /
//! REVOKE / DELETE operations.
//!
//! This crate is the trust root: **any `unsafe` bug here is a total capability-model
//! bypass**, so it is kept small and exhaustively auditable (CAP-RUST-1) — the *only*
//! `unsafe` capability fabrication in the whole kernel is [`Cap::fabricate`]. Everything
//! else is safe code that funnels through it. It is pure logic over abstract storage, so
//! `cargo test -p cap-core` exercises the derivation tree, revoke transitivity, and rights
//! monotonicity on the host; the kernel (`kernel/src/cap/`) wires RETYPE to P1's physical
//! allocator + HHDM zeroing.
#![cfg_attr(not(test), no_std)]

pub mod cap;
pub mod cspace;
pub mod rights;

pub use cap::{Cap, CapType, ObjectType, RawCap};
pub use cspace::{CSpace, CapError, Cptr};
pub use rights::Rights;
