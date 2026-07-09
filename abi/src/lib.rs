//! # abi — the kernel⇄userspace contract (ADR-0006)
//!
//! Defines Praesidium's capability-invocation syscall ABI and the `.pex`
//! executable format (including the capability manifest) — the stable seam both
//! the kernel and userspace build against.
//!
//! **In P0 it is deliberately empty.** No syscalls or executable format exist
//! until P6. Kept `#![no_std]` so both sides can depend on it.
//!
//! Note: this is *Praesidium's own* ABI. It is distinct from the frozen
//! `warden-abi` handoff contract, which the kernel vendors separately in
//! `kernel/src/boot/handoff.rs`.
#![no_std]
