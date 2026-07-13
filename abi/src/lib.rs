//! # abi — the kernel⇄userspace contract (ADR-0006)
//!
//! Defines Praesidium's capability-invocation syscall ABI and the `.pex`
//! executable format (including the capability manifest) — the stable seam both
//! the kernel and userspace build against.
//!
//! Two surfaces (P6): the **invocation syscall ABI** ([`invoke`]) — how userspace invokes the
//! kernel, carrying cptrs + rights rather than raw addresses — and the **`.pex` executable
//! format** ([`pex`], [`encode`]) — how a process is packaged and how its *initial* capabilities
//! are declared (the capability manifest). Both are pure wire contracts: this crate has no
//! `cap-core` dependency, so the manifest carries raw wire ints that the kernel validates and
//! maps to real capability types (treating every `.pex` as hostile input, GC-03).
//!
//! Note: this is *Praesidium's own* ABI. It is distinct from the frozen `warden-abi` handoff
//! contract, which the kernel vendors separately in `kernel/src/boot/handoff.rs`.
#![no_std]

pub mod encode;
pub mod invoke;
pub mod pex;
