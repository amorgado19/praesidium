//! # mem — Praesidium physical-memory allocator logic (ADR-0002, P1)
//!
//! Arch-independent, `no_std` allocator math that operates over **caller-provided
//! storage**, so the same logic runs two ways:
//! - in the kernel, over HHDM-mapped physical memory (a frame-descriptor array and
//!   byte regions bootstrapped from the Warden memory map);
//! - on the host, over plain `Vec`s, so `cargo test -p mem` exercises split/coalesce,
//!   slab freelists, and retype accounting with no bare-metal target involved.
//!
//! It deliberately contains **no arch code, no `asm!`, no entry point** — the kernel
//! integration layer (`kernel/src/mem/`) wires these into HHDM, zero-on-retype, W^X,
//! and the arch page-table seam. Per P1's scope, this is the allocator *mechanism*;
//! the capability formalization of `Untyped`/RETYPE (`Cap<T, Rights>`, CSpace, MDB)
//! is P2.
#![cfg_attr(not(test), no_std)]

pub mod buddy;
pub mod frame;
pub mod retype;
pub mod slab;

pub use frame::{Pfn, PAGE_SHIFT, PAGE_SIZE};
