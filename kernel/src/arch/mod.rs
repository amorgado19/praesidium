//! The ADR-0007 architecture seam.
//!
//! Everything above this module is arch-generic; each backend below exposes an
//! identical surface. No `#[cfg(target_arch)]` appears above this file (AR-AC1) —
//! all divergence lives here (DEC-0007-1).
//!
//! **P0 surface:** [`serial_init`], [`serial_write_byte`], [`halt`],
//! [`memory_barrier`]. The seam grows one method at a time, per phase, as new
//! arch behaviour is first exercised (DEC-0007-2): translation/`VSpace`, timers,
//! context switch, and I/D cache-maintenance arrive with the phases that need
//! them (P1/P3/P5/P6), so a never-forgotten barrier stays a *typed* obligation
//! rather than an assumption (DEC-0007-4).

#[cfg(target_arch = "x86_64")]
mod x86_64;
#[cfg(target_arch = "x86_64")]
pub use x86_64::{halt, memory_barrier, serial_init, serial_write_byte};

#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(target_arch = "aarch64")]
pub use aarch64::{halt, memory_barrier, serial_init, serial_write_byte};

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
compile_error!("Praesidium supports x86-64 and aarch64 only (ADR-0001 DEC-003 / ADR-0007).");
