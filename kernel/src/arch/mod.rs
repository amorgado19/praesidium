//! The ADR-0007 architecture seam.
//!
//! Everything above this module is arch-generic; each backend below exposes an
//! identical surface. No `#[cfg(target_arch)]` appears above this file (AR-AC1) —
//! all divergence lives here (DEC-0007-1).
//!
//! **Surface so far:** serial ([`serial_init`], [`serial_write_byte`]), CPU control
//! ([`halt`], [`memory_barrier`]), the `VSpace`/translation primitives (P1:
//! [`build_address_space`], [`activate_address_space`], [`translate`], [`page_prot`]),
//! and — added in P3b — the execution primitives: interrupt control
//! ([`interrupts_init`], [`timer_init`], [`preempt_disable`]/[`preempt_restore`]/
//! [`preempt_enable`], [`wait_for_interrupt`]) and the stackful context switch
//! ([`Context`], [`context_init`], [`context_switch`]). The seam grows one method at a
//! time, per phase (DEC-0007-2); I/D cache-maintenance lands with the phase that first
//! copies-then-executes code (P6).

/// Memory protection for a mapping. **W^X is structural (CAP-MEM-1):** there is no
/// writable-and-executable variant, so a W+X mapping is *unrepresentable* — the
/// strongest possible form of "refuse W+X".
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Prot {
    /// Read-only, non-executable.
    Ro,
    /// Read-write, non-executable.
    Rw,
    /// Read + execute, non-writable.
    Rx,
}

impl Prot {
    /// Build a protection from raw read/write/execute intents, **refusing** a
    /// writable+executable combination (W^X) and any non-readable mapping. Returns
    /// `None` for a refused request.
    #[must_use]
    pub fn checked(read: bool, write: bool, exec: bool) -> Option<Prot> {
        match (read, write, exec) {
            (true, false, false) => Some(Prot::Ro),
            (true, true, false) => Some(Prot::Rw),
            (true, false, true) => Some(Prot::Rx),
            _ => None,
        }
    }
}

/// Arch-generic description of the address space the kernel builds for itself. The
/// backend renders this into its native page-table format. The layout mirrors
/// Warden's handoff (identity + HHDM + kernel) so the boot stack and current PC stay
/// mapped across the switch; the refinement is per-section W^X on the kernel image.
pub struct KernelMap {
    /// Warden's higher-half direct-map base.
    pub hhdm_offset: u64,
    /// Bytes of low physical memory to identity- and HHDM-map.
    pub identity_bytes: u64,
    /// Kernel image virtual base (link address).
    pub kernel_vbase: u64,
    /// Kernel image virtual end (page-aligned).
    pub kernel_vend: u64,
    /// Physical base the kernel image is loaded at.
    pub kernel_phys: u64,
    /// `.text` `[start, end)` virtual range → mapped R-X.
    pub text: (u64, u64),
    /// `.rodata` `[start, end)` virtual range → mapped R--.
    pub rodata: (u64, u64),
    /// `.data`+`.bss` `[start, end)` virtual range → mapped RW-.
    pub data: (u64, u64),
}

/// Opaque handle to a built address space (root table physical addresses). The number
/// of roots is arch-specific — x86-64 uses one (`primary` = CR3/PML4); aarch64 uses
/// two (`primary` = TTBR1 high half, `secondary` = TTBR0 low half) — so treat this as
/// opaque above the seam. `secondary` is therefore read only on the arch that needs it.
#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct AddressSpace {
    pub primary: u64,
    pub secondary: u64,
}

#[cfg(target_arch = "x86_64")]
mod x86_64;
#[cfg(target_arch = "x86_64")]
pub use x86_64::{
    activate_address_space, build_address_space, context_init, context_switch, enable_wx, halt,
    interrupts_init, memory_barrier, page_prot, preempt_disable, preempt_enable, preempt_restore,
    read_translation_root, serial_init, serial_write_byte, timer_init, translate,
    wait_for_interrupt, Context,
};

#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(target_arch = "aarch64")]
pub use aarch64::{
    activate_address_space, build_address_space, context_init, context_switch, enable_wx, halt,
    interrupts_init, memory_barrier, page_prot, preempt_disable, preempt_enable, preempt_restore,
    read_translation_root, serial_init, serial_write_byte, timer_init, translate,
    wait_for_interrupt, Context,
};

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
compile_error!("Praesidium supports x86-64 and aarch64 only (ADR-0001 DEC-003 / ADR-0007).");
