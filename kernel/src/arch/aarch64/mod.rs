//! aarch64 backend of the ADR-0007 arch seam.
//!
//! PL011 serial, CPU halt, and a full-system data barrier (`dsb sy`). All
//! arch-specific `unsafe` lives here with `// SAFETY:` invariants (DEC-0007-6).
//! Barriers are explicit seam methods, never assumed — the Warden aarch64 lesson
//! encoded structurally (DEC-0007-4).

use core::arch::asm;
use core::ptr::{read_volatile, write_volatile};

mod context;
mod interrupts;
mod mte;
mod paging;
mod timer;
mod user;
pub use context::{context_init, context_switch, Context};
pub use interrupts::{contains_raw_read, interrupts_init};
pub use mte::domain_escape_contained;
pub use user::{el0_fault_blob, el0_supported, el0_test_blob, enter_user, set_kernel_stack};

/// The `.pex` architecture tag for this backend (ADR-0006) — see the x86-64 backend.
pub const PEX_ARCH: u16 = abi::pex::ARCH_AARCH64;
pub use paging::{
    activate_address_space, build_address_space, enable_wx, install_guard_page, kernel_space,
    map_page, map_user_page, new_process_space, page_prot, set_domain, sync_instruction_cache,
    translate,
};

/// The active process-vs-process **hostile-isolation** mechanism (P7b-ii, ISO-AC4 honesty). The red
/// team proved MTE is userspace-defeatable (a process controls its own pointer tags — it can forge a
/// victim's 4-bit tag), so the boundary that actually contains a hostile process is the per-domain
/// page table (`TTBR0`): the victim's pages are not mapped in the attacker's table, so the access
/// permission-faults regardless of the forged tag. MTE stays armed as the cooperative/defence layer.
#[must_use]
pub fn isolation_mechanism() -> &'static str {
    "aarch64 per-domain page tables (ADR-0008 DEC-0008-6); MTE = cooperative/defence-in-depth layer"
}

/// Arm the process-vs-process isolation mechanism before userspace runs (P7b-ii). On aarch64 this
/// enables synchronous EL0+EL1 MTE tag checking + the Tagged MAIR attr ([`mte::enable`]) so the
/// per-process granule tags set by [`paging::map_user_page`] are enforced. Harmless before any
/// tagged page exists (the kernel's own pages are untagged). x86 arms PKU earlier (at `gdt_init`).
pub fn isolation_init() {
    mte::enable();
}
pub use timer::timer_init;

/// Mask IRQs (disable preemption), returning whether they were enabled before — pass that back
/// to [`preempt_restore`] to nest correctly. `DAIF.I` is bit 7 when read via `mrs`.
#[must_use]
pub fn preempt_disable() -> bool {
    let daif: u64;
    // SAFETY: read DAIF, then mask IRQ via `daifset #2` (bit 1 = I). No memory/stack effects.
    unsafe {
        asm!("mrs {d}, daif", d = out(reg) daif, options(nomem, nostack, preserves_flags));
        asm!("msr daifset, #2", options(nomem, nostack, preserves_flags));
    }
    daif & (1 << 7) == 0 // I clear ⇒ IRQs were enabled
}

/// Re-enable IRQs iff they were enabled when [`preempt_disable`] was called.
pub fn preempt_restore(was_enabled: bool) {
    if was_enabled {
        // SAFETY: `daifclr #2` clears DAIF.I, re-enabling IRQs — only what was on before.
        unsafe { asm!("msr daifclr, #2", options(nomem, nostack, preserves_flags)) };
    }
}

/// Unconditionally unmask IRQs (a freshly-launched task becomes preemptible).
pub fn preempt_enable() {
    // SAFETY: `daifclr #2` clears DAIF.I, enabling IRQ delivery.
    unsafe { asm!("msr daifclr, #2", options(nomem, nostack, preserves_flags)) };
}

/// Unmask IRQs and wait for one — the idle path.
pub fn wait_for_interrupt() {
    // SAFETY: unmask IRQs then `wfi` parks the CPU until an interrupt (e.g. the timer) fires.
    unsafe {
        asm!(
            "msr daifclr, #2",
            "wfi",
            options(nomem, nostack, preserves_flags)
        )
    };
}

/// Read the active address-space roots: `[TTBR1_EL1, TTBR0_EL1]` (high-half + low-half). Used by
/// P4 to assert an IPC fast-path call performs **no address-space swap** (AC4.4) — the SASOS win:
/// with one address space the roots are invariant across a call, so no TLB flush / TTBR reload.
#[must_use]
pub fn read_translation_root() -> [u64; 2] {
    let (t1, t0): (u64, u64);
    // SAFETY: reading TTBR1_EL1 / TTBR0_EL1 is side-effect-free.
    unsafe {
        asm!("mrs {}, ttbr1_el1", out(reg) t1, options(nomem, nostack, preserves_flags));
        asm!("mrs {}, ttbr0_el1", out(reg) t0, options(nomem, nostack, preserves_flags));
    }
    [t1, t0]
}

/// ELF entry from Warden (`x0` = the `WardenBootInfo` pointer). Switch to the kernel's
/// own boot stack (Warden's stack is in allocator-managed RAM), then tail-call
/// [`crate::kmain`] with `x0` untouched.
#[no_mangle]
#[unsafe(naked)]
extern "C" fn _start() -> ! {
    core::arch::naked_asm!(
        "adrp x9, {stack}",
        "add x9, x9, :lo12:{stack}",
        "mov x10, {size}",
        "add x9, x9, x10",
        "mov sp, x9",
        "bl {main}",
        "brk #0",
        stack = sym crate::BOOT_STACK,
        size = const crate::BOOT_STACK_SIZE,
        main = sym crate::kmain,
    );
}

/// PL011 data register on the QEMU `virt` machine. Warden maps this MMIO window
/// as Device memory in both the TTBR0 identity map and the TTBR1 HHDM, so the
/// fixed physical address is reachable at kernel entry.
const PL011_DR: *mut u8 = 0x0900_0000 as *mut u8;
/// PL011 flag register (data register + 0x18).
const PL011_FR: *const u32 = 0x0900_0018 as *const u32;
/// Flag register bit: transmit FIFO full.
const FR_TXFF: u32 = 1 << 5;
/// Maximum TX-ready polls before dropping the byte. A wedged/absent UART must not
/// hang the kernel forever — the panic handler transmits through here before it
/// halts, so an unbounded spin would swallow the loud-failure marker (the Warden
/// serial-backend lesson).
const TX_SPIN_CAP: u32 = 1_000_000;

/// PL011 needs no software bring-up under QEMU: the firmware leaves it enabled
/// and the default baud is fine over `-serial stdio`. Kept as a seam method so
/// both backends expose an identical surface (and real-hardware init has a home).
pub fn serial_init() {}

/// Emit one byte, blocking until the transmit FIFO has room (bounded by
/// [`TX_SPIN_CAP`] — the byte is dropped rather than spinning forever).
pub fn serial_write_byte(byte: u8) {
    // SAFETY: PL011 DR/FR are Device MMIO at these fixed addresses on the QEMU
    // `virt` machine (mapped by Warden); volatile access, no other memory effects.
    unsafe {
        let mut spins = 0u32;
        while read_volatile(PL011_FR) & FR_TXFF != 0 {
            spins += 1;
            if spins >= TX_SPIN_CAP {
                return;
            }
        }
        write_volatile(PL011_DR, byte);
    }
}

/// Full-system data barrier: order all prior memory/MMIO accesses before any that
/// follow, at the CPU AND at the compiler. The absence of `nomem` is deliberate —
/// it gives the asm an implicit memory clobber, which is what makes it a *compiler*
/// barrier as well as a hardware one (DEC-0007-4).
pub fn memory_barrier() {
    // SAFETY: `dsb sy` is a full data-synchronization barrier; without `nomem` the
    // block also acts as a compiler memory barrier. No stack/flag effects.
    unsafe { asm!("dsb sy", options(nostack, preserves_flags)) };
}

/// Mask exceptions and park the CPU forever — the P0 end state.
pub fn halt() -> ! {
    // SAFETY: `msr daifset, #0xf` masks D/A/I/F exceptions; in P0 none are configured.
    unsafe {
        asm!(
            "msr daifset, #0xf",
            options(nomem, nostack, preserves_flags)
        )
    };
    loop {
        // SAFETY: `wfi` waits for an event/interrupt; with exceptions masked this
        // parks the CPU. No memory effects.
        unsafe { asm!("wfi", options(nomem, nostack, preserves_flags)) };
    }
}
