//! aarch64 backend of the ADR-0007 arch seam.
//!
//! PL011 serial, CPU halt, and a full-system data barrier (`dsb sy`). All
//! arch-specific `unsafe` lives here with `// SAFETY:` invariants (DEC-0007-6).
//! Barriers are explicit seam methods, never assumed — the Warden aarch64 lesson
//! encoded structurally (DEC-0007-4).

use core::arch::asm;
use core::ptr::{read_volatile, write_volatile};

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
