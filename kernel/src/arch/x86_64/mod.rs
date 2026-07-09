//! x86-64 backend of the ADR-0007 arch seam.
//!
//! COM1 serial, CPU halt, and a full memory fence. All arch-specific `unsafe`
//! lives here, each block carrying a `// SAFETY:` invariant (DEC-0007-6). Nothing
//! above the seam uses `#[cfg(target_arch)]`.

use core::arch::asm;

mod paging;
pub use paging::{activate_address_space, build_address_space, enable_wx, page_prot, translate};

/// ELF entry from Warden (`rdi` = the `WardenBootInfo` pointer). We switch to the
/// kernel's own boot stack — Warden's stack is in allocator-managed RAM, so we must
/// leave it before the frame allocator runs — then tail-call [`crate::kmain`] with
/// `rdi` untouched. `BOOT_STACK` is 16-aligned and `call` pushes 8, so `kmain` sees
/// the ABI-required `rsp ≡ 8 (mod 16)`.
#[no_mangle]
#[unsafe(naked)]
extern "C" fn _start() -> ! {
    core::arch::naked_asm!(
        "lea rsp, [rip + {stack}]",
        "add rsp, {size}",
        "xor ebp, ebp",
        "call {main}",
        "ud2",
        stack = sym crate::BOOT_STACK,
        size = const crate::BOOT_STACK_SIZE,
        main = sym crate::kmain,
    );
}

/// COM1 base I/O port — the UART QEMU exposes on `-serial stdio` for x86-64.
const COM1: u16 = 0x3F8;
/// Line Status Register (COM1 + 5).
const COM1_LSR: u16 = COM1 + 5;
/// LSR bit: transmit-holding register empty (ready to accept a byte).
const LSR_THR_EMPTY: u8 = 0x20;

/// Write one byte to an I/O port.
///
/// # Safety
/// The caller must ensure `port` names a valid I/O port whose write side effects
/// are understood. Used here only for the COM1 16550 UART registers.
#[inline]
unsafe fn outb(port: u16, value: u8) {
    // SAFETY: `out` writes a single byte to an I/O port and touches no memory.
    unsafe {
        asm!("out dx, al", in("dx") port, in("al") value, options(nomem, nostack, preserves_flags))
    };
}

/// Read one byte from an I/O port.
///
/// # Safety
/// The caller must ensure `port` names a valid I/O port whose read side effects
/// are understood.
#[inline]
unsafe fn inb(port: u16) -> u8 {
    let value: u8;
    // SAFETY: `in` reads a single byte from an I/O port and touches no memory.
    unsafe {
        asm!("in al, dx", out("al") value, in("dx") port, options(nomem, nostack, preserves_flags))
    };
    value
}

/// Bring up COM1 at 115200 8N1 with FIFOs enabled (the standard 16550 sequence).
pub fn serial_init() {
    // SAFETY: the canonical 16550 initialization sequence; every write targets a
    // known COM1 UART register and has no memory effects.
    unsafe {
        outb(COM1 + 1, 0x00); // IER: disable all UART interrupts
        outb(COM1 + 3, 0x80); // LCR: enable DLAB to program the baud divisor
        outb(COM1, 0x01); //     DLL: divisor low  = 1  -> 115200 baud
        outb(COM1 + 1, 0x00); // DLM: divisor high = 0
        outb(COM1 + 3, 0x03); // LCR: 8 bits, no parity, 1 stop; clear DLAB
        outb(COM1 + 2, 0xC7); // FCR: enable + clear FIFOs, 14-byte trigger level
        outb(COM1 + 4, 0x0B); // MCR: DTR, RTS, OUT2
    }
}

/// Maximum TX-ready polls before dropping the byte. A wedged/absent UART must not
/// hang the kernel forever — the panic handler transmits through here before it
/// halts, so an unbounded spin would swallow the loud-failure marker (the Warden
/// serial-backend lesson).
const TX_SPIN_CAP: u32 = 1_000_000;

/// Emit one byte, blocking until the transmit holding register is free (bounded by
/// [`TX_SPIN_CAP`] — the byte is dropped rather than spinning forever).
pub fn serial_write_byte(byte: u8) {
    // SAFETY: poll LSR then write the transmit register — both COM1 UART ports.
    unsafe {
        let mut spins = 0u32;
        while inb(COM1_LSR) & LSR_THR_EMPTY == 0 {
            spins += 1;
            if spins >= TX_SPIN_CAP {
                return;
            }
        }
        outb(COM1, byte);
    }
}

/// Full memory fence: order all prior loads/stores before any that follow, at the
/// CPU AND at the compiler. The absence of `nomem` is deliberate — it gives the asm
/// an implicit memory clobber, which is what makes it a *compiler* barrier as well
/// as a hardware one (DEC-0007-4).
pub fn memory_barrier() {
    // SAFETY: `mfence` serializes memory operations; without `nomem` the block also
    // acts as a compiler memory barrier. No stack/flag effects.
    unsafe { asm!("mfence", options(nostack, preserves_flags)) };
}

/// Mask interrupts and park the CPU forever — the P0 end state.
pub fn halt() -> ! {
    // SAFETY: `cli` masks maskable interrupts; in P0 none are configured.
    unsafe { asm!("cli", options(nomem, nostack, preserves_flags)) };
    loop {
        // SAFETY: `hlt` pauses the CPU until an interrupt; with interrupts masked
        // this parks it. No memory effects.
        unsafe { asm!("hlt", options(nomem, nostack, preserves_flags)) };
    }
}
