//! x86-64 backend of the ADR-0007 arch seam.
//!
//! COM1 serial, CPU halt, and a full memory fence. All arch-specific `unsafe`
//! lives here, each block carrying a `// SAFETY:` invariant (DEC-0007-6). Nothing
//! above the seam uses `#[cfg(target_arch)]`.

use core::arch::asm;

mod context;
mod interrupts;
mod paging;
mod timer;
pub use context::{context_init, context_switch, Context};
pub use interrupts::{contains_raw_read, interrupts_init};
pub use paging::{
    activate_address_space, build_address_space, build_domain_excluding, enable_wx,
    install_guard_page, map_page, map_user_page, page_prot, sync_instruction_cache, translate,
};

/// Whether this backend can run ring-3 userspace yet. **False in P7a:** x86-64 ring 3 needs a GDT
/// (ring-3 segments) + a TSS (RSP0) + the `syscall`/`sysret` MSRs, built after the aarch64 EL0
/// path is validated. The generic [`crate::user`] path skips EL0 while this is false.
#[must_use]
pub fn el0_supported() -> bool {
    false
}

/// Drop to ring 3 — unbuilt on x86-64 in P7a.
///
/// # Safety
/// Never called while [`el0_supported`] returns false.
pub unsafe fn enter_user(_entry: u64, _user_sp: u64) -> ! {
    unreachable!("x86-64 ring-3 userspace is not wired yet (P7a builds aarch64 first)");
}

/// The bring-up user blob — none on x86-64 until ring 3 is wired.
#[must_use]
pub fn el0_test_blob() -> &'static [u8] {
    &[]
}
pub use timer::timer_init;

/// The `.pex` architecture tag for this backend (ADR-0006): a `.pex`'s segments are native code,
/// so the loader only accepts images tagged for the arch it runs on. Behind the seam so nothing
/// above it needs `#[cfg(target_arch)]`.
pub const PEX_ARCH: u16 = abi::pex::ARCH_X86_64;

/// Mask maskable interrupts (disable preemption), returning whether they were enabled before —
/// pass that back to [`preempt_restore`] to nest correctly.
#[must_use]
pub fn preempt_disable() -> bool {
    let flags: u64;
    // SAFETY: pushfq/pop reads RFLAGS onto the stack then into a GPR; cli clears IF. The block
    // uses the stack (no `nostack`); IF is not a condition-code flag, so `preserves_flags` holds.
    unsafe {
        asm!("pushfq", "pop {f}", "cli", f = out(reg) flags, options(preserves_flags));
    }
    flags & (1 << 9) != 0 // RFLAGS.IF
}

/// Re-enable interrupts iff they were enabled when [`preempt_disable`] was called.
pub fn preempt_restore(was_enabled: bool) {
    if was_enabled {
        // SAFETY: `sti` sets IF; we only re-enable what was on before this critical section.
        unsafe { asm!("sti", options(nomem, nostack, preserves_flags)) };
    }
}

/// Unconditionally enable interrupts (a freshly-launched task becomes preemptible).
pub fn preempt_enable() {
    // SAFETY: `sti` enables maskable interrupts. No memory/stack effects.
    unsafe { asm!("sti", options(nomem, nostack, preserves_flags)) };
}

/// Enable interrupts and halt until one arrives — the idle path (`sti; hlt` is atomic wrt the
/// interrupt window, so a wake that races the halt is not lost).
pub fn wait_for_interrupt() {
    // SAFETY: `sti; hlt` enables interrupts then parks the CPU until one fires.
    unsafe { asm!("sti", "hlt", options(nomem, nostack, preserves_flags)) };
}

/// Read the active address-space root(s): `[CR3, 0]` (x86-64 has a single root). Used by P4 to
/// assert an IPC fast-path call performs **no address-space swap** (AC4.4) — the SASOS win: with
/// one address space the root is invariant across a call, so no TLB flush / page-table reload.
#[must_use]
pub fn read_translation_root() -> [u64; 2] {
    let cr3: u64;
    // SAFETY: reading CR3 is side-effect-free.
    unsafe { asm!("mov {}, cr3", out(reg) cr3, options(nomem, nostack, preserves_flags)) };
    [cr3, 0]
}

/// Isolation Layer-2 escape red-team (P5b, DEC-0008-7): stand up an "attacker domain" in which a
/// victim **frame** is unreachable, switch into it, attempt raw reads of the frame, and report
/// whether the hardware contained them (`#PF`). Returns `true` iff *every* alias faulted (isolation
/// held); `false` if any raw read *succeeded* (a breach). Runs with preemption masked so no task is
/// scheduled on the transient domain CR3.
///
/// **Alias-honest.** This SASOS double-maps every allocatable frame (< 4 GiB): a low identity alias
/// (`phys`) *and* an HHDM alias (`phys + hhdm_offset`). Isolating a *frame* therefore means making
/// *both* inaccessible — unmapping only the HHDM alias would leave the secret trivially readable via
/// the identity alias (a false "contained"). So the identity alias is unmapped globally (the kernel
/// reaches heap frames only through the HHDM) and the domain additionally excludes the HHDM alias;
/// then *both* aliases are probed and both must fault.
///
/// x86's Layer-2 mechanism is the per-domain-page-table fallback (DEC-0008-6) — see
/// [`build_domain_excluding`]. The deliberate faults are contained by the single-shot recovery seam.
pub fn domain_escape_contained() -> bool {
    use crate::arch::AddressSpace;
    /// A recognizable value written through the kernel's HHDM mapping, confirmed intact afterwards.
    const SENTINEL: u64 = 0x5A5A_1508_D0D0_BEEF;

    let phys = crate::memory::alloc_frames(0).expect("no frame for the domain-escape victim");
    let hhdm = crate::memory::phys_to_virt(phys); // the HHDM alias (high half)
    let ident = phys; // the low-half identity alias — the second mapping SASOS gives every frame
                      // SAFETY: `hhdm` is a freshly-allocated, HHDM-mapped, writable frame the kernel owns.
    unsafe { (hhdm as *mut u64).write_volatile(SENTINEL) };

    // SAFETY: `ident` is this frame's identity alias; the kernel never reaches heap frames through
    // the identity map, so unmapping it globally is sound and leaves the HHDM alias (which the
    // kernel uses) untouched.
    unsafe { install_guard_page(ident) };

    let original = read_translation_root()[0];
    let domain = build_domain_excluding(hhdm);

    let prev = preempt_disable(); // no task may run on the transient domain CR3
                                  // SAFETY: `domain` clones the active PML4 (identity alias already unmapped) and additionally
                                  // unmaps the HHDM alias; everything else — RIP, stack, MMIO — is identical, so the CR3 load
                                  // keeps executing here.
    unsafe {
        activate_address_space(AddressSpace {
            primary: domain,
            secondary: 0,
        })
    };
    kprintln!("[praesidium] isolation: entered attacker domain — x86 per-domain page table (DEC-0008-6); BOTH victim aliases (HHDM + identity) unmapped here");
    let c_hhdm = contains_raw_read(hhdm);
    let c_ident = contains_raw_read(ident);
    // SAFETY: restore the kernel's real address space (its original CR3) before re-enabling preemption.
    unsafe {
        activate_address_space(AddressSpace {
            primary: original,
            secondary: 0,
        })
    };
    preempt_restore(prev);

    // SAFETY: back in the kernel domain, the HHDM alias is mapped again; confirm the sentinel survived.
    let survived = unsafe { (hhdm as *const u64).read_volatile() } == SENTINEL;
    if !survived {
        kprintln!("[praesidium] FATAL: isolation: victim data did not survive the domain crossing");
        crate::arch::halt();
    }
    c_hhdm && c_ident
}

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
