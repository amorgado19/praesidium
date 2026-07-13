//! aarch64 EL0 (userspace) entry + syscall/fault trap (P7a) — the arch mechanism behind
//! [`crate::user`]. Dropping to EL0 is an `eret` with `SPSR_EL1.M = EL0t`; the return trap lands
//! in the "Lower EL, aarch64" vector group (wired in `interrupts.rs`), whose sync stub saves the
//! user integer context and calls [`el0_sync_handler`] here.

use core::arch::{asm, global_asm};

use abi::invoke::MSG_REGS;

/// Drop to EL0 at `entry` with user stack `user_sp`. Never returns: the process runs until it
/// exits (a syscall) or faults, each of which retires its scheduler task and schedules away.
///
/// # Safety
/// `entry` must map EL0-executable code and `user_sp` an EL0-writable, 16-aligned stack. The
/// caller must be a scheduler task whose kernel stack (SP_EL1) can host the trap frames.
pub unsafe fn enter_user(entry: u64, user_sp: u64) -> ! {
    // SPSR for EL0t (`M[3:0]=0`) with D/A/I/F masked: the P7a bring-up process yields control only
    // via a syscall. Preemptible userspace (unmask IRQ + wire the EL0-IRQ vector) is a follow-on.
    // SAFETY: set the banked user SP, the EL0 return address + PSTATE, then `eret` to EL0.
    unsafe {
        asm!(
            "msr sp_el0, {sp}",
            "msr elr_el1, {entry}",
            "msr spsr_el1, {spsr}",
            "eret",
            sp = in(reg) user_sp,
            entry = in(reg) entry,
            spsr = in(reg) 0x3c0u64, // D,A,I,F = 1 ; M = EL0t
            options(noreturn),
        );
    }
}

/// Whether this backend can run EL0 userspace yet.
#[must_use]
pub fn el0_supported() -> bool {
    true
}

extern "C" {
    static praesidium_el0_blob: u8;
    static praesidium_el0_blob_end: u8;
}

/// The P7a bring-up user program (native aarch64): `DEBUG(0xBEEF)` then `EXIT(0)` — a real EL0
/// binary exercising the privilege drop + syscall trap. Replaced by the `refproc` `.pex` in P7b.
#[must_use]
pub fn el0_test_blob() -> &'static [u8] {
    let start = core::ptr::addr_of!(praesidium_el0_blob);
    let end = core::ptr::addr_of!(praesidium_el0_blob_end);
    // SAFETY: both symbols bound the same contiguous, immutable `.rodata` blob emitted below; the
    // length is their non-negative byte difference.
    unsafe { core::slice::from_raw_parts(start, end as usize - start as usize) }
}

global_asm!(
    r#"
.section .rodata
.balign 4
.global praesidium_el0_blob
.global praesidium_el0_blob_end
praesidium_el0_blob:
    mov x8, #1          // sys::DEBUG
    movz x0, #0xBEEF    // value to log
    svc #0
    mov x8, #2          // sys::EXIT
    mov x0, #0          // exit code 0
    svc #0
    brk #0              // unreachable (EXIT does not return to EL0)
praesidium_el0_blob_end:
"#
);

/// The EL0 synchronous-exception handler. Called by `el0_sync_stub` with the saved integer frame
/// (slot `N` = `xN`). Decodes `ESR_EL1.EC`: `SVC` (0x15) is a syscall — dispatch via the generic
/// policy and place the result in the process's `x0`; a data/instruction abort (0x24/0x20) kills
/// the process; anything else fails closed.
pub extern "C" fn el0_sync_handler(frame: *mut u64) {
    let (esr, far): (u64, u64);
    // SAFETY: reading ESR/FAR_EL1 is side-effect-free; they describe the trap.
    unsafe {
        asm!(
            "mrs {e}, esr_el1", "mrs {f}, far_el1",
            e = out(reg) esr, f = out(reg) far,
            options(nomem, nostack, preserves_flags),
        );
    }
    match esr >> 26 {
        0x15 => {
            // SVC from aarch64: x8 = selector, x0.. = argument message registers.
            // SAFETY: `frame` is the 34-slot saved frame; slot i holds xi for i < 31.
            let num = unsafe { frame.add(8).read() };
            let mut args = [0u64; MSG_REGS];
            for (i, a) in args.iter_mut().enumerate() {
                // SAFETY: i < MSG_REGS <= 8, in range of the saved GPR frame.
                *a = unsafe { frame.add(i).read() };
            }
            let result = crate::user::syscall(num, &args);
            // SAFETY: write the result into the saved x0 slot; the epilogue returns it to EL0.
            unsafe { frame.add(0).write(result) };
        }
        0x24 => crate::user::fault("data abort", far), // -> ! (kills the process)
        0x20 => crate::user::fault("instruction abort", far),
        ec => {
            kprintln!("[praesidium] FATAL: unexpected EL0 exception ec={ec:#x} esr={esr:#x} far={far:#x}");
            crate::arch::halt();
        }
    }
}
