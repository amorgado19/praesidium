//! aarch64 EL0 (userspace) entry + syscall/fault trap (P7a) — the arch mechanism behind
//! [`crate::user`]. Dropping to EL0 is an `eret` with `SPSR_EL1.M = EL0t`; the return trap lands
//! in the "Lower EL, aarch64" vector group (wired in `interrupts.rs`), whose sync stub saves the
//! user integer context and calls [`el0_sync_handler`] here. Every EL0 syscall is a capability
//! invocation (there is no ambient selector): `x8 = sys::INVOKE`, `x0 = cptr`, `x1 = op`,
//! `x2..x5 = args`, dispatched through the P6 [`crate::syscall::invoke`].

use core::arch::{asm, global_asm};

use abi::invoke::{sys, Invocation, MSG_REGS};

/// Drop to EL0 at `entry` with user stack `user_sp`, in isolation `domain` (P7b-ii). Never returns:
/// the process runs until it exits (a syscall) or faults, each of which retires its scheduler task
/// and schedules away.
///
/// The `domain`'s 4-bit tag is placed in the top byte of the banked user SP (`SP_EL0`), so the
/// process's stack accesses (SP-relative) carry the domain's MTE tag and match its Normal-Tagged,
/// domain-tagged stack granules (TBI0 makes the tag byte ignored for translation). A peer forming a
/// tag-0 raw pointer to this stack therefore tag-faults. Domain 0 leaves SP untagged (the P7a
/// single-process bring-up, which needs no process-vs-process isolation and maps untagged pages).
///
/// # Safety
/// `entry` must map EL0-executable code and `user_sp` an EL0-writable, 16-aligned stack. The
/// caller must be a scheduler task whose kernel stack (SP_EL1) can host the trap frames.
pub unsafe fn enter_user(entry: u64, user_sp: u64, domain: u64) -> ! {
    // Tag the banked user SP with the process's domain (top byte; TBI0 ignores it for translation).
    let sp = user_sp | ((domain & 0xf) << 56);
    // Initial-register ABI: a process starts at EL0 with EVERY general-purpose register zero, so no
    // live kernel value (a pointer, a stack address) leaks across the privilege drop. The process
    // sets up its own registers from zero. SPSR for EL0t (`M[3:0]=0`) with D/A/I/F masked: the P7a
    // bring-up process yields control only via a syscall; preemptible userspace (unmask IRQ + wire
    // the EL0-IRQ vector) is a P7b follow-on.
    // SAFETY: set the banked user SP + EL0 return address + PSTATE (consuming the reg inputs), then
    // zero x0..x30 and `eret` to EL0. `options(noreturn)` — no registers need preserving.
    unsafe {
        asm!(
            "msr sp_el0, {sp}",
            "msr elr_el1, {entry}",
            "msr spsr_el1, {spsr}",
            "mov x0, xzr",  "mov x1, xzr",  "mov x2, xzr",  "mov x3, xzr",
            "mov x4, xzr",  "mov x5, xzr",  "mov x6, xzr",  "mov x7, xzr",
            "mov x8, xzr",  "mov x9, xzr",  "mov x10, xzr", "mov x11, xzr",
            "mov x12, xzr", "mov x13, xzr", "mov x14, xzr", "mov x15, xzr",
            "mov x16, xzr", "mov x17, xzr", "mov x18, xzr", "mov x19, xzr",
            "mov x20, xzr", "mov x21, xzr", "mov x22, xzr", "mov x23, xzr",
            "mov x24, xzr", "mov x25, xzr", "mov x26, xzr", "mov x27, xzr",
            "mov x28, xzr", "mov x29, xzr", "mov x30, xzr",
            "eret",
            sp = in(reg) sp,
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

/// Set the current task's kernel stack for the syscall/fault path — a **no-op on aarch64**: the
/// trap uses the per-task-banked `SP_EL1` (the kernel SP that `context_switch` swaps), so it is
/// already per-task without a global. The x86 seam needs the real thing (its `syscall` shares one
/// kernel-RSP global). The scheduler calls this on every switch-in.
pub fn set_kernel_stack(_top: u64) {}

extern "C" {
    static praesidium_el0_blob: u8;
    static praesidium_el0_blob_end: u8;
    static praesidium_el0_fault_blob: u8;
    static praesidium_el0_fault_blob_end: u8;
}

/// The P7a bring-up user program (native aarch64): a **capability-mediated** `DEBUG_EMIT(0xBEEF)`
/// then `PROC_EXIT(0)` — each an `SVC` carrying an [`Invocation`] on the one Endpoint capability
/// the process holds (`x0 = cptr`, `x1 = op`, `x2 = arg`). Replaced by the `refproc` `.pex` in P7b.
#[must_use]
pub fn el0_test_blob() -> &'static [u8] {
    let start = core::ptr::addr_of!(praesidium_el0_blob);
    let end = core::ptr::addr_of!(praesidium_el0_blob_end);
    // SAFETY: both symbols bound the same contiguous, immutable `.rodata` blob emitted below; the
    // length is their non-negative byte difference.
    unsafe { core::slice::from_raw_parts(start, end as usize - start as usize) }
}

/// The P7a fault bring-up program (native aarch64): a raw read of a **supervisor-only** page
/// (`FAULT_PROBE_VA`) — an EL0 permission fault. Proves the kernel kills the faulting process (not
/// itself): EL0 is isolated from the kernel by page permission alone (a taste of AC7.3).
#[must_use]
pub fn el0_fault_blob() -> &'static [u8] {
    let start = core::ptr::addr_of!(praesidium_el0_fault_blob);
    let end = core::ptr::addr_of!(praesidium_el0_fault_blob_end);
    // SAFETY: both symbols bound the same contiguous, immutable `.rodata` blob emitted below.
    unsafe { core::slice::from_raw_parts(start, end as usize - start as usize) }
}

global_asm!(
    r#"
.section .rodata
.balign 4
.global praesidium_el0_blob
.global praesidium_el0_blob_end
praesidium_el0_blob:
    mov  x8, #{invoke}      // sys::INVOKE (the only syscall selector)
    mov  x0, #{ep}          // cptr = the one Endpoint capability the process holds
    mov  x1, #{debug}       // op = DEBUG_EMIT
    movz x2, #0xBEEF        // args[0] = the value to emit
    svc  #0
    mov  x8, #{invoke}      // sys::INVOKE
    mov  x0, #{ep}          // cptr = same Endpoint
    mov  x1, #{exit}        // op = PROC_EXIT
    mov  x2, #0             // args[0] = exit code 0
    svc  #0
    brk  #0                 // unreachable (PROC_EXIT does not return to EL0)
praesidium_el0_blob_end:
"#,
    invoke = const abi::invoke::sys::INVOKE,
    ep = const crate::user::EP_SLOT as u64,
    debug = const abi::invoke::op::DEBUG_EMIT as u64,
    exit = const abi::invoke::op::PROC_EXIT as u64,
);

global_asm!(
    r#"
.section .rodata
.balign 4
.global praesidium_el0_fault_blob
.global praesidium_el0_fault_blob_end
praesidium_el0_fault_blob:
    movz x0, #0x4012, lsl #16   // x0 = 0x4012_0000 (MUST equal crate::user::FAULT_PROBE_VA)
    ldr  x1, [x0]               // EL0 read of a supervisor-only page -> data abort (permission)
    brk  #0                     // unreachable: the load traps; the kernel kills the process
praesidium_el0_fault_blob_end:
"#
);

/// The EL0 synchronous-exception handler. Called by `el0_sync_stub` with the saved integer frame
/// (slot `N` = `xN` for `N < 31`). Decodes `ESR_EL1.EC`: `SVC` (0x15) is a capability-invocation
/// syscall — decode the [`Invocation`] from the register ABI and dispatch via the generic policy,
/// placing the result in the process's `x0`; a data/instruction abort (0x24/0x20) kills the
/// process (never returning); anything else fails closed.
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
            // SVC from aarch64. Register ABI (behind the ADR-0007 seam): x8 = syscall selector; for
            // the only selector (sys::INVOKE) x0 = cptr, x1 = op, x2..x5 = args. Everything is a
            // capability invocation (DEC-0006-3) — there is no ambient selector.
            // SAFETY: `frame` is the saved 34-slot GPR frame; slot i holds xi for i < 31, and
            // 8, 0, 1, 2..2+MSG_REGS are all in range.
            let sel = unsafe { frame.add(8).read() };
            if sel != sys::INVOKE {
                crate::user::fault("bad syscall selector", sel); // -> ! (kills the process)
            }
            let inv = unsafe {
                let mut args = [0u64; MSG_REGS];
                let mut i = 0;
                while i < MSG_REGS {
                    args[i] = frame.add(2 + i).read();
                    i += 1;
                }
                Invocation {
                    cptr: frame.add(0).read() as u32,
                    op: frame.add(1).read() as u16,
                    args,
                }
            };
            let result = crate::user::syscall(&inv);
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
