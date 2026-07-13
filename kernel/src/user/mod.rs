//! Userspace bring-up (P7a): drop to EL0/ring-3, run real native code, and service its syscall
//! trap — the transport half of the v1 thesis (ADR-0001 / ADR-0006). The **mechanism** (the
//! privilege drop, the trap vector/stub, the register save/restore) lives behind the arch seam;
//! this module holds the **arch-generic policy**: it lays out a process's user-accessible pages,
//! runs it as a scheduler task, and dispatches its syscalls. A process is isolated from the kernel
//! by page permission alone — its segments are mapped EL0-accessible, everything else (the HHDM,
//! the kernel image) stays supervisor-only, so an EL0 raw pointer into kernel memory faults.
//!
//! P7a proves this with one in-kernel bring-up blob (an arch-native `SVC`/`syscall` + exit). The
//! real `refproc` `.pex` processes + cross-process IPC + the isolation red-team (AC7.3) are P7b.
//! The EL0 trap the arch backend now fires wraps [`crate::syscall::invoke`] (P6 shaped it for
//! exactly this) once `INVOKE` is wired.

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use abi::invoke::sys;

use crate::sched::scheduler;
use crate::{arch, memory};

/// Set by [`syscall`]/[`fault`] when the running bring-up process leaves EL0, so [`run`]'s drive
/// loop knows to stop. `EXIT_CODE` carries the process's exit code (or a fault sentinel).
static USER_DONE: AtomicBool = AtomicBool::new(false);
static EXIT_CODE: AtomicU64 = AtomicU64::new(0);

/// Virtual addresses for the bring-up process, inside the reserved process window `[1 GiB, 2 GiB)`
/// (`loader::PROC_VA_BASE`). Code and stack are one page each.
const USER_CODE_VA: u64 = 0x4010_0000;
const USER_STACK_VA: u64 = 0x4011_0000;
const PAGE: usize = 4096;

fn fatal(msg: &str) -> ! {
    kprintln!("[praesidium] FATAL: user: {msg}");
    arch::halt();
}

/// Dispatch a userspace syscall (called by the arch EL0 trap handler with the decoded
/// syscall-number register and the argument message registers). Returns the result word to place
/// in the process's first return register; `EXIT` diverges (schedules the process away for good).
// Called by the arch EL0 trap handler — live on aarch64 in P7a; wired on x86-64 once ring 3 lands.
#[allow(dead_code)]
pub fn syscall(num: u64, args: &[u64; abi::invoke::MSG_REGS]) -> u64 {
    match num {
        sys::DEBUG => {
            kprintln!("[praesidium] user: EL0 syscall DEBUG value={:#x}", args[0]);
            0
        }
        sys::EXIT => {
            kprintln!("[praesidium] user: process exited (code {})", args[0]);
            EXIT_CODE.store(args[0], Ordering::Relaxed);
            USER_DONE.store(true, Ordering::Release);
            scheduler::exit_current(); // -> ! : abandon this process's task, schedule away
        }
        // INVOKE (the capability-invocation path wrapping syscall::invoke) is wired in P7 once a
        // process carries its CSpace; an unknown selector fails closed with an error sentinel.
        _ => {
            kprintln!("[praesidium] user: unknown syscall {num}");
            u64::MAX
        }
    }
}

/// Handle a userspace fault (an EL0 access that trapped — a bug or a hostile raw pointer). The
/// **process** is killed, not the kernel; the kernel keeps running. Never returns.
// Called by the arch EL0 trap handler — live on aarch64 in P7a; wired on x86-64 once ring 3 lands.
#[allow(dead_code)]
pub fn fault(kind: &str, addr: u64) -> ! {
    kprintln!("[praesidium] user: EL0 fault ({kind} at {addr:#x}) — killing the process, kernel survives");
    EXIT_CODE.store(u64::MAX, Ordering::Relaxed);
    USER_DONE.store(true, Ordering::Release);
    scheduler::exit_current();
}

/// P7a bring-up: run one process at EL0 (a small arch-native blob that does a `DEBUG` syscall then
/// `EXIT`), proving the privilege drop + syscall trap + user-mapped isolation round-trip. Emits
/// `PRAESIDIUM-P7A-OK` on success.
pub fn run() {
    kprintln!("[praesidium] user: P7a — EL0 userspace transport (ADR-0006 / ADR-0007)");
    if !arch::el0_supported() {
        // x86-64 ring 3 is a P7a follow-on (the aarch64 EL0 path is validated first). Report the
        // skip with a distinct headline so the `user` smoke scenario passes cleanly on x86 instead
        // of failing by construction; this is NOT an EL0 run (see xtask `USER_REQUIRED_X86`).
        kprintln!("[praesidium] user: EL0 userspace not yet wired on x86-64 — ring 3 is a P7a follow-on");
        kprintln!("[praesidium] PRAESIDIUM-P7A-SKIP");
        return;
    }

    // Lay out the process: copy its code into an owned frame, make it coherent for instruction
    // fetch, and map code EL0-executable (R-X) + a stack EL0-writable (RW-NX) at the process VAs.
    let blob = arch::el0_test_blob();
    let code_phys = memory::alloc_frames(0).unwrap_or_else(|| fatal("no frame for user code"));
    let stack_phys = memory::alloc_frames(0).unwrap_or_else(|| fatal("no frame for user stack"));
    assert!(blob.len() <= PAGE, "bring-up blob exceeds a page");
    let code_hhdm = memory::phys_to_virt(code_phys);
    // SAFETY: `code_hhdm` is a freshly-allocated, HHDM-mapped, writable frame we own; `blob` is a
    // disjoint static slice. We then make the written code coherent for EL0 instruction fetch.
    unsafe {
        core::ptr::copy_nonoverlapping(blob.as_ptr(), code_hhdm as *mut u8, blob.len());
    }
    arch::sync_instruction_cache(code_hhdm, blob.len());
    // SAFETY: map owned frames EL0-accessible at the process VAs — R-X code, RW stack (W^X).
    unsafe {
        arch::map_user_page(USER_CODE_VA, code_phys, arch::Prot::Rx);
        arch::map_user_page(USER_STACK_VA, stack_phys, arch::Prot::Rw);
    }
    let stack_top = USER_STACK_VA + PAGE as u64;

    // Run the process as a scheduler task whose body drops to EL0; it returns control only by a
    // syscall (EXIT) or a fault, each of which retires the task. Drive it cooperatively from the
    // boot task until it leaves EL0.
    USER_DONE.store(false, Ordering::Release);
    scheduler::spawn(
        // SAFETY: `USER_CODE_VA` maps EL0-executable code (the copied blob, entry at offset 0) and
        // `stack_top` the exclusive top of an EL0-writable stack page.
        move || unsafe { arch::enter_user(USER_CODE_VA, stack_top) },
        cap_core::Budget::new(u32::MAX, u32::MAX),
    );
    while !USER_DONE.load(Ordering::Acquire) {
        scheduler::yield_now();
    }

    if EXIT_CODE.load(Ordering::Relaxed) != 0 {
        fatal("bring-up process did not exit cleanly");
    }
    kprintln!("[praesidium] PRAESIDIUM-P7A-OK");
}
