//! Userspace bring-up (P7a): drop to EL0/ring-3, run real native code, and service its syscall
//! trap — the transport half of the v1 thesis (ADR-0001 / ADR-0006). The **mechanism** (the
//! privilege drop, the trap vector/stub, the register save/restore) lives behind the arch seam;
//! this module holds the **arch-generic policy**: it lays out a process's user-accessible pages,
//! runs it as a scheduler task, and dispatches its syscalls. A process is isolated from the kernel
//! by page permission alone — its segments are mapped EL0-accessible, everything else (the HHDM,
//! the kernel image) stays supervisor-only, so an EL0 raw pointer into kernel memory faults.
//!
//! **No ambient authority (SPEC-CAP RI).** Every EL0 syscall is a capability invocation: the trap
//! decodes an [`abi::invoke::Invocation`] and dispatches it through the P6 [`crate::syscall::invoke`]
//! (resolve cptr → rights-check → act) — the trap the P6 phase *shaped*, now *fired*. The bring-up
//! process is granted, **in-kernel**, exactly the one capability it needs — a badged `Endpoint`
//! with `SEND` — so even DEBUG/EXIT are capability-gated (they require that cap), not ambient.
//!
//! P7a proves this with two in-kernel bring-up blobs: one makes capability-mediated syscalls and
//! exits cleanly; one raw-reads a supervisor page and is KILLED (the kernel survives — a taste of
//! the AC7.3 isolation payoff). The real `refproc` `.pex` processes + cross-process IPC + the full
//! isolation red-team are P7b.

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use abi::invoke::{op, Invocation};
use cap_core::{grant, CSpace, CapError, CapType, Cptr, GrantMode, Rights};
use mem::frame::{pfn_to_phys, phys_to_pfn};

use crate::sched::scheduler;
use crate::sync::SpinLock;
use crate::{arch, memory};

/// Set by [`syscall`]/[`fault`] when the running bring-up process leaves EL0, so [`run_el0_process`]
/// knows to stop. `EXIT_CODE` carries the process's exit code (or `u64::MAX`, the killed sentinel).
static USER_DONE: AtomicBool = AtomicBool::new(false);
static EXIT_CODE: AtomicU64 = AtomicU64::new(0);

/// The bring-up process's CSpace: it holds **exactly one** capability (a badged `Endpoint` with
/// `SEND`, granted in-kernel by [`setup_process_cspace`]). The EL0 syscall dispatch resolves the
/// invoked cptr against this — the "current process" a real per-Task CSpace binding generalises in
/// P7b. Read-only after setup; the lock is dropped before any divergence ([`scheduler::exit_current`]
/// never returns, so a held guard would leak / deadlock).
static PROC_CS: SpinLock<Option<CSpace<PROC_SLOTS>>> = SpinLock::new(None);

/// Virtual addresses for the bring-up process, inside the reserved process window `[1 GiB, 2 GiB)`
/// (`loader::PROC_VA_BASE`). Code and stack are one page each; reused across the two bring-up
/// processes (they run one at a time).
const USER_CODE_VA: u64 = 0x4010_0000;
const USER_STACK_VA: u64 = 0x4011_0000;
/// A **supervisor-only** probe page (mapped via [`arch::map_page`], no user access) the fault
/// bring-up process reads to trigger a userspace permission fault. `pub` so the arch fault blobs
/// can name it (a `const` operand); the aarch64 blob hardcodes it (`0x4012_0000`) via `movz` — keep
/// them in sync. x86-64 references this const directly.
pub const FAULT_PROBE_VA: u64 = 0x4012_0000;
const PAGE: usize = 4096;

/// The CSpace slot holding the process's one capability (its bring-up-service `Endpoint`). `pub`
/// so the arch EL0 blob can name it as the invoked cptr (a `const` operand in its `global_asm!`).
pub const EP_SLOT: Cptr = 1;
/// The badge the kernel stamps on the bring-up Endpoint (identifies the sender — DEC-0006-3).
const EP_BADGE: u64 = 0xB1;
/// Slots in the transient in-kernel authority CSpace (an Untyped + the Endpoint it retypes).
const AUTH_SLOTS: usize = 4;
/// Slots in the bring-up process's CSpace (it holds exactly one capability).
const PROC_SLOTS: usize = 4;

fn fatal(msg: &str) -> ! {
    kprintln!("[praesidium] FATAL: user: {msg}");
    arch::halt();
}

fn fatal_cap(what: &str, e: CapError) -> ! {
    kprintln!("[praesidium] FATAL: user: {what}: {e:?}");
    arch::halt();
}

/// Frame-zeroing hook for the bring-up CSpaces (RETYPE zeroes objects before they are observable —
/// CAP-MEM-2), through the HHDM. Receives `(objref = frame number, frames)`.
fn zero_frames(frame: u64, frames: u32) {
    for i in 0..u64::from(frames) {
        memory::zero_frame(pfn_to_phys((frame + i) as u32));
    }
}

/// Build the bring-up process's minimal CSpace: **exactly one** capability — a badged `Endpoint`
/// with `SEND` — derived monotonically from a transient in-kernel authority (an `Untyped` we
/// carve an `Endpoint` from, then MINT-grant into the process). This is deliberately NOT the
/// `.pex` manifest→loader pipeline (that packages real userspace programs — P7b); P7a grants the
/// one cap the transport proof needs directly. The process's DEBUG/EXIT syscalls are invocations
/// on this cap, so no EL0 authority is ambient (SPEC-CAP RI).
fn setup_process_cspace() {
    let ut_phys = memory::alloc_frames(0).unwrap_or_else(|| fatal("no frame for bring-up authority"));
    let mut authority = CSpace::<AUTH_SLOTS>::new(zero_frames);
    authority.set_root_untyped(u64::from(phys_to_pfn(ut_phys)), 1);
    authority
        .retype(0, CapType::Endpoint, 1, 1, 1)
        .unwrap_or_else(|e| fatal_cap("retype bring-up Endpoint", e));
    let mut cs = CSpace::<PROC_SLOTS>::new(zero_frames);
    grant(
        &mut authority,
        1,
        &mut cs,
        EP_SLOT,
        Rights::SEND,
        EP_BADGE,
        GrantMode::Mint,
    )
    .unwrap_or_else(|e| fatal_cap("grant bring-up Endpoint", e));
    *PROC_CS.lock() = Some(cs);
}

/// Dispatch an EL0 syscall (called by the arch trap handler with the [`Invocation`] decoded from
/// the register ABI). Resolves + rights-checks the cptr through the P6 [`crate::syscall::invoke`]
/// — no held capability ⇒ refused, there is no ambient path. On success the *effect* is applied
/// here (a divergent exit cannot be expressed through the value-returning dispatch): `DEBUG_EMIT`
/// logs the value; `PROC_EXIT` retires the process; any other op returns its result word to EL0.
// Called by the arch EL0/ring-3 syscall trap handlers (both arches).
pub fn syscall(inv: &Invocation) -> u64 {
    // Release the CSpace lock BEFORE any divergence: `exit_current` never returns, so a held guard
    // would never drop (deadlocking the next process's access).
    let result = {
        let g = PROC_CS.lock();
        let cs = g
            .as_ref()
            .unwrap_or_else(|| fatal("EL0 syscall before the bring-up CSpace exists"));
        crate::syscall::invoke(cs, inv)
    };
    match result {
        Ok(v) => match inv.op {
            op::DEBUG_EMIT => {
                kprintln!("[praesidium] user: EL0 syscall DEBUG value={v:#x}");
                0
            }
            op::PROC_EXIT => {
                kprintln!("[praesidium] user: process exited (code {v})");
                EXIT_CODE.store(v, Ordering::Relaxed);
                USER_DONE.store(true, Ordering::Release);
                scheduler::exit_current(); // -> ! : retire this process's task, schedule away
            }
            // A real capability op (CAP_IDENTIFY / FRAME_PROBE / ENDPOINT_SEND): return its result.
            _ => v,
        },
        Err(e) => {
            // The trusted bring-up process always holds its cap; a refusal is a bug. Fail closed —
            // kill the process (the kernel survives), exactly as an unexpected fault would.
            kprintln!("[praesidium] user: EL0 invocation refused ({e:?}) — killing the process, kernel survives");
            EXIT_CODE.store(u64::MAX, Ordering::Relaxed);
            USER_DONE.store(true, Ordering::Release);
            scheduler::exit_current();
        }
    }
}

/// Handle a userspace fault (an EL0 access that trapped — a bug or a hostile raw pointer). The
/// **process** is killed, not the kernel; the kernel keeps running. Never returns.
// Called by the arch EL0/ring-3 trap handlers (both arches) on a userspace fault.
pub fn fault(kind: &str, addr: u64) -> ! {
    kprintln!("[praesidium] user: EL0 fault ({kind} at {addr:#x}) — killing the process, kernel survives");
    EXIT_CODE.store(u64::MAX, Ordering::Relaxed);
    USER_DONE.store(true, Ordering::Release);
    scheduler::exit_current();
}

/// Lay out a bring-up process (copy its native code into an owned frame, make it coherent for
/// instruction fetch, map code R-X + stack RW EL0-accessible at the process VAs), run it as a
/// scheduler task that drops to EL0, and drive it cooperatively from the boot task until it leaves
/// EL0 (a syscall exit or a fault, each of which retires the task). Resets the outcome signals
/// before spawning; the caller inspects `EXIT_CODE` afterwards.
fn run_el0_process(blob: &[u8], code_phys: u64, code_va: u64, stack_phys: u64, stack_va: u64) {
    assert!(blob.len() <= PAGE, "bring-up blob exceeds a page");
    let code_hhdm = memory::phys_to_virt(code_phys);
    // SAFETY: `code_hhdm` is an HHDM-mapped, writable frame we own; `blob` is a disjoint static
    // slice. We copy the blob (overwriting any prior process's code — the two run sequentially),
    // then make the written code coherent for EL0 instruction fetch.
    unsafe {
        core::ptr::copy_nonoverlapping(blob.as_ptr(), code_hhdm as *mut u8, blob.len());
    }
    arch::sync_instruction_cache(code_hhdm, blob.len());
    // SAFETY: map owned frames EL0-accessible at the process VAs — R-X code, RW stack (W^X).
    unsafe {
        arch::map_user_page(code_va, code_phys, arch::Prot::Rx);
        arch::map_user_page(stack_va, stack_phys, arch::Prot::Rw);
    }
    let stack_top = stack_va + PAGE as u64;

    USER_DONE.store(false, Ordering::Release);
    EXIT_CODE.store(0, Ordering::Relaxed);
    scheduler::spawn(
        // SAFETY: `code_va` maps EL0-executable code (entry at offset 0) and `stack_top` the
        // exclusive top of an EL0-writable stack page.
        move || unsafe { arch::enter_user(code_va, stack_top) },
        cap_core::Budget::new(u32::MAX, u32::MAX),
    );
    while !USER_DONE.load(Ordering::Acquire) {
        scheduler::yield_now();
    }
}

/// P7a bring-up: prove the EL0 transport end to end.
///  1. Grant the process its one capability in-kernel (an `Endpoint` with `SEND`).
///  2. Run a process that makes **capability-mediated** syscalls (`DEBUG_EMIT` then `PROC_EXIT`,
///     each an invocation resolved + rights-checked by the P6 dispatch) and exits cleanly.
///  3. Run a process that raw-reads a **supervisor-only** page — it is KILLED, the kernel survives
///     (a taste of AC7.3: EL0 is isolated from the kernel by page permission alone).
///
/// Emits `PRAESIDIUM-P7A-OK` on success.
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

    // (1) Grant the trusted bring-up process its one capability, so its EL0 syscalls are capability
    // invocations, never ambient (SPEC-CAP RI).
    setup_process_cspace();

    // Allocate the code + stack frames ONCE and reuse them for both bring-up processes (they run
    // strictly one at a time and reuse the same VAs). These two frames — plus the fault probe and
    // the authority Endpoint frame below, and each task's kernel stack (scheduler, P-later reaper)
    // — are a small, bounded, one-time bring-up leak: P7a has no process/stack reaper, so real
    // reclamation lands with process teardown in P7b.
    let code_phys = memory::alloc_frames(0).unwrap_or_else(|| fatal("no frame for user code"));
    let stack_phys = memory::alloc_frames(0).unwrap_or_else(|| fatal("no frame for user stack"));

    // (2) Capability-mediated transport: real EL0 code makes DEBUG_EMIT then PROC_EXIT syscalls,
    // each resolved + rights-checked through the P6 invoke dispatch. Clean exit (code 0).
    run_el0_process(arch::el0_test_blob(), code_phys, USER_CODE_VA, stack_phys, USER_STACK_VA);
    if EXIT_CODE.load(Ordering::Relaxed) != 0 {
        fatal("bring-up process did not exit cleanly");
    }

    // (3) Isolation taste: map a supervisor-only page, then run a process that raw-reads it. The
    // EL0 access faults; the kernel KILLS the process and keeps running.
    let probe_phys = memory::alloc_frames(0).unwrap_or_else(|| fatal("no frame for the fault probe"));
    // SAFETY: map a supervisor-only (no EL0 access) read-only page at `FAULT_PROBE_VA` in the
    // process window; an EL0 read of it is a permission fault, exercising the kill-process path.
    unsafe {
        arch::map_page(FAULT_PROBE_VA, probe_phys, arch::Prot::Ro);
    }
    run_el0_process(arch::el0_fault_blob(), code_phys, USER_CODE_VA, stack_phys, USER_STACK_VA);
    if EXIT_CODE.load(Ordering::Relaxed) != u64::MAX {
        fatal("fault process was not killed by the EL0 fault path");
    }
    kprintln!("[praesidium] user: EL0 fault CONTAINED — supervisor page unreadable from EL0, process killed, kernel survives");

    kprintln!("[praesidium] PRAESIDIUM-P7A-OK");
}
