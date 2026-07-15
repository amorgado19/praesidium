//! Userspace processes — P7a EL0/ring-3 transport + P7b multi-process. Drop to EL0/ring-3, run real
//! native code, service capability-mediated syscalls, and (P7b) run TWO real `.pex` processes at
//! once. The **mechanism** (privilege drop, trap stub, register save/restore) lives behind the arch
//! seam; this module is the **arch-generic policy**: it lays out each process's user-accessible
//! pages, runs it as a scheduler task, resolves its syscalls against ITS OWN CSpace (a per-Task
//! binding, keyed by the running task's `proc_id`), and dispatches through the P6
//! [`crate::syscall::invoke`] (resolve cptr → rights-check → act) — no ambient authority
//! (SPEC-CAP RI). A process is isolated from the kernel by page permission (its segments are
//! EL0-accessible; kernel/HHDM stay supervisor-only); process-vs-process isolation is P7b-ii.

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use abi::invoke::{op, Invocation};
use cap_core::{grant, Budget, CSpace, CapError, CapType, Cptr, GrantMode, Rights};
use mem::frame::{pfn_to_phys, phys_to_pfn};

use crate::sched::scheduler;
use crate::sync::SpinLock;
use crate::{arch, memory};

/// Max concurrent userspace processes (P7b runs 2: ping + pong).
const MAX_PROCS: usize = 4;
/// Process CSpace size — the loader's `PROCESS_SLOTS`, so loaded `.pex` and bring-up CSpaces match.
const PROC_SLOTS: usize = crate::loader::PROCESS_SLOTS;
/// The loader authority CSpace size (for [`load_process`]'s `loader` parameter).
const LOADER_SLOTS: usize = crate::loader::LOADER_SLOTS;

/// Per-process CSpaces — the EL0/ring-3 syscall handler resolves invoked cptrs against the CURRENT
/// process's CSpace, indexed by the running task's `proc_id` ([`scheduler::current_proc_id`]). This
/// per-Task binding generalises P7a's single bring-up CSpace. The lock is dropped before any
/// divergence ([`scheduler::exit_current`] never returns, so a held guard would deadlock the next).
static PROC_CSPACES: SpinLock<[Option<CSpace<PROC_SLOTS>>; MAX_PROCS]> =
    SpinLock::new([None, None, None, None]);
/// Per-process exit signal (set by a process's own syscall/fault, polled by the boot task's drive
/// loop). Atomics — NOT under `PROC_CSPACES` — so a `PROC_EXIT`/fault can signal + `exit_current`
/// without holding a lock.
static PROC_DONE: [AtomicBool; MAX_PROCS] = [
    AtomicBool::new(false),
    AtomicBool::new(false),
    AtomicBool::new(false),
    AtomicBool::new(false),
];
/// Per-process exit code (or `u64::MAX`, the killed sentinel).
static PROC_EXIT: [AtomicU64; MAX_PROCS] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

/// The CSpace slot the kernel mints a process's single-use `Reply` capability into on RECV (matches
/// `refproc::REPLY`); the process invokes it to REPLY. Free of the manifest slots (1=Sched, 2=Endpoint).
const REPLY_SLOT: Cptr = 3;

/// The cross-process IPC rendezvous over the shared Endpoint (P7b runs one Endpoint). A `caller`
/// posts a `msg` and blocks (spin-yielding to the receiver) for `reply`; the receiver takes the
/// message, mints a single-use `Reply` cap naming the caller, and (on REPLY) sets `reply`,
/// unblocking the caller — synchronous call/reply over the stackful scheduler, NO address-space
/// swap (SASOS). Reuses cap-core's single-use Reply (`mint_reply`/`consume_reply`, CAP-REPLY-1); a
/// real per-Endpoint registry keyed by objref generalises this single-Endpoint rendezvous.
struct Rdv {
    caller: Option<usize>,
    msg: u64,
    reply: Option<u64>,
}
static EP_RDV: SpinLock<Rdv> = SpinLock::new(Rdv {
    caller: None,
    msg: 0,
    reply: None,
});

/// The async-signal state for the bridge-substrate `Notification` (substrate.2). A single global
/// notification (like [`EP_RDV`], a real per-Notification registry keyed by objref generalises it):
/// a waiter `NOTIFY_WAIT`s (registers, then spin-yields until `signaled`), the kernel or a
/// `NOTIFY_SIGNAL` sets `signaled` (waking it). `signaled` is a pending-signal latch (consumed on
/// wait), so a signal that arrives before the wait is not lost — the wake IS the message (no payload).
struct Notify {
    signaled: bool,
    waiter: Option<usize>,
}
static NOTIFY: SpinLock<Notify> = SpinLock::new(Notify {
    signaled: false,
    waiter: None,
});

// ---- P7a in-kernel bring-up blob VAs (the transport + fault-kill regression) ----
const USER_CODE_VA: u64 = 0x4010_0000;
const USER_STACK_VA: u64 = 0x4011_0000;
/// A **supervisor-only** probe page (mapped via [`arch::map_page`], no user access) the fault
/// bring-up blob reads to trigger a userspace permission fault. `pub` so the arch fault blobs can
/// name it (a `const` operand); the aarch64 blob hardcodes it (`0x4012_0000`) via `movz`.
pub const FAULT_PROBE_VA: u64 = 0x4012_0000;
const PAGE: usize = 4096;

/// The CSpace slot holding the P7a bring-up blob's `Endpoint` cap. `pub` so the arch blob names it.
pub const EP_SLOT: Cptr = 1;
const EP_BADGE: u64 = 0xB1;
const AUTH_SLOTS: usize = 4;
/// Isolation domain for the P7a bring-up process: 0 (untagged / PKU key 0). P7a is single-process
/// and Fork-A-independent — it needs no process-vs-process isolation, and the P7b isolation
/// mechanism ([`arch::isolation_init`]) is armed only when [`run_processes`] runs, after P7a.
const P7A_DOMAIN: u64 = 0;

// ---- P7b reference processes (real .pex, embedded by xtask) ----
const PING_PEX: &[u8] = include_bytes!(env!("PRAESIDIUM_PING_PEX"));
const PONG_PEX: &[u8] = include_bytes!(env!("PRAESIDIUM_PONG_PEX"));
/// The HOSTILE red-team binary (P7b-ii / AC7.3): raw-reads pong's segment VA.
const EVIL_PEX: &[u8] = include_bytes!(env!("PRAESIDIUM_EVIL_PEX"));
/// Bridge substrate: the persistent echo SERVER + its client (a RECV-serve loop serving many requests).
const ECHOD_PEX: &[u8] = include_bytes!(env!("PRAESIDIUM_ECHOD_PEX"));
const ECHOCLI_PEX: &[u8] = include_bytes!(env!("PRAESIDIUM_ECHOCLI_PEX"));
/// Bridge substrate.2: a process that WAITs on a Notification (the P9 IRQ→driver-wake path).
const WAITER_PEX: &[u8] = include_bytes!(env!("PRAESIDIUM_WAITER_PEX"));
/// The CSpace slot the kernel installs the waiter's `Notification` cap at (matches `refproc::NOTIF`);
/// past the manifest (1=Sched, 2=Endpoint), Reply (3), and SharedRo (4) slots.
const NOTIF_SLOT: Cptr = 5;
/// The substrate Notification's object id (informational; one global notification for the demo).
const NOTIF_ID: u64 = 0x0090_7117;
/// User-stack VAs (one page each), distinct from the processes' segments (ping @ 0x4010_0000, pong
/// @ 0x4030_0000, evil @ 0x4050_0000) and each other — all inside the reserved window [1 GiB, 2 GiB).
const PING_STACK_VA: u64 = 0x4020_0000;
const PONG_STACK_VA: u64 = 0x4040_0000;
const EVIL_STACK_VA: u64 = 0x4060_0000;
/// Isolation domains (P7b-ii): the small non-zero value used as each process's PKU protection key
/// (x86, PTE bits [62:59]) and MTE tag (aarch64). Key 0 is the kernel/default domain, so processes
/// start at 1. A process's `PKRU` (set on switch-in) allows ONLY its own key, so a raw read of a
/// peer's page faults — the CAP-MEM-3 payoff within one address space.
const DOMAIN_PING: u64 = 1;
const DOMAIN_PONG: u64 = 2;
const DOMAIN_EVIL: u64 = 3;

// ---- v1.1: shared read-only transfer region (ADR-0004 no-swap bulk data-passing) ----
/// The VA the kernel co-maps the shared region at, in every holder's table (matches
/// `refproc::SHARED_VA`). Disjoint from the processes' segments/stacks, inside the reserved window.
const SHARED_VA: u64 = 0x4070_0000;
/// The CSpace slot the shared-region cap is installed at (matches `refproc::SHARED`); free of the
/// manifest slots (1=Sched, 2=Endpoint, 3=Reply).
const SHARED_SLOT: Cptr = 4;
/// The shared region's isolation domain: 0 (the shared key / untagged). Reachable in every process's
/// isolation context (x86 PKRU allows key 0; aarch64 leaves it untagged) — but the per-domain page
/// table maps it ONLY into a cap-holder's table, so it is the *only* co-mapped thing and nothing
/// else of the owner's is reachable. PKU/MTE are defence-in-depth; the page table + cap are the gate.
const SHARED_DOMAIN: u64 = 0;
/// Process indices — a process's scheduler `proc_id` and its registry slot.
const PING: usize = 0;
const PONG: usize = 1;
const EVIL: usize = 2;

fn fatal(msg: &str) -> ! {
    kprintln!("[praesidium] FATAL: user: {msg}");
    arch::halt();
}

fn fatal_cap(what: &str, e: CapError) -> ! {
    kprintln!("[praesidium] FATAL: user: {what}: {e:?}");
    arch::halt();
}

fn fatal_load(msg: &str, e: crate::loader::LoadError) -> ! {
    kprintln!("[praesidium] FATAL: user: {msg}: {e:?}");
    arch::halt();
}

/// Frame-zeroing hook for the bring-up CSpaces (RETYPE zeroes objects before they are observable —
/// CAP-MEM-2), through the HHDM. Receives `(objref = frame number, frames)`.
fn zero_frames(frame: u64, frames: u32) {
    for i in 0..u64::from(frames) {
        memory::zero_frame(pfn_to_phys((frame + i) as u32));
    }
}

/// Signal process `id` done with exit `code`, then retire its task (never returns). `PROC_DONE` is
/// set with `Release` so the boot task's drive loop observes the code; the task is abandoned.
fn retire(id: usize, code: u64) -> ! {
    PROC_EXIT[id].store(code, Ordering::Relaxed);
    PROC_DONE[id].store(true, Ordering::Release);
    scheduler::exit_current();
}

/// Dispatch an EL0 syscall (called by the arch trap handler with the [`Invocation`] decoded from
/// the register ABI). Resolves + rights-checks the cptr against the CURRENT process's CSpace via
/// the P6 [`crate::syscall::invoke`] — no held capability ⇒ refused, no ambient path. On success
/// the *effect* is applied here (a divergent exit cannot be expressed through the value-returning
/// dispatch): `DEBUG_EMIT` logs; `PROC_EXIT` retires the process; any other op returns its result.
// Called by the arch EL0/ring-3 syscall trap handlers (both arches).
pub fn syscall(inv: &Invocation) -> u64 {
    let id = scheduler::current_proc_id()
        .unwrap_or_else(|| fatal("EL0 syscall from a task that is not a userspace process"));
    match inv.op {
        // Cross-process IPC (AC7.2): rights-checked + block/deliver over the shared rendezvous.
        op::ENDPOINT_CALL => ipc_call(id, inv),
        op::ENDPOINT_RECV => ipc_recv(id, inv),
        op::ENDPOINT_REPLY => ipc_reply(id, inv),
        // Async signal (bridge substrate): WAIT blocks until the Notification is raised (by the
        // kernel — modelling a P9 IRQ — or a NOTIFY_SIGNAL); SIGNAL raises it. Rights-checked (RI).
        op::NOTIFY_WAIT => notify_wait(id, inv),
        op::NOTIFY_SIGNAL => notify_signal_op(id, inv),
        // DEBUG_EMIT / PROC_EXIT / the P6 ops — resolve + rights-check via the P6 dispatch, then
        // apply the effect (a divergent exit cannot be expressed through the value-returning
        // dispatch). Release the CSpace lock BEFORE any divergence (`exit_current` never returns).
        _ => {
            let result = {
                let g = PROC_CSPACES.lock();
                let cs = g[id]
                    .as_ref()
                    .unwrap_or_else(|| fatal("EL0 syscall before the process CSpace exists"));
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
                        retire(id, v);
                    }
                    _ => v,
                },
                Err(e) => {
                    kprintln!("[praesidium] user: EL0 invocation refused ({e:?}) — killing the process, kernel survives");
                    retire(id, u64::MAX);
                }
            }
        }
    }
}

/// RI check for an IPC op: resolve + rights-check the invoked cptr via the P6 dispatch. Kills the
/// process (never returns) if the cap is missing/wrong — there is no ambient IPC authority.
fn ipc_check(id: usize, inv: &Invocation) {
    let check = {
        let g = PROC_CSPACES.lock();
        let cs = g[id]
            .as_ref()
            .unwrap_or_else(|| fatal("EL0 IPC before the process CSpace exists"));
        crate::syscall::invoke(cs, inv)
    };
    if let Err(e) = check {
        kprintln!("[praesidium] user: EL0 IPC refused ({e:?}) — killing the process, kernel survives");
        retire(id, u64::MAX);
    }
}

/// Cross-process CALL (AC7.2): rights-check SEND on the Endpoint, post the message to the shared
/// rendezvous, and block (spin-yielding to the receiver) until the reply arrives; return the reply.
fn ipc_call(id: usize, inv: &Invocation) -> u64 {
    ipc_check(id, inv);
    {
        let mut r = EP_RDV.lock();
        r.caller = Some(id);
        r.msg = inv.args[0];
        r.reply = None;
    }
    loop {
        {
            let mut r = EP_RDV.lock();
            if let Some(v) = r.reply.take() {
                r.caller = None;
                return v;
            }
        }
        scheduler::yield_now();
    }
}

/// Cross-process RECV: rights-check RECV, block (spin-yielding) until a caller's message arrives,
/// mint a single-use `Reply` cap naming the caller into this process's CSpace, and return the message.
fn ipc_recv(id: usize, inv: &Invocation) -> u64 {
    ipc_check(id, inv);
    let (caller, msg) = loop {
        {
            let mut r = EP_RDV.lock();
            if let Some(c) = r.caller {
                // Consume the caller slot as the message is taken, so a receiver that RECVs again
                // before replying does not re-observe the same caller + re-mint a Reply cap (the
                // caller identity is now carried by the minted Reply cap; the caller unblocks on
                // `reply`, not `caller`). Hardens the single-slot rendezvous against a hostile server.
                r.caller = None;
                break (c, r.msg);
            }
        }
        scheduler::yield_now();
    };
    let mint = {
        let mut g = PROC_CSPACES.lock();
        g[id]
            .as_mut()
            .unwrap_or_else(|| fatal("EL0 RECV before the process CSpace exists"))
            .mint_reply(REPLY_SLOT, caller as u64, 0)
    };
    if let Err(e) = mint {
        kprintln!("[praesidium] user: RECV could not mint the Reply cap ({e:?}) — killing the process");
        retire(id, u64::MAX);
    }
    msg
}

/// REPLY: consume the single-use `Reply` cap at the invoked cptr (CAP-REPLY-1) and deliver the reply
/// word to the caller it names, unblocking the caller's CALL.
fn ipc_reply(id: usize, inv: &Invocation) -> u64 {
    let consumed = {
        let mut g = PROC_CSPACES.lock();
        g[id]
            .as_mut()
            .unwrap_or_else(|| fatal("EL0 REPLY before the process CSpace exists"))
            .consume_reply(inv.cptr as usize)
    };
    if let Err(e) = consumed {
        kprintln!("[praesidium] user: REPLY on a non-Reply/empty cptr ({e:?}) — killing the process");
        retire(id, u64::MAX);
    }
    EP_RDV.lock().reply = Some(inv.args[0]);
    0
}

/// `NOTIFY_WAIT` (bridge substrate.2): rights-check WAIT on the Notification (RI, via the P6
/// dispatch — kills the process if the cap is missing/wrong), register as the waiter, then BLOCK
/// (spin-yield) until the notification is raised (by the kernel — a P9-IRQ stand-in — or a
/// `NOTIFY_SIGNAL`). Consumes the pending signal on wake and returns 0 (no payload; the wake IS the
/// message). The lock is dropped before each yield (no lock across a context switch).
fn notify_wait(id: usize, inv: &Invocation) -> u64 {
    ipc_check(id, inv); // WAIT-right check via the P6 dispatch; kills the process if unauthorized
    NOTIFY.lock().waiter = Some(id); // register so the kernel/IRQ knows a waiter is blocked
    loop {
        {
            let mut n = NOTIFY.lock();
            if n.signaled {
                n.signaled = false; // consume the pending signal
                n.waiter = None;
                return 0; // woke
            }
        }
        scheduler::yield_now();
    }
}

/// `NOTIFY_SIGNAL` (bridge substrate.2): rights-check SIGNAL on the Notification (RI), then raise it,
/// waking a blocked waiter (or latching the signal for the next wait). Fire-and-forget.
fn notify_signal_op(id: usize, inv: &Invocation) -> u64 {
    ipc_check(id, inv); // SIGNAL-right check; kills the process if unauthorized
    NOTIFY.lock().signaled = true;
    0
}

/// Raise the substrate Notification from the KERNEL — the stand-in for the P9 in-kernel IRQ core
/// signalling a userspace driver (which will call this from the raw-IRQ handler). Wakes a blocked
/// `NOTIFY_WAIT`er; needs no capability (the kernel/IRQ source is trusted, like any hardware event).
fn notify_signal() {
    NOTIFY.lock().signaled = true;
}

/// Whether a process is currently blocked in `NOTIFY_WAIT` — the demo's boot task signals once it
/// sees the waiter has actually blocked (so the wake genuinely follows the signal).
fn notify_has_waiter() -> bool {
    NOTIFY.lock().waiter.is_some()
}

/// Handle a userspace fault (an EL0 access that trapped — a bug or a hostile raw pointer). The
/// **current process** is killed, not the kernel; the kernel keeps running. Never returns.
// Called by the arch EL0/ring-3 trap handlers (both arches) on a userspace fault.
pub fn fault(kind: &str, addr: u64) -> ! {
    let id = scheduler::current_proc_id()
        .unwrap_or_else(|| fatal("EL0 fault from a task that is not a userspace process"));
    kprintln!("[praesidium] user: EL0 fault ({kind} at {addr:#x}) — killing the process, kernel survives");
    retire(id, u64::MAX);
}

// ============================ P7a in-kernel bring-up (transport + fault-kill regression) ===========

/// Build the P7a bring-up blob's CSpace — **exactly one** capability, a badged `Endpoint` with
/// `SEND`, derived monotonically from a transient in-kernel authority (NOT the `.pex` loader
/// pipeline) — and register it as process `id`. The blob's DEBUG/EXIT syscalls are invocations on
/// this cap, so no EL0 authority is ambient (SPEC-CAP RI).
fn setup_blob_cspace(id: usize) {
    let ut_phys =
        memory::alloc_frames(0).unwrap_or_else(|| fatal("no frame for bring-up authority"));
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
    PROC_CSPACES.lock()[id] = Some(cs);
}

/// Lay out a bring-up blob (copy its native code into an owned frame, make it coherent for
/// instruction fetch, map code R-X + stack RW EL0-accessible), run it as process `id`'s scheduler
/// task that drops to EL0, and drive it from the boot task until it leaves EL0.
fn run_el0_process(blob: &[u8], id: usize, domain: u64, code_phys: u64, code_va: u64, stack_phys: u64, stack_va: u64) {
    assert!(blob.len() <= PAGE, "bring-up blob exceeds a page");
    let code_hhdm = memory::phys_to_virt(code_phys);
    // SAFETY: `code_hhdm` is an HHDM-mapped, writable frame we own; `blob` is a disjoint static
    // slice. Copy the blob (overwriting any prior blob's code — they run sequentially), then make it
    // coherent for EL0 instruction fetch.
    unsafe {
        core::ptr::copy_nonoverlapping(blob.as_ptr(), code_hhdm as *mut u8, blob.len());
    }
    arch::sync_instruction_cache(code_hhdm, blob.len());
    // Give the process its OWN page table (the hostile-isolation boundary) and map its pages into
    // it: activate it while mapping (preemption-masked so the scheduler's active-space tracking stays
    // consistent — the copy above went through the table-independent HHDM), then restore the kernel
    // space for the boot task. The scheduler swaps to `space` when this process is scheduled.
    let space = arch::new_process_space();
    let prev = arch::preempt_disable();
    // SAFETY: `space` shares the kernel mappings, so the boot task keeps running across the switch;
    // map the frames EL0-accessible R-X code / RW stack (W^X) in the process's `domain`, then restore.
    unsafe {
        arch::activate_address_space(space);
        arch::map_user_page(code_va, code_phys, arch::Prot::Rx, domain);
        arch::map_user_page(stack_va, stack_phys, arch::Prot::Rw, domain);
        arch::activate_address_space(arch::kernel_space());
    }
    arch::preempt_restore(prev);
    let stack_top = stack_va + PAGE as u64;

    PROC_DONE[id].store(false, Ordering::Release);
    PROC_EXIT[id].store(0, Ordering::Relaxed);
    scheduler::spawn_proc(
        // SAFETY: `code_va` maps EL0-executable code (entry at offset 0) and `stack_top` the top of
        // an EL0-writable stack page, both in `space`.
        move || unsafe { arch::enter_user(code_va, stack_top, domain) },
        Budget::new(u32::MAX, u32::MAX),
        id,
        domain,
        space,
    );
    while !PROC_DONE[id].load(Ordering::Acquire) {
        scheduler::yield_now();
    }
}

/// P7a bring-up: prove the EL0 transport end to end with in-kernel blobs — a capability-mediated
/// DEBUG/EXIT round-trip, then a supervisor-page raw read that KILLS the process (kernel survives,
/// a taste of AC7.3). Emits `PRAESIDIUM-P7A-OK`. (Superseded as the transport proof by the real
/// `.pex` in [`run_processes`], but kept as a regression + the fault-kill demonstration.)
pub fn run() {
    kprintln!("[praesidium] user: P7a — EL0 userspace transport (ADR-0006 / ADR-0007)");
    if !arch::el0_supported() {
        kprintln!("[praesidium] user: EL0 userspace not yet wired on x86-64 — ring 3 is a P7a follow-on");
        kprintln!("[praesidium] PRAESIDIUM-P7A-SKIP");
        return;
    }

    setup_blob_cspace(PING);
    let code_phys = memory::alloc_frames(0).unwrap_or_else(|| fatal("no frame for user code"));
    let stack_phys = memory::alloc_frames(0).unwrap_or_else(|| fatal("no frame for user stack"));

    // (1) Capability-mediated transport: DEBUG_EMIT then PROC_EXIT, each resolved + rights-checked.
    run_el0_process(arch::el0_test_blob(), PING, P7A_DOMAIN, code_phys, USER_CODE_VA, stack_phys, USER_STACK_VA);
    if PROC_EXIT[PING].load(Ordering::Relaxed) != 0 {
        fatal("bring-up process did not exit cleanly");
    }

    // (2) Isolation taste: an EL0 raw-read of a supervisor page faults; the kernel kills the process
    // and keeps running. In the fault blob's OWN page table, `FAULT_PROBE_VA` is the un-shadowed
    // supervisor identity mapping (no EL0 page there — the blob maps only its code+stack), so the
    // read is a permission fault (P7b-ii: no separate probe frame needed — the per-process table
    // already denies EL0 every VA it did not map).
    run_el0_process(arch::el0_fault_blob(), PING, P7A_DOMAIN, code_phys, USER_CODE_VA, stack_phys, USER_STACK_VA);
    if PROC_EXIT[PING].load(Ordering::Relaxed) != u64::MAX {
        fatal("fault process was not killed by the EL0 fault path");
    }
    kprintln!("[praesidium] user: EL0 fault CONTAINED — supervisor page unreadable from EL0, process killed, kernel survives");

    kprintln!("[praesidium] PRAESIDIUM-P7A-OK");
}

// ================================ P7b multi-process (real .pex) ====================================

/// Load a real `.pex` as process `id`: derive its EXACTLY-manifest caps from the shared `loader`
/// authority (no ambient authority, AC6.4; the `.pex` is HOSTILE input — the fuzzed parser + loader
/// fail closed), register its CSpace, and map a user stack. Returns `(entry, stack_top)` for the
/// scheduler task that drops to EL0. Both processes load from ONE authority, so they share the one
/// Endpoint (each a badged derivation) — the substrate for cross-process IPC (AC7.2).
fn load_process(
    loader: &mut CSpace<LOADER_SLOTS>,
    scratch: &mut Cptr,
    pex: &[u8],
    domain: u64,
    id: usize,
    stack_va: u64,
    name: &str,
) -> (u64, u64, arch::AddressSpace) {
    // Each process runs on its OWN page table (the hostile-isolation boundary): build it, then map
    // this process's segments + stack into it. Activate it while mapping (preemption-masked so the
    // scheduler's active-space tracking stays consistent), then restore the kernel space for the
    // boot task; the scheduler swaps to `space` when the process is scheduled.
    let space = arch::new_process_space();
    let mut cs = CSpace::<PROC_SLOTS>::new(crate::loader::zero_frames);
    let prev = arch::preempt_disable();
    // SAFETY: `space` shares the kernel mappings (HHDM, kernel, identity) the loader + kernel use, so
    // execution continues across the switch; the loader maps the process's segments into it.
    unsafe {
        arch::activate_address_space(space);
    }
    let loaded = crate::loader::load(pex, loader, &mut cs, domain, true, scratch)
        .unwrap_or_else(|e| fatal_load("refproc .pex failed to load", e));
    let stack_phys = memory::alloc_frames(0).unwrap_or_else(|| fatal("no frame for refproc stack"));
    // SAFETY: map an owned frame RW + EL0-accessible at the process's stack VA into `space` (in the
    // reserved window, disjoint from the segments), tagged with `domain` (defence-in-depth); then
    // restore the kernel space for the boot task.
    unsafe {
        arch::map_user_page(stack_va, stack_phys, arch::Prot::Rw, domain);
        arch::activate_address_space(arch::kernel_space());
    }
    arch::preempt_restore(prev);
    kprintln!(
        "[praesidium] user: {name}.pex loaded — entry {:#x}, budget {}, {} manifest caps (AC6.4)",
        loaded.entry,
        loaded.budget,
        crate::loader::occupied_slots(&cs)
    );
    PROC_CSPACES.lock()[id] = Some(cs);
    (loaded.entry, stack_va + PAGE as u64, space)
}

/// v1.1: map the shared region frame `phys` at [`SHARED_VA`] into `space` with `prot` (RW for the
/// owner, RO for a peer), in the shared domain. Preemption-masked activate/map/restore (like
/// [`load_process`]) so the scheduler's active-space tracking stays consistent. The co-map is done by
/// the KERNEL at share-time — userspace never edits a page table (the ruling: no EL0 map primitive).
fn map_shared_into(space: arch::AddressSpace, phys: u64, prot: arch::Prot) {
    let prev = arch::preempt_disable();
    // SAFETY: `space` shares the kernel mappings, so the boot task keeps running across the switch;
    // map the shared frame at SHARED_VA with `prot` in the shared domain, then restore kernel space.
    unsafe {
        arch::activate_address_space(space);
        arch::map_user_page(SHARED_VA, phys, prot, SHARED_DOMAIN);
        arch::activate_address_space(arch::kernel_space());
    }
    arch::preempt_restore(prev);
}

/// v1.1 red-team evidence (kernel-side): confirm the shared region is mapped READ-ONLY in a peer's
/// `space` — a hostile holder of the RO window can never write it to corrupt the owner. FATAL if it
/// is writable or unmapped (fail closed).
fn assert_shared_ro(space: arch::AddressSpace) {
    let prev = arch::preempt_disable();
    // SAFETY: activate the peer space to walk its tables for SHARED_VA's protection, then restore.
    let prot = unsafe {
        arch::activate_address_space(space);
        let p = arch::page_prot(SHARED_VA);
        arch::activate_address_space(arch::kernel_space());
        p
    };
    arch::preempt_restore(prev);
    match prot {
        Some((false, _)) => {} // read-only (not writable) — good
        other => {
            kprintln!("[praesidium] FATAL: user: shared region not READ-ONLY in the peer: {other:?}");
            arch::halt();
        }
    }
}

/// v1.1: set up the shared read-only transfer region between the OWNER (`ping`, RW) and the PEER
/// (`pong`, RO), installing the authorizing caps (RI: the kernel co-maps the region ONLY for cap
/// holders — a `Frame`(RW) for the owner, a `SharedRo` for the peer; no EL0 map op). Returns the
/// region frame number so the red-team peer (`evil`) can be co-mapped the SAME frame.
fn setup_shared_region(owner_space: arch::AddressSpace, peer_space: arch::AddressSpace) -> u64 {
    let phys = memory::alloc_frames(0).unwrap_or_else(|| fatal("no frame for the shared region"));
    memory::zero_frame(phys); // no prior contents leak into the shared window
    let pfn = u64::from(phys_to_pfn(phys));
    map_shared_into(owner_space, phys, arch::Prot::Rw); // owner: read-write
    map_shared_into(peer_space, phys, arch::Prot::Ro); // peer: read-only
    assert_shared_ro(peer_space); // red-team: the peer's window is RO — it cannot corrupt the owner
    {
        let mut g = PROC_CSPACES.lock();
        g[PING]
            .as_mut()
            .unwrap_or_else(|| fatal("shared: ping CSpace missing"))
            .install_frame(SHARED_SLOT, pfn, 1, Rights::READ | Rights::WRITE);
        g[PONG]
            .as_mut()
            .unwrap_or_else(|| fatal("shared: pong CSpace missing"))
            .install_shared_ro(SHARED_SLOT, pfn, 1, SHARED_VA);
    }
    kprintln!(
        "[praesidium] user: v1.1 shared RO transfer region co-mapped @ {SHARED_VA:#x} (owner=ping RW, peer=pong RO); RI via Frame/SharedRo caps, kernel co-map (no EL0 map op)"
    );
    pfn
}

/// P7b (i.2/i.3): load the real `refproc` ping + pong `.pex` binaries and run them CONCURRENTLY at
/// EL0/ring-3, each resolving its OWN capabilities via the per-Task CSpace binding. Both load from
/// ONE loader authority, so they share the one Endpoint (ping SEND, pong SEND+RECV): ping CALLs a
/// value, pong RECVs + REPLYs, ping gets the reply — a **cross-process capability IPC round-trip**
/// (AC7.2) with no address-space swap (SASOS). Emits `PRAESIDIUM-P7B-I3-OK`.
pub fn run_processes() {
    kprintln!("[praesidium] user: P7b — loading refproc ping + pong (real userspace binaries)");
    if !arch::el0_supported() {
        kprintln!("[praesidium] user: EL0 not wired on this arch — skipping refproc");
        return;
    }

    // Arm the process-vs-process isolation mechanism BEFORE any process page is mapped/tagged
    // (aarch64 enables MTE tag checking here; x86 armed PKU earlier at gdt_init). P7b-ii.
    arch::isolation_init();

    // Both processes load from ONE authority (so they share the one Endpoint) with a SHARED scratch
    // cursor that advances across loads — each load's retyped segment-frame cap stays in the
    // authority, so a reset base would collide (LOADER_SLOTS=32 holds both processes' scratch).
    let mut loader = crate::loader::authority();
    let mut scratch = crate::loader::L_SCRATCH;
    let (ping_entry, ping_sp, ping_space) =
        load_process(&mut loader, &mut scratch, PING_PEX, DOMAIN_PING, PING, PING_STACK_VA, "ping");
    let (pong_entry, pong_sp, pong_space) =
        load_process(&mut loader, &mut scratch, PONG_PEX, DOMAIN_PONG, PONG, PONG_STACK_VA, "pong");

    kprintln!(
        "[praesidium] user: process-vs-process isolation armed — {} (ping=domain {DOMAIN_PING}, pong=domain {DOMAIN_PONG}); the red-team proves a cross-domain raw read is contained (P7b-ii)",
        arch::isolation_mechanism()
    );

    // v1.1: co-map a shared read-only transfer region between ping (owner, RW) and pong (peer, RO)
    // BEFORE they run, so ping's round 2 can publish bulk data zero-copy and pong reads it through
    // its RO window. The SAME frame is co-mapped into evil (the red-team peer) below.
    let shared_pfn = setup_shared_region(ping_space, pong_space);

    for id in [PING, PONG] {
        PROC_DONE[id].store(false, Ordering::Release);
        PROC_EXIT[id].store(0, Ordering::Relaxed);
    }
    scheduler::spawn_proc(
        // SAFETY: `ping_entry` is loader-mapped EL0-executable code; `ping_sp` the top of ping's
        // EL0-writable stack page.
        move || unsafe { arch::enter_user(ping_entry, ping_sp, DOMAIN_PING) },
        Budget::new(u32::MAX, u32::MAX),
        PING,
        DOMAIN_PING,
        ping_space,
    );
    scheduler::spawn_proc(
        // SAFETY: as above, for pong.
        move || unsafe { arch::enter_user(pong_entry, pong_sp, DOMAIN_PONG) },
        Budget::new(u32::MAX, u32::MAX),
        PONG,
        DOMAIN_PONG,
        pong_space,
    );

    // Drive both from the boot task until each leaves EL0.
    while !(PROC_DONE[PING].load(Ordering::Acquire) && PROC_DONE[PONG].load(Ordering::Acquire)) {
        scheduler::yield_now();
    }
    if PROC_EXIT[PING].load(Ordering::Relaxed) != 0 || PROC_EXIT[PONG].load(Ordering::Relaxed) != 0 {
        fatal("a refproc process did not exit cleanly");
    }
    kprintln!("[praesidium] PRAESIDIUM-P7B-I3-OK");
    kprintln!(
        "[praesidium] user: v1.1 zero-copy transfer OK — ping published bulk to the shared region, pong read it through its RO window and echoed it (no per-message Frame map / kernel copy)"
    );

    // ---- AC7.3 red-team: a HOSTILE .pex raw-reads pong's memory. The isolation backstop must fault
    // it, the kernel kills it, and ping/pong + the kernel survive. The existential proof (CAP-MEM-3).
    run_redteam(&mut loader, &mut scratch, shared_pfn);
}

/// P7b-ii red-team (AC7.3): load the hostile `evil` `.pex` in its OWN isolation domain and run it. It
/// forms a raw pointer to `pong`'s segment VA — memory it holds NO capability for — and reads it. The
/// hardware backstop (x86 PKU / aarch64 MTE) must fault the read; the kernel kills `evil` via the EL0
/// fault path (no new wiring — a PKU `#PF` / MTE Data Abort routes there like any EL0 fault). We
/// assert `evil` was KILLED (a clean exit would mean the read SUCCEEDED — a breach — and fails
/// closed + loud) and that the kernel ran on to here. `pong`'s pages remain mapped after it exited
/// (no reaper yet), so the target is live. This is the CAP-MEM-3 payoff against a REAL hostile
/// native binary — distinct from P5b's *armed* in-kernel proof. Emits `PRAESIDIUM-P7B-II-OK`.
fn run_redteam(loader: &mut CSpace<LOADER_SLOTS>, scratch: &mut Cptr, shared_pfn: u64) {
    kprintln!(
        "[praesidium] user: P7b-ii AC7.3 red-team — loading HOSTILE evil.pex (it raw-reads ping's memory at {:#x}, in the demo-split window)",
        PING_STACK_VA - 0x10_0000 // ping's segment base 0x4010_0000 (see evil::VICTIM_VA)
    );
    let (evil_entry, evil_sp, evil_space) =
        load_process(loader, scratch, EVIL_PEX, DOMAIN_EVIL, EVIL, EVIL_STACK_VA, "evil");

    // v1.1: evil is a LEGITIMATE SharedRo holder — co-map the SAME shared region RO into evil's table
    // and install its SharedRo cap. evil may READ the region (allowed) but the per-domain table maps
    // ONLY the region into evil's table, so its attempt to use that foothold to reach ping's OTHER
    // memory (below) still faults — a shared window cannot be turned into a general read of the owner.
    map_shared_into(evil_space, pfn_to_phys(shared_pfn as u32), arch::Prot::Ro);
    PROC_CSPACES.lock()[EVIL]
        .as_mut()
        .unwrap_or_else(|| fatal("shared: evil CSpace missing"))
        .install_shared_ro(SHARED_SLOT, shared_pfn, 1, SHARED_VA);

    PROC_DONE[EVIL].store(false, Ordering::Release);
    PROC_EXIT[EVIL].store(0, Ordering::Relaxed);
    scheduler::spawn_proc(
        // SAFETY: `evil_entry` is loader-mapped EL0-executable code; `evil_sp` the top of evil's
        // EL0-writable stack page, both in evil's OWN table (which does NOT map ping/pong's pages).
        move || unsafe { arch::enter_user(evil_entry, evil_sp, DOMAIN_EVIL) },
        Budget::new(u32::MAX, u32::MAX),
        EVIL,
        DOMAIN_EVIL,
        evil_space,
    );
    while !PROC_DONE[EVIL].load(Ordering::Acquire) {
        scheduler::yield_now();
    }

    // evil MUST have been KILLED (`u64::MAX`) by the isolation fault. A clean exit means the raw
    // cross-domain read RETURNED — isolation FAILED (a breach). Fail closed + loud.
    if PROC_EXIT[EVIL].load(Ordering::Relaxed) != u64::MAX {
        fatal("AC7.3 BREACH — a hostile process read another domain's memory; ISOLATION FAILED");
    }
    kprintln!(
        "[praesidium] user: AC7.3 isolation red-team CONTAINED — hostile cross-domain raw read faulted; evil KILLED by {}; ping+pong+kernel survive (CAP-MEM-3 payoff vs a real native binary)",
        arch::isolation_mechanism()
    );
    kprintln!("[praesidium] PRAESIDIUM-P7B-II-OK");
    // v1.1: evil held a legit SharedRo window and READ the region (allowed), yet could NOT use it as
    // a foothold to reach ping's OTHER memory — the out-of-region read faulted. The shared region is
    // the ONLY co-mapped thing, read-only, and reached only through a capability (RI). Target A done.
    kprintln!(
        "[praesidium] user: v1.1 shared-region red-team CONTAINED — a hostile RO-window holder read the region but could NOT reach beyond it (per-domain tables map only the region); RO enforced"
    );
    kprintln!("[praesidium] PRAESIDIUM-V1.1-A-OK");
}

// ================================ Bridge substrate: persistent server ==============================

/// Bridge substrate.1: prove a PERSISTENT userspace SERVER — the shape the P8 FS server and P9
/// driver servers take. `echod` runs a RECV-serve LOOP (it services MANY requests from ONE
/// long-lived task, unlike v1's one-shot ping/pong); `echocli` CALLs it several times + a STOP.
/// Both load from ONE loader authority so they share the one Endpoint (echod RECV, echocli SEND).
/// The KERNEL just loads + drives — the persistence lives entirely in echod's refproc loop, served
/// by the P4 rendezvous, so no new kernel mechanism is needed here (that comes in substrate.2/.3:
/// Notification + supervisor/restart). Reuses the process slots (ping/pong/evil have exited; each
/// process still gets its OWN per-domain page table). Emits `PRAESIDIUM-SERVER-1-OK`.
pub fn run_server_demo() {
    kprintln!(
        "[praesidium] user: bridge substrate — a PERSISTENT echo SERVER (echod, a RECV-serve loop) + client (echocli)"
    );
    if !arch::el0_supported() {
        kprintln!("[praesidium] user: EL0 not wired on this arch — skipping the server demo");
        return;
    }
    let mut loader = crate::loader::authority();
    let mut scratch = crate::loader::L_SCRATCH;
    // echod = SERVER (slot PING, RECV); echocli = CLIENT (slot PONG, SEND). Distinct bases so they
    // coexist; ONE shared Endpoint (each cap derived from the one authority) carries the requests.
    let (echod_entry, echod_sp, echod_space) =
        load_process(&mut loader, &mut scratch, ECHOD_PEX, DOMAIN_PING, PING, PING_STACK_VA, "echod");
    let (cli_entry, cli_sp, cli_space) = load_process(
        &mut loader,
        &mut scratch,
        ECHOCLI_PEX,
        DOMAIN_PONG,
        PONG,
        PONG_STACK_VA,
        "echocli",
    );

    for id in [PING, PONG] {
        PROC_DONE[id].store(false, Ordering::Release);
        PROC_EXIT[id].store(0, Ordering::Relaxed);
    }
    scheduler::spawn_proc(
        // SAFETY: `echod_entry` is loader-mapped EL0-executable code; `echod_sp` the top of its stack.
        move || unsafe { arch::enter_user(echod_entry, echod_sp, DOMAIN_PING) },
        Budget::new(u32::MAX, u32::MAX),
        PING,
        DOMAIN_PING,
        echod_space,
    );
    scheduler::spawn_proc(
        // SAFETY: as above, for the client.
        move || unsafe { arch::enter_user(cli_entry, cli_sp, DOMAIN_PONG) },
        Budget::new(u32::MAX, u32::MAX),
        PONG,
        DOMAIN_PONG,
        cli_space,
    );

    // Drive until both leave EL0: the client after its requests + STOP, the server after the STOP.
    while !(PROC_DONE[PING].load(Ordering::Acquire) && PROC_DONE[PONG].load(Ordering::Acquire)) {
        scheduler::yield_now();
    }
    if PROC_EXIT[PING].load(Ordering::Relaxed) != 0 || PROC_EXIT[PONG].load(Ordering::Relaxed) != 0 {
        fatal("the persistent server or its client did not exit cleanly");
    }
    kprintln!(
        "[praesidium] user: persistent server served 4 requests + STOP from ONE long-lived RECV-loop task (the P8/P9 server shape) and shut down gracefully"
    );
    kprintln!("[praesidium] PRAESIDIUM-SERVER-1-OK");
}

/// Bridge substrate.2: prove the `Notification` async-signal runtime (SIGNAL/WAIT) — the P9
/// IRQ→driver wake path. A `waiter` process holds a WAIT-only `Notification` cap (kernel-installed)
/// and blocks in `NOTIFY_WAIT`; once it has genuinely blocked, the KERNEL raises the notification
/// ([`notify_signal`], the stand-in for a P9 in-kernel IRQ handler) and the waiter wakes. Proves a
/// userspace principal sleeps until a kernel/hardware event, capability-gated (no ambient wakeups).
/// Emits `PRAESIDIUM-NOTIFY-OK`.
pub fn run_notify_demo() {
    kprintln!(
        "[praesidium] user: bridge substrate — Notification WAIT/SIGNAL (a userspace waiter woken by a kernel signal, the P9 IRQ->driver path)"
    );
    if !arch::el0_supported() {
        kprintln!("[praesidium] user: EL0 not wired on this arch — skipping the notification demo");
        return;
    }
    let mut loader = crate::loader::authority();
    let mut scratch = crate::loader::L_SCRATCH;
    let (entry, sp, space) =
        load_process(&mut loader, &mut scratch, WAITER_PEX, DOMAIN_PING, PING, PING_STACK_VA, "waiter");
    // Install the waiter's Notification cap (WAIT only — the kernel holds the signal side). The wake
    // is thus capability-gated: a process with no Notification cap cannot be signalled (RI).
    PROC_CSPACES.lock()[PING]
        .as_mut()
        .unwrap_or_else(|| fatal("notify: waiter CSpace missing"))
        .install_notification(NOTIF_SLOT, NOTIF_ID, Rights::WAIT);
    {
        let mut n = NOTIFY.lock();
        n.signaled = false;
        n.waiter = None;
    }

    PROC_DONE[PING].store(false, Ordering::Release);
    PROC_EXIT[PING].store(0, Ordering::Relaxed);
    scheduler::spawn_proc(
        // SAFETY: `entry` is loader-mapped EL0-executable code; `sp` the top of the waiter's stack.
        move || unsafe { arch::enter_user(entry, sp, DOMAIN_PING) },
        Budget::new(u32::MAX, u32::MAX),
        PING,
        DOMAIN_PING,
        space,
    );

    // Drive until the waiter leaves EL0. Once it has genuinely BLOCKED in NOTIFY_WAIT (registered as
    // the waiter), the kernel raises the notification exactly once — so the wake provably follows.
    let mut signalled = false;
    while !PROC_DONE[PING].load(Ordering::Acquire) {
        if !signalled && notify_has_waiter() {
            kprintln!("[praesidium] user: waiter BLOCKED in NOTIFY_WAIT — kernel raises the Notification (stand-in for a P9 IRQ)");
            notify_signal();
            signalled = true;
        }
        scheduler::yield_now();
    }
    if !signalled {
        fatal("the waiter exited without ever blocking in NOTIFY_WAIT");
    }
    if PROC_EXIT[PING].load(Ordering::Relaxed) != 0 {
        fatal("the waiter did not exit cleanly after waking");
    }
    kprintln!(
        "[praesidium] user: Notification WAIT/SIGNAL OK — the waiter slept until the kernel signalled, then woke (the P9 IRQ->driver wake path, capability-gated)"
    );
    kprintln!("[praesidium] PRAESIDIUM-NOTIFY-OK");
}
