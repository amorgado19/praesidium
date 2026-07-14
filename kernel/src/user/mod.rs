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

    // ---- AC7.3 red-team: a HOSTILE .pex raw-reads pong's memory. The isolation backstop must fault
    // it, the kernel kills it, and ping/pong + the kernel survive. The existential proof (CAP-MEM-3).
    run_redteam(&mut loader, &mut scratch);
}

/// P7b-ii red-team (AC7.3): load the hostile `evil` `.pex` in its OWN isolation domain and run it. It
/// forms a raw pointer to `pong`'s segment VA — memory it holds NO capability for — and reads it. The
/// hardware backstop (x86 PKU / aarch64 MTE) must fault the read; the kernel kills `evil` via the EL0
/// fault path (no new wiring — a PKU `#PF` / MTE Data Abort routes there like any EL0 fault). We
/// assert `evil` was KILLED (a clean exit would mean the read SUCCEEDED — a breach — and fails
/// closed + loud) and that the kernel ran on to here. `pong`'s pages remain mapped after it exited
/// (no reaper yet), so the target is live. This is the CAP-MEM-3 payoff against a REAL hostile
/// native binary — distinct from P5b's *armed* in-kernel proof. Emits `PRAESIDIUM-P7B-II-OK`.
fn run_redteam(loader: &mut CSpace<LOADER_SLOTS>, scratch: &mut Cptr) {
    kprintln!(
        "[praesidium] user: P7b-ii AC7.3 red-team — loading HOSTILE evil.pex (it raw-reads ping's memory at {:#x}, in the demo-split window)",
        PING_STACK_VA - 0x10_0000 // ping's segment base 0x4010_0000 (see evil::VICTIM_VA)
    );
    let (evil_entry, evil_sp, evil_space) =
        load_process(loader, scratch, EVIL_PEX, DOMAIN_EVIL, EVIL, EVIL_STACK_VA, "evil");

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
}
