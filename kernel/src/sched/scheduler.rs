//! The stackful task scheduler (ADR-0003 P3b — Tier-2 preemptive fallback).
//!
//! Each [`Task`] is a green thread: its own kernel stack + a saved [`Context`] + a bound
//! [`Budget`]. The scheduler round-robins **runnable** tasks (a task with a depleted `Sched` is
//! skipped — CAP-SCHED-1) and switches between them with [`arch::context_switch`]. The *same*
//! switch serves both tiers: a cooperative [`yield_now`] is a plain call into [`schedule`], and
//! the timer IRQ ([`timer_tick`], P3b Phase 2) calls [`schedule`] too — so a task that refuses to
//! yield is preempted at the tick exactly as if it had yielded.
//!
//! Single-CPU: mutual exclusion between mainline code and the timer IRQ is by **masking
//! preemption** (disabling interrupts) around every scheduler critical section, not a spinlock
//! held across the switch. Task control blocks are boxed so their `Context` has a stable address
//! across the `tasks` vector reallocating.

use alloc::boxed::Box;
use alloc::vec::Vec;

use core::sync::atomic::{AtomicU64, Ordering};

use cap_core::Budget;

use crate::arch::{self, AddressSpace, Context};
use crate::memory;
use crate::sync::SpinLock;

/// Order of the per-task kernel stack: 2^2 frames = 16 KiB usable.
const STACK_ORDER: u8 = 2;
const STACK_BYTES: usize = (1 << STACK_ORDER) * 4096;
const PAGE: usize = 4096;
/// Order of the whole stack *block*: the usable stack (4 frames) plus a guard frame
/// immediately below it, rounded up to the next power of two → 2^3 = 8 frames (32 KiB).
/// The usable stack sits at the top of the block; the guard is the frame directly beneath
/// it; the remaining frames below the guard are owned-but-unused slack (isolation Layer 3,
/// ADR-0008 — closing the P3b/P4 "guard pages below task stacks" deferral).
const GUARDED_STACK_ORDER: u8 = 3;
const BLOCK_BYTES: usize = (1 << GUARDED_STACK_ORDER) * PAGE;
/// Frames in a task's kernel-stack block — exactly what [`reap_finished`] returns to the buddy for a
/// reaped task. Exported so the bridge-substrate conservation proof can account the kernel stack as a
/// KNOWN constant rather than a spawn-time free-count delta: `spawn_inner` also does an
/// [`arch::install_guard_page`] that may split an HHDM huge page (a one-off *permanent kernel* table
/// frame the reaper does NOT reclaim), so folding the kernel stack in by measurement would let that
/// split perturb the footprint. Counting it as this constant excludes the guard split structurally.
pub const KERNEL_STACK_FRAMES: u64 = 1 << GUARDED_STACK_ORDER;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum State {
    /// Eligible to run (subject to budget).
    Runnable,
    /// Body returned or the process was killed; never scheduled again. Its resources are still
    /// held until [`reap_finished`] reclaims them (its kernel stack + per-domain page table).
    Finished,
    /// Reaped: its kernel stack + page-table frames have been freed and it holds nothing more. An
    /// inert tombstone (kept so `current` indices stay stable), never scheduled — [`pick_next`] and
    /// [`runnable_besides`] skip it like any non-`Runnable` state.
    Reaped,
}

/// A stackful scheduling context (SPEC-CAP `Task`): a saved register context, its own kernel
/// stack, a CPU-time budget, and the body it runs on first schedule.
struct Task {
    context: Context,
    /// Physical base of the guarded stack *block* (`None` for the bootstrap/idle task, which runs
    /// on the kernel `BOOT_STACK`, and for an already-reaped task). [`reap_finished`] restores the
    /// guard page's HHDM alias and returns this block to the buddy when the task is reaped.
    stack_phys: Option<u64>,
    budget: Budget,
    state: State,
    /// The task body, taken and run by [`task_enter`] on first schedule. `Send` because the
    /// scheduler is a `static` shared with the timer ISR.
    body: Option<Box<dyn FnOnce() + Send>>,
    /// Which userspace process this task runs, if any (P7b): the index into `crate::user`'s process
    /// registry. The EL0/ring-3 syscall handler reads [`current_proc_id`] to resolve invoked cptrs
    /// against the CURRENT process's CSpace — the per-Task binding that generalises the single
    /// bring-up CSpace. `None` for kernel tasks (the idle/boot task, the hog, the executor).
    proc_id: Option<usize>,
    /// The task's isolation domain (P7b-ii): the process's PKU protection key (x86) / MTE tag
    /// (aarch64) — the cooperative-compartment / defence-in-depth layer. The scheduler programs it
    /// via [`arch::set_domain`] on switch-in. `None` for kernel tasks.
    domain: Option<u64>,
    /// The task's address space (P7b-ii hostile-isolation boundary): a userspace process runs on its
    /// OWN page table, which maps only ITS pages + the shared kernel — so it cannot even name another
    /// process's memory. The scheduler swaps to it on switch-in ([`arch::activate_address_space`]).
    /// Kernel tasks use the shared [`arch::kernel_space`]. This swap is the real cross-process
    /// boundary (PKU/MTE are userspace-defeatable; a page table is not).
    space: AddressSpace,
}

struct Scheduler {
    // Box is REQUIRED, not redundant: `schedule` derives raw `*mut Context` pointers into these
    // task control blocks and uses them across the context switch (after dropping the lock). A
    // bare `Vec<Task>` would move the Tasks on reallocation and dangle those pointers; boxing
    // pins each `Context` at a stable heap address.
    #[allow(clippy::vec_box)]
    tasks: Vec<Box<Task>>,
    current: usize,
}

impl Scheduler {
    /// Raw pointer to task `i`'s context. The `Box<Task>` keeps this address stable across
    /// `tasks` reallocation, so it is valid to use after the scheduler lock is dropped.
    fn ctx_ptr(&mut self, i: usize) -> *mut Context {
        core::ptr::addr_of_mut!(self.tasks[i].context)
    }

    /// Next runnable task after `current`, round-robin, or `None` if the current task is the
    /// only runnable one. A `Finished` task or one whose budget is depleted is skipped.
    fn pick_next(&self) -> Option<usize> {
        let n = self.tasks.len();
        for step in 1..=n {
            let i = (self.current + step) % n;
            let t = &self.tasks[i];
            if t.state == State::Runnable && !t.budget.is_depleted() {
                return Some(i);
            }
        }
        None
    }

    fn runnable_besides(&self, who: usize) -> usize {
        self.tasks
            .iter()
            .enumerate()
            .filter(|(i, t)| *i != who && t.state == State::Runnable && !t.budget.is_depleted())
            .count()
    }
}

/// The scheduler, `None` until [`bootstrap`]. Every access masks preemption first.
static SCHED: SpinLock<Option<Scheduler>> = SpinLock::new(None);

/// The address-space roots currently loaded (to skip a redundant page-table swap + TLB flush when
/// switching between tasks that share a space — every kernel task, the SASOS cooperative fast path).
/// Initialised lazily on the first switch. Only touched under preemption-masking (single CPU).
static ACTIVE_PRIMARY: AtomicU64 = AtomicU64::new(0);
static ACTIVE_SECONDARY: AtomicU64 = AtomicU64::new(0);

/// Swap to `space`'s page table iff it differs from the currently-loaded one. Called on switch-in
/// (preemption-masked). Kernel↔kernel switches share [`arch::kernel_space`], so they skip the swap
/// (zero-swap for cooperative/kernel tasks); a switch to/from an isolated userspace process swaps
/// (hostile isolation costs a page-table reload + TLB flush, like any OS).
fn activate_space(space: AddressSpace) {
    if space.primary != ACTIVE_PRIMARY.load(Ordering::Relaxed)
        || space.secondary != ACTIVE_SECONDARY.load(Ordering::Relaxed)
    {
        // SAFETY: every process/kernel space shares the kernel high half + HHDM + identity, so the
        // current PC, stack, and kernel data stay mapped across the switch; called preemption-masked.
        unsafe { arch::activate_address_space(space) };
        ACTIVE_PRIMARY.store(space.primary, Ordering::Relaxed);
        ACTIVE_SECONDARY.store(space.secondary, Ordering::Relaxed);
    }
}

/// Install the bootstrap ("idle") task — the currently-running boot context becomes task 0, so
/// the first [`schedule`] saves its real register state. Call once, before spawning.
pub fn bootstrap() {
    let idle = Box::new(Task {
        context: Context::EMPTY,
        stack_phys: None,
        budget: Budget::new(u32::MAX, u32::MAX), // the idle task always has budget
        state: State::Runnable,
        body: None,
        proc_id: None,
        domain: None,
        space: arch::kernel_space(),
    });
    let mut g = SCHED.lock();
    *g = Some(Scheduler {
        tasks: alloc::vec![idle],
        current: 0,
    });
}

/// Spawn a runnable kernel task with `body` and CPU-time `budget` (no bound userspace process; runs
/// on the shared kernel address space).
pub fn spawn(body: impl FnOnce() + Send + 'static, budget: Budget) {
    spawn_inner(body, budget, None, None, arch::kernel_space());
}

/// Spawn a runnable task bound to userspace process `proc_id` in isolation `domain`, running on its
/// OWN address `space` (P7b): the EL0/ring-3 syscall handler resolves this task's invoked cptrs
/// against that process's CSpace (see [`current_proc_id`]); the scheduler swaps to `space` (the
/// hostile-isolation boundary — the process's table maps only its own pages) and programs `domain`
/// (PKU key / MTE tag, defence-in-depth) on switch-in.
pub fn spawn_proc(
    body: impl FnOnce() + Send + 'static,
    budget: Budget,
    proc_id: usize,
    domain: u64,
    space: AddressSpace,
) {
    spawn_inner(body, budget, Some(proc_id), Some(domain), space);
}

/// Allocate a guarded kernel stack, prime the initial context, and enqueue a new runnable task.
fn spawn_inner(
    body: impl FnOnce() + Send + 'static,
    budget: Budget,
    proc_id: Option<usize>,
    domain: Option<u64>,
    space: AddressSpace,
) {
    // Allocate the whole guarded block; the usable stack occupies its top `STACK_BYTES`, with a
    // guard frame directly below the stack's lowest byte.
    let block = memory::alloc_frames(GUARDED_STACK_ORDER)
        .unwrap_or_else(|| fatal("no frames for a task stack"));
    let stack_top = (memory::phys_to_virt(block) as usize + BLOCK_BYTES) as *mut u8;
    let stack_base_phys = block + (BLOCK_BYTES - STACK_BYTES) as u64;
    let guard_phys = stack_base_phys - PAGE as u64;
    // SAFETY: `guard_phys` is the frame immediately below the usable stack, inside our own freshly
    // allocated block — never handed to anything else — so unmapping its HHDM alias affects only
    // this allocation. A downward stack overflow past `stack_base_phys` now faults on the guard
    // instead of corrupting a neighbour. The stack is never freed (see `task_exit`), so the
    // guard's unmapped alias is never returned to the buddy; a future stack-reclaim path MUST
    // restore this mapping before freeing the block.
    unsafe { arch::install_guard_page(memory::phys_to_virt(guard_phys)) };
    // SAFETY: `stack_top` is the exclusive top of `STACK_BYTES` of freshly-allocated, HHDM-mapped,
    // writable stack we own; context_init writes only the small initial frame just below it.
    let context = unsafe { arch::context_init(stack_top) };
    let task = Box::new(Task {
        context,
        stack_phys: Some(block),
        budget,
        state: State::Runnable,
        body: Some(Box::new(body)),
        proc_id,
        domain,
        space,
    });
    with_sched(|s| s.tasks.push(task));
}

/// The userspace process bound to the currently-running task, or `None` for a kernel task. Read by
/// the EL0/ring-3 syscall handler to resolve invoked cptrs against the right process's CSpace (P7b).
#[must_use]
pub fn current_proc_id() -> Option<usize> {
    with_sched(|s| s.tasks[s.current].proc_id)
}

/// Cooperatively yield the CPU: pick the next runnable task and switch to it. Returns when this
/// task is next scheduled. A no-op if nothing else is runnable.
pub fn yield_now() {
    schedule();
}

/// The core switch: hand the CPU to the next runnable task. Shared by cooperative yields and the
/// preemptive timer tick. Runs with preemption masked while touching scheduler state; the
/// context switch itself happens after the lock is dropped (the target re-enters the scheduler
/// on its own terms).
fn schedule() {
    let prev = arch::preempt_disable();
    let switch = with_sched(|s| {
        let from = s.current;
        match s.pick_next() {
            Some(next) if next != from => {
                s.current = next;
                let ksp = kernel_stack_top(s, next);
                let domain = s.tasks[next].domain;
                let space = s.tasks[next].space;
                Some((s.ctx_ptr(from), s.ctx_ptr(next) as *const Context, ksp, domain, space))
            }
            _ => None,
        }
    });
    if let Some((from_ptr, to_ptr, ksp, domain, space)) = switch {
        // Swap to the incoming task's page table (the hostile-isolation boundary) BEFORE it runs;
        // shared for kernel↔kernel switches, so those skip the swap (SASOS cooperative fast path).
        activate_space(space);
        // x86 `syscall` shares one kernel-RSP global, so point it (+ TSS.RSP0) at the incoming
        // task's kernel stack BEFORE it runs — no-op on aarch64 (per-task-banked SP_EL1).
        if let Some(top) = ksp {
            arch::set_kernel_stack(top);
        }
        // Program the incoming task's isolation domain (PKU key / MTE tag — defence-in-depth /
        // cooperative-compartment layer); a kernel task (`None`) gets all keys (P7b-ii).
        arch::set_domain(domain);
        // SAFETY: both pointers address stable boxed `Task` contexts that outlive the switch;
        // preemption is masked (single-CPU exclusion) so no concurrent code mutates them, and
        // `to` was primed by context_init or a prior switch. Control resumes here when this task
        // is scheduled again.
        unsafe { arch::context_switch(from_ptr, to_ptr) };
    }
    arch::preempt_restore(prev);
}

/// The (exclusive) top of task `i`'s kernel stack, or `None` for the bootstrap/idle task (which
/// runs on the kernel `BOOT_STACK` and never syscalls from ring 3). Used to point the x86 syscall
/// kernel stack at the incoming task on each switch.
fn kernel_stack_top(s: &Scheduler, i: usize) -> Option<u64> {
    s.tasks[i]
        .stack_phys
        .map(|block| memory::phys_to_virt(block) + BLOCK_BYTES as u64)
}

/// The generic launcher a new task's stack `ret`s into (via the arch trampoline) the first time
/// it is scheduled. Runs the task body with preemption enabled, then retires the task.
pub extern "C" fn task_enter() -> ! {
    // We arrived here from `schedule`, which masked preemption around the switch; enable it so
    // this task is preemptible while it runs.
    arch::preempt_enable();
    let body = with_sched(|s| s.tasks[s.current].body.take());
    if let Some(body) = body {
        body();
    }
    task_exit();
}

/// Retire the CURRENT task from outside the normal body-return path — e.g. a userspace process
/// exiting via a syscall or being killed on a fault, called from the EL0 trap handler running on
/// that task's kernel stack. Marks it finished and schedules away for good (its kernel stack, and
/// the abandoned handler frame on it, are discarded — the task never resumes).
// Called by the arch EL0/ring-3 trap handlers (both arches) when a process exits or is killed.
pub fn exit_current() -> ! {
    task_exit();
}

/// Retire the current task: mark it finished and hand the CPU away for good. Never returns.
fn task_exit() -> ! {
    let _ = arch::preempt_disable();
    with_sched(|s| s.tasks[s.current].state = State::Finished);
    loop {
        // Switch to any other runnable task. Once none remain, fall through to a parked idle.
        let switch = with_sched(|s| {
            s.pick_next().map(|next| {
                let from = s.current;
                s.current = next;
                let ksp = kernel_stack_top(s, next);
                let domain = s.tasks[next].domain;
                let space = s.tasks[next].space;
                (s.ctx_ptr(from), s.ctx_ptr(next) as *const Context, ksp, domain, space)
            })
        });
        match switch {
            // SAFETY: as in `schedule`; this finished task is never resumed, so the outgoing
            // context save is discarded — that is fine, its stack is abandoned.
            Some((from_ptr, to_ptr, ksp, domain, space)) => {
                activate_space(space);
                if let Some(top) = ksp {
                    arch::set_kernel_stack(top);
                }
                arch::set_domain(domain);
                unsafe { arch::context_switch(from_ptr, to_ptr) }
            }
            None => {
                // Nothing else to run: park until the next interrupt (which may wake a task).
                arch::wait_for_interrupt();
            }
        }
    }
}

/// Reap the MOST-RECENTLY-spawned `Finished` task bound to userspace process `proc_id` (bridge
/// substrate.3 — closes the v1 "finished tasks leak their stack" deferral): return its kernel stack
/// **block** to the buddy (restoring the guard page's HHDM alias first — [`spawn_inner`] unmapped
/// it), mark the task `Reaped` so it is never scheduled or double-reaped, and hand its address
/// `space` back to the caller so the arch layer can reclaim the per-domain page-table frames +
/// user-leaf frames. Returns `None` if no such finished task exists.
///
/// **Last-match, not first:** a process *slot* (`proc_id`) is reused across demos, so several older
/// finished generations of the same slot may still be unreaped — the supervisor wants the one it just
/// saw die (the current occupant it is about to restart), which is the most recently spawned, hence
/// [`Iterator::rposition`].
///
/// Safe to call from the boot/supervisor task while the dead task's `space` is **not** active: a
/// `Finished` task is never switched to again, so its saved context + kernel stack are dead, and the
/// supervisor task always runs on [`arch::kernel_space`] (so the freed root is never the live one).
pub fn reap_finished(proc_id: usize) -> Option<AddressSpace> {
    let (block, space) = with_sched(|s| {
        let i = s
            .tasks
            .iter()
            .rposition(|t| t.proc_id == Some(proc_id) && t.state == State::Finished)?;
        let block = s.tasks[i].stack_phys.take();
        let space = s.tasks[i].space;
        s.tasks[i].state = State::Reaped;
        Some((block, space))
    })?;
    // Reclaim the kernel stack outside the scheduler lock (frame ops take the allocator lock). The
    // usable stack sits at the top of the block; the guard is the frame directly below it (see
    // `spawn_inner`).
    if let Some(block) = block {
        let stack_base_phys = block + (BLOCK_BYTES - STACK_BYTES) as u64;
        let guard_phys = stack_base_phys - PAGE as u64;
        // SAFETY: `spawn_inner` unmapped this guard frame's HHDM alias to catch stack overflow;
        // restore it RW before the block returns to the buddy, else a later `alloc_zeroed_frame`
        // would fault zeroing that frame through the (still non-present) HHDM alias. The guard frame
        // is inside our own block, so remapping its alias affects only this allocation.
        unsafe { arch::map_page(memory::phys_to_virt(guard_phys), guard_phys, arch::Prot::Rw) };
        memory::free_frames(block);
    }
    Some(space)
}

/// Charge one tick of CPU time to the running task and return whether it should be preempted
/// now (the P3b Phase-2 timer calls this from the IRQ, then, if `true`, calls [`preempt`]).
pub fn on_tick() -> bool {
    with_sched(|s| {
        let cur = s.current;
        s.tasks[cur].budget.charge(1);
        // Preempt if another runnable task exists — round-robin fairness / bound the hog.
        s.runnable_besides(cur) > 0
    })
}

/// Force a reschedule from interrupt context (the timer IRQ). Preemption is already masked in
/// the IRQ; [`schedule`]'s own mask/restore nests harmlessly.
pub fn preempt() {
    schedule();
}

/// Run `f` with the initialized scheduler, **preemption masked** across the whole critical
/// section. Masking (not just the spinlock) is what prevents an ISR-vs-mainline deadlock: the
/// timer IRQ also enters the scheduler, so if mainline held the lock unmasked and the timer
/// fired, the ISR would spin on a lock only mainline can release. On a single CPU, masking gives
/// the mutual exclusion; the lock adds the type-safety (and SMP-readiness).
fn with_sched<R>(f: impl FnOnce(&mut Scheduler) -> R) -> R {
    let prev = arch::preempt_disable();
    let mut g = SCHED.lock();
    let r = f(g
        .as_mut()
        .unwrap_or_else(|| fatal("scheduler used before bootstrap")));
    drop(g);
    arch::preempt_restore(prev);
    r
}

fn fatal(msg: &str) -> ! {
    kprintln!("[praesidium] FATAL: sched: {msg}");
    crate::arch::halt();
}
