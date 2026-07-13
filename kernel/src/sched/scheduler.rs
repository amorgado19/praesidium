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

use cap_core::Budget;

use crate::arch::{self, Context};
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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum State {
    /// Eligible to run (subject to budget).
    Runnable,
    /// Body returned; never scheduled again (stack leaked until a reaper exists — P-later).
    Finished,
}

/// A stackful scheduling context (SPEC-CAP `Task`): a saved register context, its own kernel
/// stack, a CPU-time budget, and the body it runs on first schedule.
struct Task {
    context: Context,
    /// Physical base of the stack frames (`None` for the bootstrap/idle task, which runs on the
    /// kernel `BOOT_STACK`). Held for a future stack-reclaiming reaper — finished tasks leak
    /// their stack today (P-later).
    #[allow(dead_code)]
    stack_phys: Option<u64>,
    budget: Budget,
    state: State,
    /// The task body, taken and run by [`task_enter`] on first schedule. `Send` because the
    /// scheduler is a `static` shared with the timer ISR.
    body: Option<Box<dyn FnOnce() + Send>>,
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

/// Install the bootstrap ("idle") task — the currently-running boot context becomes task 0, so
/// the first [`schedule`] saves its real register state. Call once, before spawning.
pub fn bootstrap() {
    let idle = Box::new(Task {
        context: Context::EMPTY,
        stack_phys: None,
        budget: Budget::new(u32::MAX, u32::MAX), // the idle task always has budget
        state: State::Runnable,
        body: None,
    });
    let mut g = SCHED.lock();
    *g = Some(Scheduler {
        tasks: alloc::vec![idle],
        current: 0,
    });
}

/// Spawn a runnable task with `body` and CPU-time `budget`. Allocates a fresh kernel stack and
/// primes its context so the first schedule lands in the task body.
pub fn spawn(body: impl FnOnce() + Send + 'static, budget: Budget) {
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
    });
    with_sched(|s| s.tasks.push(task));
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
                Some((s.ctx_ptr(from), s.ctx_ptr(next) as *const Context))
            }
            _ => None,
        }
    });
    if let Some((from_ptr, to_ptr)) = switch {
        // SAFETY: both pointers address stable boxed `Task` contexts that outlive the switch;
        // preemption is masked (single-CPU exclusion) so no concurrent code mutates them, and
        // `to` was primed by context_init or a prior switch. Control resumes here when this task
        // is scheduled again.
        unsafe { arch::context_switch(from_ptr, to_ptr) };
    }
    arch::preempt_restore(prev);
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
// Called by the arch EL0 trap handler — live on aarch64 in P7a; wired on x86-64 once ring 3 lands.
#[allow(dead_code)]
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
                (s.ctx_ptr(from), s.ctx_ptr(next) as *const Context)
            })
        });
        match switch {
            // SAFETY: as in `schedule`; this finished task is never resumed, so the outgoing
            // context save is discarded — that is fine, its stack is abandoned.
            Some((from_ptr, to_ptr)) => unsafe { arch::context_switch(from_ptr, to_ptr) },
            None => {
                // Nothing else to run: park until the next interrupt (which may wake a task).
                arch::wait_for_interrupt();
            }
        }
    }
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
