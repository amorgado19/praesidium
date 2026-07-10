//! Executor + capability-scheduling integration (P3).
//!
//! Wires the pure `sched` crate (the cooperative async executor) and `cap-core`'s `Sched`
//! budget model to the running kernel, and runs the P3 boot demo asserting the gates:
//!  - **AC3.4** — `Sched` SPLIT/DELEGATE are monotonic (CPU time conserved), and a `Sched`
//!    subtree revokes cleanly (destroying a `Sched` never touches frames).
//!  - **AC3.1** — the executor advances multiple `Future`s cooperatively; their `.await`
//!    yields interleave them rather than running one to completion first.
//!  - **AC3.2** — a task's `Sched` budget gates runnability (CAP-SCHED-1): a depleted task is
//!    parked, not polled, until replenishment.
//!  - **AC3.3** (P3b) — a non-yielding task is **preempted** by the hardware timer so another
//!    task progresses; the [`scheduler`] provides the stackful context switch this needs.
//!
//! P3a (AC3.1/3.2/3.4) runs the async executor cooperatively with a logical replenishment tick.
//! P3b then stands up interrupts + the timer + the stackful [`scheduler`] and runs the executor
//! **as a task** alongside a non-yielding hog — proving the two ADR-0003 tiers (cooperative
//! primary + preemptive fallback) compose. No capability is fabricated here — every
//! `Cap`/`Budget` comes from `cap-core` (CAP-RUST-1).

use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use cap_core::{Budget, CSpace, CapError};
use sched::{yield_now, Executor, Task};

use crate::arch;

pub(crate) mod scheduler;
pub use scheduler::task_enter;

/// Slots for the `Sched` demo CSpace (single root CNode).
const SLOTS: usize = 16;

/// Zeroing hook for the `Sched` demo CSpace. A `Sched` is pure CPU-time accounting with no
/// backing frames, so `cap-core` never invokes this for a `Sched` (see `destroy_slot`); it
/// exists only to satisfy the `CSpace` constructor and must never actually be called.
fn never_zero(_frame: u64, _frames: u32) {
    fatal("Sched demo zeroed a frame (a Sched has no backing memory)");
}

fn fatal(msg: &str) -> ! {
    kprintln!("[praesidium] FATAL: sched: {msg}");
    crate::arch::halt();
}

fn fatal_err(op: &str, e: CapError) -> ! {
    kprintln!("[praesidium] FATAL: sched: {op} failed: {e:?}");
    crate::arch::halt();
}

/// Run P3: the `Sched` capability model + cooperative executor (P3a), then the preemptive
/// stackful scheduler (P3b). Prints `PRAESIDIUM-P3A-OK` after the cooperative half and
/// `PRAESIDIUM-P3-OK` after preemption; any violated invariant fails the boot closed.
pub fn run() {
    demo_sched_caps();
    demo_executor();
    kprintln!("[praesidium] PRAESIDIUM-P3A-OK");
    demo_preemption();
    kprintln!("[praesidium] PRAESIDIUM-P3-OK");
}

/// The preemption timer frequency. aarch64 derives a precise rate from `CNTFRQ`; x86 uses a
/// nominal (uncalibrated) LAPIC count — either way fast enough to preempt a spinning task.
const PREEMPT_HZ: u32 = 100;

/// Ticked forever by the non-yielding hog task; the worker reads it to prove the hog ran (and
/// was therefore preempted, since it never yields).
static HOG_COUNTER: AtomicU64 = AtomicU64::new(0);
/// Set by the worker task when it has demonstrated preemption; drives the boot task's exit.
static WORKER_DONE: AtomicBool = AtomicBool::new(false);

/// AC3.3 + integration — real hardware preemption of a non-yielding task, with the cooperative
/// async executor running as a task under the same scheduler.
fn demo_preemption() {
    arch::interrupts_init();
    scheduler::bootstrap();
    arch::timer_init(PREEMPT_HZ);
    kprintln!("[praesidium] sched: interrupts + {PREEMPT_HZ}Hz timer up; stackful scheduler live");

    // A non-yielding CPU hog + the cooperative executor as a task. The hog never yields, so the
    // worker can only get the CPU if the timer preempts the hog.
    scheduler::spawn(hog_task, Budget::new(u32::MAX, u32::MAX));
    scheduler::spawn(worker_task, Budget::new(u32::MAX, u32::MAX));

    // Drive from the boot/idle task until the worker signals success.
    while !WORKER_DONE.load(Ordering::Acquire) {
        scheduler::yield_now();
    }
    // Mask preemption before returning so the still-spinning hog can't be scheduled again between
    // here and the kernel's final halt — a clean shutdown, not a live-locked CPU.
    let _ = arch::preempt_disable();
    kprintln!("[praesidium] sched: preemption demo complete; scheduler parked");
}

/// The adversarial non-cooperator: spins forever, never yielding or awaiting. Only preemption
/// can take the CPU from it.
fn hog_task() {
    loop {
        HOG_COUNTER.fetch_add(1, Ordering::Relaxed);
        core::hint::spin_loop();
    }
}

/// The cooperative worker, run as a stackful task: it runs the P3a async executor (Tier-1
/// cooperative Futures) to completion — which can only happen because the timer preempted the
/// never-yielding hog to schedule this task — then confirms ongoing preemptive time-sharing.
fn worker_task() {
    // Tier 1 under Tier 2: the async executor advances cooperative Futures while this whole task
    // is itself subject to preemption.
    let log: Rc<RefCell<Vec<char>>> = Rc::new(RefCell::new(Vec::new()));
    let mut ex = Executor::new();
    ex.spawn(Task::new(
        worker(log.clone(), 'X', 4),
        Budget::new(1000, 1000),
    ));
    ex.spawn(Task::new(
        worker(log.clone(), 'Y', 4),
        Budget::new(1000, 1000),
    ));
    ex.run_until_idle();
    {
        let l = log.borrow();
        if l.len() < 4 || !(l[0] == 'X' && l[1] == 'Y' && l[2] == 'X' && l[3] == 'Y') {
            fatal("cooperative executor did not interleave while under preemption");
        }
    }
    // We are running, yet the hog never yields — so the timer must have preempted it. Confirm the
    // hog actually got the CPU first (spun ≥ once) before being preempted to schedule us.
    if HOG_COUNTER.load(Ordering::Relaxed) == 0 {
        fatal("hog never ran — preemption evidence inconclusive");
    }
    kprintln!("[praesidium] sched: cooperative Futures interleaved while preemptible (Tier-1 under Tier-2)");

    // Ongoing time-sharing: yield the CPU a few times and confirm the hog advances each time we
    // are descheduled, then preemption returns the CPU to us.
    for _ in 0..3 {
        let before = HOG_COUNTER.load(Ordering::Relaxed);
        scheduler::yield_now();
        if HOG_COUNTER.load(Ordering::Relaxed) <= before {
            fatal("hog made no progress across a yield — scheduler not round-robining");
        }
    }
    kprintln!(
        "[praesidium] sched: non-yielding hog preempted (spun {} ticks); worker made progress (AC3.3)",
        HOG_COUNTER.load(Ordering::Relaxed)
    );
    WORKER_DONE.store(true, Ordering::Release);
}

/// AC3.4 — `Sched` SPLIT/DELEGATE conserve CPU time, and a `Sched` subtree revokes cleanly.
fn demo_sched_caps() {
    let mut cs = CSpace::<SLOTS>::new(never_zero);
    // Primordial Sched: 100 CPU-time units per period 10 — the kernel's root CPU-time authority.
    const ROOT: u32 = 100;
    cs.set_root_sched(0, ROOT, 10);

    // SPLIT off two child allotments (the mechanism a parent uses to fund children).
    cs.split(0, 1, 30)
        .unwrap_or_else(|e| fatal_err("split→child A", e));
    cs.split(0, 2, 20)
        .unwrap_or_else(|e| fatal_err("split→child B", e));
    conserved(&cs, ROOT, "after SPLIT");

    // DELEGATE 10 units from child A to child B (the passive-server transfer primitive).
    cs.delegate(1, 2, 10)
        .unwrap_or_else(|e| fatal_err("delegate A→B", e));
    conserved(&cs, ROOT, "after DELEGATE");
    let (root, a, b) = (cap_units(&cs, 0), cap_units(&cs, 1), cap_units(&cs, 2));
    kprintln!(
        "[praesidium] sched: SPLIT/DELEGATE monotonic — root {root} + A {a} + B {b} = {ROOT} conserved (AC3.4)"
    );

    // A Sched subtree revokes leaf-first with no frame-zeroing, and reclaims the root's budget.
    cs.revoke(0)
        .unwrap_or_else(|e| fatal_err("revoke root Sched", e));
    if cs.resolve(1).is_ok() || cs.resolve(2).is_ok() {
        fatal("REVOKE left a child Sched alive");
    }
    if cap_units(&cs, 0) != ROOT {
        fatal("REVOKE did not reclaim the root Sched budget");
    }
    kprintln!("[praesidium] sched: Sched subtree revoked cleanly, budget reclaimed to {ROOT}");
}

/// AC3.1 + AC3.2 — cooperative interleaving and budget-gated runnability, on the executor.
fn demo_executor() {
    // Shared record of the order in which tasks make progress (proves interleaving).
    let log: Rc<RefCell<Vec<char>>> = Rc::new(RefCell::new(Vec::new()));
    let mut ex = Executor::new();

    // Task A is bound to a small Sched budget (3 units); Task B to an ample one. Both want 5
    // steps. In a real system A/B's budgets come from Sched caps like those above; here we
    // build the Budgets directly (binding a Sched cap to a task is P4's job).
    ex.spawn(Task::new(worker(log.clone(), 'A', 5), Budget::new(3, 100)));
    ex.spawn(Task::new(
        worker(log.clone(), 'B', 5),
        Budget::new(100, 100),
    ));

    ex.run_until_idle();

    let count = |c: char| log.borrow().iter().filter(|&&x| x == c).count();
    // AC3.1: progress interleaves (A,B,A,B,...) — the executor did not run A to completion
    // before starting B. Round-robin over the ready queue guarantees the alternation.
    {
        let l = log.borrow();
        if l.len() < 4 || !(l[0] == 'A' && l[1] == 'B' && l[2] == 'A' && l[3] == 'B') {
            fatal("tasks did not interleave cooperatively");
        }
    }
    kprintln!(
        "[praesidium] sched: cooperative yields interleaved 2 tasks ({} steps woven) (AC3.1)",
        count('A') + count('B')
    );

    // AC3.2: A is gated at its 3-unit budget while B (ample budget) completes; A is parked.
    if count('A') != 3 || count('B') != 5 || ex.task_count() != 1 || ex.parked_count() != 1 {
        fatal("Sched budget did not gate runnability as expected");
    }
    kprintln!(
        "[praesidium] sched: budget gated task A at 3/5 steps (depleted ⇒ parked), B ran to 5 (AC3.2)"
    );

    // Replenish the period: A's budget refreshes and it runs to completion (CAP-SCHED-1 lifts).
    ex.replenish();
    ex.run_until_idle();
    if count('A') != 5 || ex.task_count() != 0 {
        fatal("replenishment did not resume the gated task");
    }
    kprintln!(
        "[praesidium] sched: replenishment resumed task A to completion (5/5), all tasks done"
    );
}

/// A demo task: append `mark` to the shared log `iters` times, yielding cooperatively between
/// each step so the executor can interleave other work.
async fn worker(log: Rc<RefCell<Vec<char>>>, mark: char, iters: usize) {
    for _ in 0..iters {
        log.borrow_mut().push(mark); // borrow released before the await (no lock across yield)
        yield_now().await;
    }
}

/// CPU-time units (capacity) a `Sched` cap currently holds.
fn cap_units<const N: usize>(cs: &CSpace<N>, c: usize) -> u32 {
    Budget::from_cap(
        &cs.resolve(c)
            .unwrap_or_else(|e| fatal_err("resolve Sched", e)),
    )
    .capacity
}

/// Fail closed unless the whole `Sched` tree still sums to `expected` (no CPU time created).
fn conserved<const N: usize>(cs: &CSpace<N>, expected: u32, when: &str) {
    let total = cap_units(cs, 0) + cap_units(cs, 1) + cap_units(cs, 2);
    if total != expected {
        kprintln!("[praesidium] FATAL: sched: budget not conserved {when}: {total} != {expected}");
        crate::arch::halt();
    }
}
