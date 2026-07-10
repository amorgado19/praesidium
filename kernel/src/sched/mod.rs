//! Executor + capability-scheduling integration (P3a).
//!
//! Wires the pure `sched` crate (the cooperative async executor) and `cap-core`'s `Sched`
//! budget model to the running kernel, and runs the P3a boot demo asserting the gates:
//!  - **AC3.4** — `Sched` SPLIT/DELEGATE are monotonic (CPU time conserved), and a `Sched`
//!    subtree revokes cleanly (destroying a `Sched` never touches frames).
//!  - **AC3.1** — the executor advances multiple `Future`s cooperatively; their `.await`
//!    yields interleave them rather than running one to completion first.
//!  - **AC3.2** — a task's `Sched` budget gates runnability (CAP-SCHED-1): a depleted task is
//!    parked, not polled, until replenishment.
//!
//! The **preemptive fallback** (AC3.3) and its context-switch machinery are P3b; here the
//! executor is driven purely cooperatively and replenishment is a logical period tick. No
//! capability is fabricated here — every `Cap`/`Budget` comes from `cap-core` (CAP-RUST-1).
//!
//! SCH-T8 (endpoint-waker hook): the mechanism P4 will use lives in the `sched` crate
//! ([`sched::Executor::endpoint_waker`]); P3a builds no IPC, so nothing is wired to it yet.

use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::RefCell;

use cap_core::{Budget, CSpace, CapError};
use sched::{yield_now, Executor, Task};

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

/// Run the P3a milestone: `Sched` capability accounting + the cooperative executor. Prints
/// `PRAESIDIUM-P3A-OK` on success; any violated invariant fails the boot closed.
pub fn run() {
    demo_sched_caps();
    demo_executor();
    kprintln!("[praesidium] PRAESIDIUM-P3A-OK");
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
