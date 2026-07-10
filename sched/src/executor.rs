//! The cooperative executor: a run loop that polls ready tasks while their budget lasts.

use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::task::{Context, Poll};

use crate::task::{Task, TaskId};
use crate::waker::{make_waker, ReadyQueue};

/// A single-CPU cooperative executor (ADR-0003 Tier 1). Holds the live tasks and a shared
/// ready queue; [`run_until_idle`](Self::run_until_idle) drains the queue, polling each ready
/// task that still has CPU-time budget. A task whose `Sched` budget is depleted is **parked**
/// (CAP-SCHED-1) until [`replenish`](Self::replenish) starts a new period.
///
/// P3a drives replenishment logically (the caller ticks periods); P3b binds it to a hardware
/// timer and adds the preemptive fallback around this same executor.
pub struct Executor {
    tasks: BTreeMap<TaskId, Task>,
    ready: Arc<ReadyQueue>,
    /// Tasks parked because their budget is depleted — re-enqueued on the next replenishment.
    parked: Vec<TaskId>,
}

impl Default for Executor {
    fn default() -> Self {
        Self::new()
    }
}

impl Executor {
    #[must_use]
    pub fn new() -> Self {
        Self {
            tasks: BTreeMap::new(),
            ready: Arc::new(ReadyQueue::new()),
            parked: Vec::new(),
        }
    }

    /// Admit a task and mark it runnable. Returns its id.
    pub fn spawn(&mut self, task: Task) -> TaskId {
        let id = task.id();
        self.tasks.insert(id, task);
        self.ready.push(id);
        id
    }

    /// Poll ready tasks until none remain runnable this period — i.e. every task has either
    /// completed or been parked on a depleted budget. Cooperative: a task advances until it
    /// `.await`s a yield point, then the next ready task runs (round-robin via the queue).
    pub fn run_until_idle(&mut self) {
        while let Some(id) = self.ready.pop() {
            let Some(task) = self.tasks.get_mut(&id) else {
                continue; // stale wake for an already-completed task
            };
            // Runnability gate (CAP-SCHED-1): a depleted Sched ⇒ not scheduled. Park it; the
            // next replenishment re-enqueues it. No default timeslice — no budget, no run.
            if task.budget.is_depleted() {
                self.park(id);
                continue;
            }
            let waker = make_waker(id, self.ready.clone());
            let mut cx = Context::from_waker(&waker);
            match task.poll(&mut cx) {
                Poll::Ready(()) => {
                    self.tasks.remove(&id); // completed — drop the future
                }
                Poll::Pending => {
                    // Charge this slice of CPU time to the task's budget. If that depletes it,
                    // the wake it just queued (e.g. via yield_now) will be popped, seen
                    // depleted, and parked — so it stops advancing until replenishment.
                    task.budget.charge(1);
                }
            }
        }
    }

    /// Start a new sporadic-server period: replenish every task's budget and re-admit the
    /// tasks that were parked on a depleted one. (P3b calls this from the timer per period.)
    pub fn replenish(&mut self) {
        for task in self.tasks.values_mut() {
            task.budget.replenish();
        }
        for id in self.parked.drain(..) {
            if self.tasks.contains_key(&id) {
                self.ready.push(id);
            }
        }
    }

    /// SCH-T8 — the endpoint-waker hook (the P4 seam). Returns a `Waker` that re-admits task
    /// `id` to the run queue when signalled. A blocked async operation — a P4 `Endpoint`
    /// `call`/`recv`, a `Notification` — will hold this waker and fire it when its peer is
    /// ready, so the task is re-polled without any busy-wait. P3a builds no IPC, so this is the
    /// plumbing only; the executor mechanism it hands out needs no change when IPC lands.
    #[must_use]
    pub fn endpoint_waker(&self, id: TaskId) -> core::task::Waker {
        make_waker(id, self.ready.clone())
    }

    /// Number of tasks not yet completed (running, ready, or parked).
    #[must_use]
    pub fn task_count(&self) -> usize {
        self.tasks.len()
    }

    /// Number of tasks currently parked on a depleted budget.
    #[must_use]
    pub fn parked_count(&self) -> usize {
        self.parked.len()
    }

    fn park(&mut self, id: TaskId) {
        if !self.parked.contains(&id) {
            self.parked.push(id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task::yield_now;
    use cap_core::Budget;
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::vec::Vec;

    /// A future that appends `mark`, `iters` times, yielding cooperatively between each.
    fn worker(
        log: Rc<RefCell<Vec<char>>>,
        mark: char,
        iters: usize,
    ) -> impl core::future::Future<Output = ()> {
        async move {
            for _ in 0..iters {
                log.borrow_mut().push(mark);
                yield_now().await;
            }
        }
    }

    #[test]
    fn cooperative_yields_interleave_tasks() {
        // Two well-budgeted tasks must interleave round-robin, not run to completion serially.
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut ex = Executor::new();
        ex.spawn(Task::new(
            worker(log.clone(), 'A', 3),
            Budget::new(1000, 1000),
        ));
        ex.spawn(Task::new(
            worker(log.clone(), 'B', 3),
            Budget::new(1000, 1000),
        ));
        ex.run_until_idle();
        assert_eq!(
            *log.borrow(),
            vec!['A', 'B', 'A', 'B', 'A', 'B'],
            "cooperative yields must interleave, not serialize"
        );
        assert_eq!(ex.task_count(), 0, "both tasks completed");
    }

    #[test]
    fn budget_gates_runnability_and_replenish_frees_it() {
        // Task A has budget for only 3 slices; B has plenty. A must stall at 3 while B finishes.
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut ex = Executor::new();
        ex.spawn(Task::new(worker(log.clone(), 'A', 5), Budget::new(3, 100)));
        ex.spawn(Task::new(
            worker(log.clone(), 'B', 5),
            Budget::new(100, 100),
        ));
        ex.run_until_idle();

        let count = |c: char| log.borrow().iter().filter(|&&x| x == c).count();
        assert_eq!(count('A'), 3, "A gated at its 3-unit budget (CAP-SCHED-1)");
        assert_eq!(count('B'), 5, "B had budget to complete");
        assert_eq!(ex.task_count(), 1, "A remains, parked on a depleted budget");
        assert_eq!(ex.parked_count(), 1);

        // New period: A's budget replenishes and it resumes to completion.
        ex.replenish();
        ex.run_until_idle();
        assert_eq!(count('A'), 5, "A finished after replenishment");
        assert_eq!(ex.task_count(), 0);
    }

    #[test]
    fn endpoint_waker_resumes_an_externally_blocked_task() {
        // Models the P4 pattern (SCH-T8): a task blocks on an external event (returning
        // Pending without self-waking), and only an outside signal via `endpoint_waker`
        // re-admits it. No busy-wait: run_until_idle goes idle while the task waits.
        use core::cell::Cell;
        use core::future::Future;
        use core::pin::Pin;
        use core::task::{Context, Poll};

        struct WaitFor {
            ready: Rc<Cell<bool>>,
        }
        impl Future for WaitFor {
            type Output = ();
            fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
                if self.ready.get() {
                    Poll::Ready(())
                } else {
                    Poll::Pending // deliberately does NOT wake — waits for an external signal
                }
            }
        }

        let flag = Rc::new(Cell::new(false));
        let f = flag.clone();
        let mut ex = Executor::new();
        let id = ex.spawn(Task::new(
            async move { WaitFor { ready: f }.await },
            Budget::new(10, 10),
        ));

        ex.run_until_idle();
        assert_eq!(
            ex.task_count(),
            1,
            "task is blocked on the external event, idle"
        );

        // The "endpoint" becomes ready and signals the task's waker.
        let waker = ex.endpoint_waker(id);
        flag.set(true);
        waker.wake();

        ex.run_until_idle();
        assert_eq!(
            ex.task_count(),
            0,
            "external wake re-admitted and completed the task"
        );
    }

    #[test]
    fn zero_budget_task_never_runs() {
        // A task bound to a zero-capacity Sched is never runnable, even across replenishment.
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut ex = Executor::new();
        ex.spawn(Task::new(worker(log.clone(), 'Z', 3), Budget::new(0, 10)));
        ex.run_until_idle();
        assert!(
            log.borrow().is_empty(),
            "zero-budget task must not be polled"
        );
        assert_eq!(ex.task_count(), 1);
        ex.replenish();
        ex.run_until_idle();
        assert!(
            log.borrow().is_empty(),
            "still zero capacity ⇒ still never runs"
        );
    }
}
