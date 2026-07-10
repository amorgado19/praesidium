//! A schedulable async task: a pinned `Future` plus its `Sched` budget.

use alloc::boxed::Box;
use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicU64, Ordering};
use core::task::{Context, Poll};

use cap_core::Budget;

/// A process-wide unique task identity. Used as the key into the executor's task table and
/// carried by wakers to name the task to re-enqueue.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct TaskId(u64);

impl TaskId {
    fn allocate() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        // Relaxed is sufficient: we need uniqueness, not ordering against other memory.
        TaskId(NEXT.fetch_add(1, Ordering::Relaxed))
    }
}

/// A unit of cooperatively-scheduled work: a type-erased `Future` bound to a CPU-time
/// [`Budget`] (its `Sched` capability's allotment). The executor polls it while the budget
/// lasts and drops it when it completes.
pub struct Task {
    id: TaskId,
    future: Pin<Box<dyn Future<Output = ()>>>,
    /// The task's runnability budget (a copy of its bound `Sched`; the executor charges +
    /// replenishes it). Depleted ⇒ the task is parked, not polled (CAP-SCHED-1).
    pub(crate) budget: Budget,
}

impl Task {
    /// Bind `future` to CPU-time `budget`, giving it a fresh identity.
    #[must_use]
    pub fn new(future: impl Future<Output = ()> + 'static, budget: Budget) -> Self {
        Self {
            id: TaskId::allocate(),
            future: Box::pin(future),
            budget,
        }
    }

    /// This task's identity.
    #[must_use]
    pub fn id(&self) -> TaskId {
        self.id
    }

    /// Poll the underlying future once with `cx`.
    pub(crate) fn poll(&mut self, cx: &mut Context<'_>) -> Poll<()> {
        self.future.as_mut().poll(cx)
    }
}

/// Cooperatively yield to the executor **once**: on first poll this re-enqueues the current
/// task (via its waker) and returns `Pending`, so the executor advances other ready tasks
/// before coming back; on the next poll it completes. This is the explicit yield point that
/// makes a task a good citizen of the cooperative core (ADR-0003 Tier 1).
pub fn yield_now() -> impl Future<Output = ()> {
    YieldNow { yielded: false }
}

struct YieldNow {
    yielded: bool,
}

impl Future for YieldNow {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if self.yielded {
            Poll::Ready(())
        } else {
            self.yielded = true;
            // Re-arm ourselves so the executor re-polls us after servicing other tasks.
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}
