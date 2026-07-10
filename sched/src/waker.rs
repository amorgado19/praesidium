//! Wakers + the shared ready queue.
//!
//! A woken task is one whose id sits in the [`ReadyQueue`]; the executor drains that queue.
//! Each task's [`Waker`] holds its id + a handle to the queue, so waking = "push my id back."
//! P3a wakes only from within `poll` (cooperative `yield_now`); the same mechanism is what
//! P3b's timer and P4's IPC endpoints will use (SCH-T8) — hence the queue is behind a lock and
//! the waker is `Send + Sync`, ready for an interrupt-context waker without a redesign.

use alloc::collections::VecDeque;
use alloc::sync::Arc;
use alloc::task::Wake;
use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicBool, Ordering};
use core::task::Waker;

use crate::task::TaskId;

/// The set of runnable task ids, FIFO. Shared (`Arc`) between the executor and every waker.
pub(crate) struct ReadyQueue {
    inner: SpinLock<VecDeque<TaskId>>,
}

impl ReadyQueue {
    pub(crate) fn new() -> Self {
        Self {
            inner: SpinLock::new(VecDeque::new()),
        }
    }

    /// Mark `id` runnable.
    pub(crate) fn push(&self, id: TaskId) {
        self.inner.lock().push_back(id);
    }

    /// Take the next runnable task id, if any.
    pub(crate) fn pop(&self) -> Option<TaskId> {
        self.inner.lock().pop_front()
    }
}

/// Build a `Waker` that re-enqueues `id` onto `queue` when woken.
pub(crate) fn make_waker(id: TaskId, queue: Arc<ReadyQueue>) -> Waker {
    Waker::from(Arc::new(TaskWaker { id, queue }))
}

struct TaskWaker {
    id: TaskId,
    queue: Arc<ReadyQueue>,
}

impl Wake for TaskWaker {
    fn wake(self: Arc<Self>) {
        self.queue.push(self.id);
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.queue.push(self.id);
    }
}

// ---- a minimal spinlock (the sched crate is arch-free and cannot use the kernel's) --------

/// Tiny test-and-set spinlock guarding the ready queue. The critical sections are a single
/// `push`/`pop` and never span a `.await` (the executor's run loop is synchronous), so it can
/// never hold across a yield — the GC-07 hazard does not arise here by construction.
struct SpinLock<T> {
    locked: AtomicBool,
    data: UnsafeCell<T>,
}

// SAFETY: `lock()` hands out access only after winning the atomic flag, so at most one path
// touches `data` at a time; `T: Send` because the guarded value effectively moves to whoever
// holds the lock. This mirrors kernel::sync::SpinLock.
unsafe impl<T: Send> Sync for SpinLock<T> {}

impl<T> SpinLock<T> {
    const fn new(data: T) -> Self {
        Self {
            locked: AtomicBool::new(false),
            data: UnsafeCell::new(data),
        }
    }

    fn lock(&self) -> SpinGuard<'_, T> {
        while self
            .locked
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            core::hint::spin_loop();
        }
        SpinGuard { lock: self }
    }
}

struct SpinGuard<'a, T> {
    lock: &'a SpinLock<T>,
}

impl<T> core::ops::Deref for SpinGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        // SAFETY: holding the guard guarantees exclusive access to `data`.
        unsafe { &*self.lock.data.get() }
    }
}

impl<T> core::ops::DerefMut for SpinGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: holding the guard guarantees exclusive access to `data`.
        unsafe { &mut *self.lock.data.get() }
    }
}

impl<T> Drop for SpinGuard<'_, T> {
    fn drop(&mut self) {
        self.lock.locked.store(false, Ordering::Release);
    }
}
