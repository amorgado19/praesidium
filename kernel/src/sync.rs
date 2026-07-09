//! Minimal synchronization primitives.
//!
//! A test-and-set [`SpinLock`] giving kernel globals a sound `Sync` interface. In P1
//! the kernel is single-CPU with interrupts masked, so contention never actually
//! occurs; the lock exists so a `static` mutable allocator is well-typed rather than
//! `static mut`. Preemption-safe locking and the no-lock-across-`.await` rule arrive
//! with the executor (P3).

use core::cell::UnsafeCell;
use core::ops::{Deref, DerefMut};
use core::sync::atomic::{AtomicBool, Ordering};

/// A spinlock protecting a `T`.
pub struct SpinLock<T> {
    locked: AtomicBool,
    data: UnsafeCell<T>,
}

// SAFETY: `lock()` hands out at most one `SpinGuard` at a time (the atomic flag
// serializes acquisition), so all access to `data` is exclusive. `T: Send` is
// required because the guarded value effectively moves between whoever holds the lock.
unsafe impl<T: Send> Sync for SpinLock<T> {}

impl<T> SpinLock<T> {
    /// Create an unlocked lock around `data`.
    pub const fn new(data: T) -> Self {
        Self {
            locked: AtomicBool::new(false),
            data: UnsafeCell::new(data),
        }
    }

    /// Acquire the lock, spinning until it is free.
    pub fn lock(&self) -> SpinGuard<'_, T> {
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

/// RAII guard; releases the lock on drop.
pub struct SpinGuard<'a, T> {
    lock: &'a SpinLock<T>,
}

impl<T> Deref for SpinGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        // SAFETY: holding the guard guarantees exclusive access to `data`.
        unsafe { &*self.lock.data.get() }
    }
}

impl<T> DerefMut for SpinGuard<'_, T> {
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
