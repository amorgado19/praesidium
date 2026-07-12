//! The `Notification` object (IPC-T3): a payload-less async signal (SIGNAL/WAIT). Unlike an
//! `Endpoint`, a signal is **fire-and-forget** — it wakes a waiter if one is present, otherwise
//! it latches a pending bit that the next `wait` consumes. Used for IRQ→driver signalling (P9).
//!
//! Pure logic: the kernel wakes the returned party via the executor. A single pending bit
//! (binary semaphore) is the P4 model; per-badge signal words are a later refinement.

use alloc::collections::VecDeque;

use crate::PartyId;

/// The state of one Notification: a latched-pending flag plus any parties blocked in `wait`.
/// Invariant: `pending` and a non-empty `waiters` never coexist (a signal delivers to a waiter
/// rather than latching if one is present).
#[derive(Default)]
pub struct NotificationState {
    pending: bool,
    waiters: VecDeque<PartyId>,
}

/// The result of a [`NotificationState::signal`].
pub enum SignalOutcome {
    /// A waiter was present and should be woken by the kernel.
    Woke(PartyId),
    /// No waiter; the signal is latched pending for the next `wait`.
    Latched,
}

/// The result of a [`NotificationState::wait`].
pub enum WaitOutcome {
    /// A signal was already pending (consumed now); the waiter proceeds without blocking.
    Ready,
    /// No signal; the waiter is queued until a future `signal`.
    Queued,
}

impl NotificationState {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Signal the notification: wake the oldest waiter if any, else latch a pending bit.
    pub fn signal(&mut self) -> SignalOutcome {
        if let Some(p) = self.waiters.pop_front() {
            SignalOutcome::Woke(p)
        } else {
            self.pending = true;
            SignalOutcome::Latched
        }
    }

    /// Wait for a signal: if one is pending, consume it and proceed (`Ready`); else queue.
    pub fn wait(&mut self, by: PartyId) -> WaitOutcome {
        if self.pending {
            self.pending = false;
            WaitOutcome::Ready
        } else {
            self.waiters.push_back(by);
            WaitOutcome::Queued
        }
    }

    /// Remove `party` from the wait queue (abort a blocked wait on revoke/timeout).
    pub fn cancel(&mut self, party: PartyId) -> bool {
        let before = self.waiters.len();
        self.waiters.retain(|&p| p != party);
        before != self.waiters.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wait_then_signal_wakes_the_waiter() {
        let mut n = NotificationState::new();
        assert!(matches!(n.wait(5), WaitOutcome::Queued));
        assert!(matches!(n.signal(), SignalOutcome::Woke(5)));
    }

    #[test]
    fn signal_then_wait_is_ready_and_consumes_the_pending_bit() {
        let mut n = NotificationState::new();
        assert!(matches!(n.signal(), SignalOutcome::Latched));
        assert!(matches!(n.wait(5), WaitOutcome::Ready), "pending consumed");
        // The pending bit is one-shot: a second wait finds nothing and queues.
        assert!(matches!(n.wait(6), WaitOutcome::Queued));
    }

    #[test]
    fn cancel_removes_a_blocked_waiter() {
        let mut n = NotificationState::new();
        n.wait(9);
        assert!(n.cancel(9));
        assert!(!n.cancel(9));
        // Signalling now latches (the cancelled waiter is gone).
        assert!(matches!(n.signal(), SignalOutcome::Latched));
    }
}
