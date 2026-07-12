//! The `Endpoint` rendezvous state machine (IPC-T1): a synchronous meeting point where at most
//! one side ever queues (senders XOR receivers wait, never both — the moment both are present
//! they are matched). Pure logic: it decides *who pairs with whom* and carries the pending
//! message across the wait; the kernel performs the register copy + cross-CSpace GRANT + Reply
//! mint + waker on the [`SendOutcome`]/[`RecvOutcome`], runs the callee on the caller's budget,
//! and — for a `call` — awaits the reply (which is delivered out-of-band via the Reply cap, not
//! back through this queue).

use alloc::collections::VecDeque;

use crate::message::Message;
use crate::PartyId;

/// A party blocked trying to send, holding the message it is offering.
struct SendWaiter {
    party: PartyId,
    msg: Message,
}

/// The rendezvous state of one Endpoint. Invariant: `senders` and `receivers` are never both
/// non-empty (a match drains one against the other immediately).
#[derive(Default)]
pub struct EndpointState {
    senders: VecDeque<SendWaiter>,
    receivers: VecDeque<PartyId>,
}

/// The result of a [`EndpointState::send`].
pub enum SendOutcome {
    /// A receiver was already waiting: the kernel delivers `msg` to `receiver` and wakes it (and,
    /// for a `call`, mints a Reply naming the sender). The sender does not queue.
    Delivered { receiver: PartyId, msg: Message },
    /// No receiver: the sender is now queued on the endpoint. The kernel parks it (a `call`
    /// registers its waker and awaits the reply; a one-way send awaits room).
    Queued,
}

/// The result of a [`EndpointState::recv`].
pub enum RecvOutcome {
    /// A sender was already waiting: here is its message and identity. The kernel delivers it and
    /// (for a `call`) mints a Reply naming `sender`.
    Received { sender: PartyId, msg: Message },
    /// No sender: the receiver is now queued, awaiting a future send.
    Queued,
}

impl EndpointState {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Offer `msg` from `from`. If a receiver is waiting, they are matched immediately
    /// (`Delivered`); otherwise `from` queues as a sender (`Queued`).
    pub fn send(&mut self, from: PartyId, msg: Message) -> SendOutcome {
        if let Some(receiver) = self.receivers.pop_front() {
            SendOutcome::Delivered { receiver, msg }
        } else {
            self.senders.push_back(SendWaiter { party: from, msg });
            SendOutcome::Queued
        }
    }

    /// Wait to receive as `by`. If a sender is waiting, take its message (`Received`); otherwise
    /// `by` queues as a receiver (`Queued`). Senders are matched in FIFO order.
    pub fn recv(&mut self, by: PartyId) -> RecvOutcome {
        if let Some(w) = self.senders.pop_front() {
            RecvOutcome::Received {
                sender: w.party,
                msg: w.msg,
            }
        } else {
            self.receivers.push_back(by);
            RecvOutcome::Queued
        }
    }

    /// Remove `party` from whichever queue it sits in (abort a blocked call/recv). Used when a
    /// blocked caller is REVOKE'd or times out: its in-flight rendezvous is cancelled cleanly so
    /// no stale waiter is ever matched (part of CAP-REVOKE-1 in-flight teardown). Returns whether
    /// a waiter was actually removed.
    pub fn cancel(&mut self, party: PartyId) -> bool {
        let before = self.senders.len() + self.receivers.len();
        self.senders.retain(|w| w.party != party);
        self.receivers.retain(|&p| p != party);
        before != self.senders.len() + self.receivers.len()
    }

    /// No party is blocked on this endpoint.
    #[must_use]
    pub fn is_idle(&self) -> bool {
        self.senders.is_empty() && self.receivers.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Message;

    const CLIENT: PartyId = 1;
    const SERVER: PartyId = 2;

    fn matched_send(o: SendOutcome) -> (PartyId, Message) {
        match o {
            SendOutcome::Delivered { receiver, msg } => (receiver, msg),
            SendOutcome::Queued => panic!("expected Delivered"),
        }
    }
    fn matched_recv(o: RecvOutcome) -> (PartyId, Message) {
        match o {
            RecvOutcome::Received { sender, msg } => (sender, msg),
            RecvOutcome::Queued => panic!("expected Received"),
        }
    }

    #[test]
    fn recv_then_send_delivers_to_the_waiting_receiver() {
        let mut ep = EndpointState::new();
        assert!(matches!(ep.recv(SERVER), RecvOutcome::Queued));
        let (receiver, msg) = matched_send(ep.send(CLIENT, Message::with_data(7, &[42])));
        assert_eq!(receiver, SERVER);
        assert_eq!(msg.data(), &[42]);
        assert!(ep.is_idle(), "both sides matched, nothing left queued");
    }

    #[test]
    fn send_then_recv_hands_the_message_to_the_receiver() {
        let mut ep = EndpointState::new();
        assert!(matches!(
            ep.send(CLIENT, Message::with_data(7, &[99])),
            SendOutcome::Queued
        ));
        let (sender, msg) = matched_recv(ep.recv(SERVER));
        assert_eq!(sender, CLIENT);
        assert_eq!(msg.data(), &[99]);
        assert!(ep.is_idle());
    }

    #[test]
    fn senders_are_matched_fifo() {
        let mut ep = EndpointState::new();
        ep.send(10, Message::with_data(0, &[1]));
        ep.send(11, Message::with_data(0, &[2]));
        let (first, m1) = matched_recv(ep.recv(SERVER));
        let (second, m2) = matched_recv(ep.recv(SERVER));
        assert_eq!((first, m1.data()[0]), (10, 1));
        assert_eq!((second, m2.data()[0]), (11, 2));
    }

    #[test]
    fn cancel_removes_a_blocked_waiter() {
        // A blocked caller that is revoked/times-out is removed and never matched.
        let mut ep = EndpointState::new();
        ep.send(CLIENT, Message::empty(0));
        assert!(ep.cancel(CLIENT), "the queued sender was removed");
        assert!(!ep.cancel(CLIENT), "cancelling again removes nothing");
        // A later recv finds no sender (the cancelled call is gone) and queues instead.
        assert!(matches!(ep.recv(SERVER), RecvOutcome::Queued));
    }
}
