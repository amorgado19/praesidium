//! # ipc — Praesidium synchronous capability IPC (ADR-0004, P4)
//!
//! The **portable, host-testable** half of IPC: the `Endpoint` rendezvous state machine, the
//! fixed **register message** format (with hostile-input validation), and the `Notification`
//! signal/wait logic. It is arch-free and executor-free — it decides *who pairs with whom* and
//! *what a message means*, but never touches wakers, page tables, or capabilities directly. The
//! kernel integration layer (`kernel/src/ipc/`) drives it: on a rendezvous match it performs the
//! actual register copy + cross-CSpace `GRANT` (cap-core) + single-use `Reply` mint (cap-core),
//! wakes the peer via the P3 executor's `endpoint_waker`, and runs the callee on the caller's
//! `Sched` budget (DELEGATE).
//!
//! Message payloads are **~4 registers, registers-only** (architect ruling): no per-task buffer
//! to zero/validate on the hot path, and the `Frame`-GRANT boundary for bulk data is sharp and
//! explicit. The [`MessageInfo`] decoder is the hostile-input parser (a `cargo fuzz` target):
//! a malicious caller's descriptor word is bounds-checked, never blind-trusted to index anything.
#![cfg_attr(not(test), no_std)]

extern crate alloc;

pub mod endpoint;
pub mod message;
pub mod notification;

pub use endpoint::{EndpointState, RecvOutcome, SendOutcome};
pub use message::{CapXfer, Message, MessageInfo, MsgError, MAX_CAPS, MSG_REGS};
pub use notification::{NotificationState, SignalOutcome, WaitOutcome};

/// Identity of a blocked party in a rendezvous. The kernel maps it to an executor task (and its
/// `endpoint_waker`); the pure logic here only needs an opaque, comparable id.
pub type PartyId = u64;
