//! # sched — Praesidium async executor + capability scheduling (ADR-0003, P3)
//!
//! The **Tier-1 cooperative core**: a `no_std` async executor that polls `Future` tasks and
//! advances them at their explicit `.await` yield points (cheap continuations, not register
//! swaps — DEC-0003-1). Each task is bound to a [`Budget`](cap_core::Budget) (its `Sched`
//! capability's CPU-time allotment); the executor **gates runnability** on it (CAP-SCHED-1): a
//! task whose budget is depleted is parked, not polled, until the next replenishment. Budget
//! subdivision (SPLIT/DELEGATE) lives in `cap-core`; this crate only *consumes* a budget.
//!
//! This is arch-free and driven purely cooperatively (P3a): replenishment is a logical tick.
//! The **Tier-2 preemptive fallback** — a hardware timer that context-switches away from a
//! non-yielding task — is P3b, and lands in the kernel integration layer behind the arch seam;
//! nothing here needs to change for it (the executor stays the cooperative hot path).
#![cfg_attr(not(test), no_std)]

extern crate alloc;

pub mod executor;
pub mod task;
mod waker;

pub use executor::Executor;
pub use task::{yield_now, Task, TaskId};
