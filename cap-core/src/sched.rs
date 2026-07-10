//! CPU time as a capability: the `Sched` **budget model** (SPEC-CAP §6, ADR-0003 DEC-0003-2).
//!
//! A [`Budget`] is a sporadic-server-style allotment — up to `capacity` CPU-time units per
//! `period`, replenished each period. It is the value carried *inside* a `Sched` capability
//! (`RawCap.size`/`watermark`/`aux`), so budget is **per-cap**: a `Sched` cannot be COPY/MINT'd
//! (that would fork the allotment = CPU-time forgery, mirroring the Untyped watermark), and CPU
//! time is subdivided only through [`split`](Budget::split) / [`delegate`](Budget::delegate),
//! which are **monotonic** — they move capacity between allotments and never create it
//! (DEC-0003-4, the passive-server enabler). This is pure logic; `cargo test -p cap-core`
//! proves conservation.

use crate::cap::RawCap;

/// A CPU-time allotment: `capacity` units of CPU per `period`, `consumed` so far this period.
/// "Units" and "period" are abstract here (P3a drives them by a logical tick; P3b binds them
/// to a hardware timer); the conservation guarantees hold regardless of the unit.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Budget {
    /// CPU-time units granted per period (the "budget").
    pub capacity: u32,
    /// Units consumed in the current period (`>= capacity` ⇒ depleted).
    pub consumed: u32,
    /// Replenishment period (units of the same abstract clock; carried for P3b's timer).
    pub period: u32,
}

impl Budget {
    /// A fresh, fully-available allotment.
    #[must_use]
    pub const fn new(capacity: u32, period: u32) -> Self {
        Self {
            capacity,
            consumed: 0,
            period,
        }
    }

    /// Read the allotment out of a `Sched` capability record. (Caller ensures `cap` is a
    /// `Sched`; the field layout is Sched-specific — see [`RawCap`].)
    #[must_use]
    pub const fn from_cap(cap: &RawCap) -> Self {
        Self {
            capacity: cap.size,
            consumed: cap.watermark,
            period: cap.aux,
        }
    }

    /// Write the allotment back into a `Sched` capability record.
    pub fn write_into(&self, cap: &mut RawCap) {
        cap.size = self.capacity;
        cap.watermark = self.consumed;
        cap.aux = self.period;
    }

    /// Is this allotment out of budget for the current period? A task bound to a depleted
    /// `Sched` is not runnable (CAP-SCHED-1) until [`replenish`](Self::replenish).
    #[must_use]
    pub const fn is_depleted(&self) -> bool {
        self.consumed >= self.capacity
    }

    /// Units still available to run this period.
    #[must_use]
    pub const fn remaining(&self) -> u32 {
        self.capacity.saturating_sub(self.consumed)
    }

    /// Charge `n` units of CPU time against this period (saturating — never wraps, GC-03).
    pub fn charge(&mut self, n: u32) {
        self.consumed = self.consumed.saturating_add(n);
    }

    /// Replenish for a new period: consumed resets to zero (sporadic-server replenishment).
    pub fn replenish(&mut self) {
        self.consumed = 0;
    }

    /// SPLIT: carve `amount` capacity into a fresh child allotment, reducing this one by
    /// exactly the same amount. Returns `None` (leaving `self` untouched) if `amount` is zero
    /// or exceeds the still-available (un-consumed) capacity — you cannot split off budget you
    /// have already spent. **Conservation:** `self.capacity` before == `self.capacity` after +
    /// `child.capacity` (DEC-0003-4: no CPU time is created).
    #[must_use]
    pub fn split(&mut self, amount: u32) -> Option<Budget> {
        if amount == 0 || amount > self.remaining() {
            return None;
        }
        self.capacity -= amount;
        Some(Budget {
            capacity: amount,
            consumed: 0,
            period: self.period,
        })
    }

    /// DELEGATE: transfer `amount` capacity from `self` into `dst`. Returns `false` (leaving
    /// both untouched) if `amount` exceeds `self`'s available capacity **or** would overflow
    /// `dst`'s `u32` capacity. **Conservation:** `self.capacity + dst.capacity` is unchanged.
    /// The passive-server primitive: a caller hands budget to a server so the server runs on
    /// the caller's time (P4 wires it to IPC).
    #[must_use]
    pub fn delegate(&mut self, dst: &mut Budget, amount: u32) -> bool {
        if amount == 0 || amount > self.remaining() {
            return false;
        }
        // Credit with a CHECKED add: a `dst` that has itself received delegations from other
        // Sched trees could be near u32::MAX, and a raw `+=` would trap under overflow-checks
        // (a kernel DoS). Refuse on overflow rather than saturate — saturating would silently
        // destroy CPU time (source debited more than dest credited), breaking conservation.
        // Compute the sum before debiting the source so the transfer stays all-or-nothing.
        let Some(credited) = dst.capacity.checked_add(amount) else {
            return false;
        };
        self.capacity -= amount;
        dst.capacity = credited;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn charge_depletes_and_replenish_restores() {
        let mut b = Budget::new(3, 10);
        assert!(!b.is_depleted());
        b.charge(1);
        b.charge(1);
        assert_eq!(b.remaining(), 1);
        assert!(!b.is_depleted());
        b.charge(1);
        assert!(b.is_depleted(), "3/3 consumed ⇒ depleted");
        assert_eq!(b.remaining(), 0);
        b.replenish();
        assert!(!b.is_depleted(), "replenish restores the allotment");
        assert_eq!(b.remaining(), 3);
    }

    #[test]
    fn charge_saturates_never_wraps() {
        let mut b = Budget::new(2, 10);
        b.charge(u32::MAX); // a runaway charge must not wrap the counter (GC-03)
        assert!(b.is_depleted());
        assert_eq!(b.remaining(), 0);
    }

    #[test]
    fn split_conserves_total_capacity() {
        let mut parent = Budget::new(100, 50);
        let child = parent.split(30).expect("30 <= 100 available");
        assert_eq!(child.capacity, 30);
        assert_eq!(child.consumed, 0);
        assert_eq!(child.period, 50, "child inherits the period");
        assert_eq!(parent.capacity, 70);
        // Conservation: nothing created.
        assert_eq!(parent.capacity + child.capacity, 100);
    }

    #[test]
    fn split_refuses_zero_and_over_available() {
        let mut b = Budget::new(10, 5);
        assert!(b.split(0).is_none(), "zero split refused");
        assert!(b.split(11).is_none(), "over-capacity split refused");
        b.charge(4); // 4 consumed ⇒ only 6 available
        assert!(
            b.split(7).is_none(),
            "cannot split off already-consumed budget"
        );
        assert_eq!(b.capacity, 10, "refused split leaves capacity untouched");
        let c = b.split(6).expect("6 <= 6 available");
        assert_eq!(c.capacity, 6);
        assert_eq!(b.capacity, 4);
        assert_eq!(b.consumed, 4, "consumed preserved across split");
    }

    #[test]
    fn delegate_conserves_and_is_monotonic() {
        let mut a = Budget::new(100, 10);
        let mut b = Budget::new(20, 10);
        assert!(a.delegate(&mut b, 30));
        assert_eq!(a.capacity, 70);
        assert_eq!(b.capacity, 50);
        assert_eq!(a.capacity + b.capacity, 120, "total conserved");
        // Over-available delegation refused, both untouched.
        assert!(!a.delegate(&mut b, 71));
        assert_eq!(a.capacity, 70);
        assert_eq!(b.capacity, 50);
    }

    #[test]
    fn delegate_refuses_destination_overflow_all_or_nothing() {
        // A destination near u32::MAX (e.g. concentrated by cross-tree delegations) must not
        // overflow on credit — that would trap under overflow-checks (a kernel DoS). Refuse,
        // leaving both budgets untouched (all-or-nothing); do NOT saturate (that loses budget).
        let mut src = Budget::new(u32::MAX, 10);
        let mut dst = Budget {
            capacity: u32::MAX - 3,
            consumed: 0,
            period: 10,
        };
        assert!(
            !src.delegate(&mut dst, 10),
            "credit would overflow ⇒ refuse"
        );
        assert_eq!(src.capacity, u32::MAX, "source untouched on refusal");
        assert_eq!(
            dst.capacity,
            u32::MAX - 3,
            "destination untouched on refusal"
        );
        // The exact-fit transfer (fills dst to u32::MAX) is still allowed and conserves total.
        assert!(src.delegate(&mut dst, 3));
        assert_eq!(dst.capacity, u32::MAX);
        assert_eq!(src.capacity, u32::MAX - 3);
    }

    #[test]
    fn cap_roundtrip_preserves_budget() {
        use crate::cap::CapType;
        let b = Budget {
            capacity: 42,
            consumed: 7,
            period: 99,
        };
        let mut cap = RawCap {
            cap_type: CapType::Sched,
            ..RawCap::NULL
        };
        b.write_into(&mut cap);
        assert_eq!(Budget::from_cap(&cap), b);
    }
}
