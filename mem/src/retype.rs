//! Minimal `Untyped` retype accounting — the non-capability seam between raw frame
//! allocation and the capability model (SPEC-CAP §5, §8).
//!
//! An [`Untyped`] is a physical frame region consumed **monotonically** (seL4-style:
//! RETYPE bump-allocates within it; there is no per-object free — reclamation happens
//! by REVOKE, which resets the whole region). In P1 this is a plain accounting
//! mechanism; **P2 wraps it as `Cap<Untyped>` → `Cap<Frame/CNode/Endpoint>` with
//! CSpace + the derivation tree**. Zero-on-retype (CAP-MEM-2) is enforced by the
//! kernel integration layer, which zeroes the returned frames (via the HHDM) before
//! the object is observable; this accounting layer only hands out the frame range.

use crate::frame::Pfn;

/// A retypable region of physical frames with a bump watermark.
#[derive(Clone, Copy, Debug)]
pub struct Untyped {
    base: Pfn,
    frames: Pfn,
    watermark: Pfn,
}

impl Untyped {
    /// A region of `frames` frames starting at frame `base`.
    #[must_use]
    pub fn new(base: Pfn, frames: Pfn) -> Self {
        Self {
            base,
            frames,
            watermark: 0,
        }
    }

    /// First frame of the region.
    #[must_use]
    pub fn base(&self) -> Pfn {
        self.base
    }

    /// Total frames in the region.
    #[must_use]
    pub fn frames(&self) -> Pfn {
        self.frames
    }

    /// Frames consumed so far.
    #[must_use]
    pub fn used(&self) -> Pfn {
        self.watermark
    }

    /// Frames still available to retype.
    #[must_use]
    pub fn remaining(&self) -> Pfn {
        self.frames - self.watermark
    }

    /// Charge `frames_per_obj * count` frames and return the base frame of the run,
    /// or `None` if the budget is insufficient or the size overflows. Monotonic: the
    /// only way to reclaim is [`reset`](Self::reset). The caller (kernel) must zero
    /// the returned frames before the object is observable (CAP-MEM-2).
    pub fn retype(&mut self, frames_per_obj: Pfn, count: Pfn) -> Option<Pfn> {
        let need = frames_per_obj.checked_mul(count)?;
        if need > self.remaining() {
            return None;
        }
        let start = self.base + self.watermark;
        self.watermark += need;
        Some(start)
    }

    /// REVOKE-equivalent reset — reclaims the whole region. P2 gates this behind
    /// capability revocation (destroying all descendants); here it is the raw
    /// mechanism.
    pub fn reset(&mut self) {
        self.watermark = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retype_charges_budget_monotonically() {
        let mut u = Untyped::new(100, 10);
        assert_eq!(u.remaining(), 10);
        assert_eq!(u.retype(1, 4), Some(100));
        assert_eq!(u.used(), 4);
        assert_eq!(u.retype(2, 2), Some(104)); // 2 objs * 2 frames = 4 frames
        assert_eq!(u.used(), 8);
        assert_eq!(u.remaining(), 2);
    }

    #[test]
    fn retype_refuses_over_budget() {
        let mut u = Untyped::new(0, 4);
        assert_eq!(u.retype(1, 5), None); // more than the region
        assert_eq!(u.used(), 0, "a refused retype must not consume budget");
        assert_eq!(u.retype(1, 4), Some(0));
        assert_eq!(u.retype(1, 1), None); // exhausted
    }

    #[test]
    fn retype_size_overflow_is_refused() {
        let mut u = Untyped::new(0, u32::MAX);
        assert_eq!(u.retype(u32::MAX, 2), None); // frames_per_obj * count overflows
        assert_eq!(u.used(), 0);
    }

    #[test]
    fn reset_reclaims() {
        let mut u = Untyped::new(0, 8);
        assert_eq!(u.retype(1, 8), Some(0));
        assert_eq!(u.remaining(), 0);
        u.reset();
        assert_eq!(u.remaining(), 8);
        assert_eq!(u.retype(1, 8), Some(0));
    }
}
