//! Binary buddy allocator over a caller-provided frame-descriptor array.
//!
//! Frames are relative indices `[0, nframes)`. The per-frame [`FrameDesc`] array
//! (`&mut [FrameDesc]`) holds the intrusive doubly-linked free-list nodes and each
//! free block's order; the kernel bootstraps it from a reserved physical region,
//! host tests back it with a `Vec`. All frames start reserved (holes / MMIO /
//! non-usable stay reserved forever); [`Buddy::add_frames`] releases usable runs,
//! decomposed into maximally-aligned power-of-two blocks. Buddies coalesce on free.

use crate::frame::{Pfn, NIL};

/// Largest block order: `2^MAX_ORDER` frames (order 18 → 1 GiB). The free-list head
/// array is just `MAX_ORDER + 1` entries.
pub const MAX_ORDER: u8 = 18;

/// Per-frame descriptor. For the frame that *heads* a free block it stores the
/// intrusive free-list links and the block order; reserved/allocated frames carry
/// `free == false` (allocated heads keep `order` so [`Buddy::free`] knows the size).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameDesc {
    next: Pfn,
    prev: Pfn,
    order: u8,
    free: bool,
}

impl FrameDesc {
    /// A reserved (unusable or not-yet-added) frame.
    pub const RESERVED: Self = Self {
        next: NIL,
        prev: NIL,
        order: 0,
        free: false,
    };
}

impl Default for FrameDesc {
    fn default() -> Self {
        Self::RESERVED
    }
}

/// Buddy allocator state; borrows the frame-descriptor array for its lifetime.
pub struct Buddy<'a> {
    desc: &'a mut [FrameDesc],
    free: [Pfn; MAX_ORDER as usize + 1],
    nframes: Pfn,
    free_frames: Pfn,
}

impl<'a> Buddy<'a> {
    /// Create an allocator over `desc` (one entry per frame), all frames reserved.
    /// Release usable frames with [`add_frames`](Self::add_frames).
    pub fn new(desc: &'a mut [FrameDesc]) -> Self {
        let nframes = desc.len() as Pfn;
        for d in desc.iter_mut() {
            *d = FrameDesc::RESERVED;
        }
        Self {
            desc,
            free: [NIL; MAX_ORDER as usize + 1],
            nframes,
            free_frames: 0,
        }
    }

    /// Frames currently free (for accounting / leak checks).
    #[must_use]
    pub fn free_frames(&self) -> Pfn {
        self.free_frames
    }

    /// Total frames the allocator spans.
    #[must_use]
    pub fn total_frames(&self) -> Pfn {
        self.nframes
    }

    /// Release the run `[start, start + count)` into the free pool. Frames must be in
    /// range and currently reserved — the caller (the kernel memmap parse) guarantees
    /// this after hostile-input validation.
    pub fn add_frames(&mut self, start: Pfn, count: Pfn) {
        debug_assert!(start.checked_add(count).is_some_and(|e| e <= self.nframes));
        if count == 0 {
            return;
        }
        let end = start + count;
        let mut base = start;
        while base < end {
            // Largest order whose block is both aligned at `base` and fits in the run.
            let align_order = if base == 0 {
                u32::from(MAX_ORDER)
            } else {
                base.trailing_zeros()
            };
            let size_order = (Pfn::BITS - 1) - (end - base).leading_zeros(); // floor(log2)
            let order = align_order.min(size_order).min(u32::from(MAX_ORDER)) as u8;
            self.free_frames += 1u32 << order;
            self.insert(base, order);
            base += 1u32 << order;
        }
    }

    /// Allocate a block of `2^order` contiguous frames; returns its base frame number,
    /// or `None` if no block that large is available.
    pub fn alloc(&mut self, order: u8) -> Option<Pfn> {
        if order > MAX_ORDER {
            return None;
        }
        // Smallest available order >= requested.
        let mut o = order;
        while o <= MAX_ORDER && self.free[o as usize] == NIL {
            o += 1;
        }
        if o > MAX_ORDER {
            return None;
        }
        let base = self.free[o as usize];
        self.unlink(base, o);
        // Split down to the requested order, releasing the upper buddy at each step.
        while o > order {
            o -= 1;
            let upper = base + (1u32 << o);
            self.insert(upper, o);
        }
        self.desc[base as usize].order = order;
        self.desc[base as usize].free = false;
        self.free_frames -= 1u32 << order;
        Some(base)
    }

    /// Free a block previously returned by [`alloc`](Self::alloc); the order recorded
    /// at allocation time is used. Coalesces with the buddy where possible.
    pub fn free(&mut self, base: Pfn) {
        debug_assert!(
            base < self.nframes && !self.desc[base as usize].free,
            "buddy: double-free or out-of-range frame"
        );
        let order = self.desc[base as usize].order;
        self.free_frames += 1u32 << order;
        self.coalesce(base, order);
    }

    // ---- internals ----

    fn coalesce(&mut self, mut base: Pfn, mut order: u8) {
        while order < MAX_ORDER {
            let buddy = base ^ (1u32 << order);
            if buddy >= self.nframes {
                break;
            }
            let bd = self.desc[buddy as usize];
            if !bd.free || bd.order != order {
                break;
            }
            self.unlink(buddy, order);
            if buddy < base {
                base = buddy;
            }
            order += 1;
        }
        self.insert(base, order);
    }

    fn insert(&mut self, base: Pfn, order: u8) {
        let head = self.free[order as usize];
        self.desc[base as usize] = FrameDesc {
            next: head,
            prev: NIL,
            order,
            free: true,
        };
        if head != NIL {
            self.desc[head as usize].prev = base;
        }
        self.free[order as usize] = base;
    }

    fn unlink(&mut self, base: Pfn, order: u8) {
        let FrameDesc { next, prev, .. } = self.desc[base as usize];
        if prev != NIL {
            self.desc[prev as usize].next = next;
        } else {
            self.free[order as usize] = next;
        }
        if next != NIL {
            self.desc[next as usize].prev = prev;
        }
        self.desc[base as usize].free = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;
    use std::vec;
    use std::vec::Vec;

    fn descs(n: usize) -> Vec<FrameDesc> {
        vec![FrameDesc::RESERVED; n]
    }

    #[test]
    fn roundtrip_single_frame() {
        let mut d = descs(1024);
        let mut b = Buddy::new(&mut d);
        b.add_frames(0, 1024);
        assert_eq!(b.free_frames(), 1024);
        let a = b.alloc(0).unwrap();
        assert_eq!(b.free_frames(), 1023);
        b.free(a);
        assert_eq!(b.free_frames(), 1024);
    }

    #[test]
    fn full_free_coalesces_to_one_max_block() {
        // 1024 frames == 2^10: after allocating everything as order-0 and freeing it
        // all, it must coalesce back into a single order-10 block.
        let mut d = descs(1024);
        let mut b = Buddy::new(&mut d);
        b.add_frames(0, 1024);
        let mut allocated = Vec::new();
        for _ in 0..1024 {
            allocated.push(b.alloc(0).unwrap());
        }
        assert_eq!(b.free_frames(), 0);
        assert!(b.alloc(0).is_none(), "should be out of memory");
        for a in allocated {
            b.free(a);
        }
        assert_eq!(b.free_frames(), 1024);
        // Fully coalesced: one order-10 block spanning all frames.
        assert_eq!(b.alloc(10), Some(0));
        b.free(0);
    }

    #[test]
    fn split_produces_distinct_aligned_blocks() {
        let mut d = descs(1024);
        let mut b = Buddy::new(&mut d);
        b.add_frames(0, 1024);
        let a = b.alloc(2).unwrap(); // 4 frames
        let c = b.alloc(2).unwrap();
        assert_ne!(a, c);
        assert_eq!(a % 4, 0);
        assert_eq!(c % 4, 0);
        assert_eq!(b.free_frames(), 1024 - 8);
        b.free(a);
        b.free(c);
        assert_eq!(b.free_frames(), 1024);
    }

    #[test]
    fn non_power_of_two_region() {
        // 1000 frames is not a power of two; add_frames must decompose it and account
        // exactly, with no frame lost.
        let mut d = descs(1000);
        let mut b = Buddy::new(&mut d);
        b.add_frames(0, 1000);
        assert_eq!(b.free_frames(), 1000);
        let mut got = Vec::new();
        while let Some(f) = b.alloc(0) {
            got.push(f);
        }
        assert_eq!(got.len(), 1000, "every frame must be allocatable");
        for f in got {
            b.free(f);
        }
        assert_eq!(b.free_frames(), 1000);
    }

    #[test]
    fn reserved_holes_are_never_handed_out() {
        // Add two disjoint runs with a reserved hole [16, 32); the hole must never
        // be allocated.
        let mut d = descs(64);
        let mut b = Buddy::new(&mut d);
        b.add_frames(0, 16);
        b.add_frames(32, 32);
        assert_eq!(b.free_frames(), 48);
        let mut got = Vec::new();
        while let Some(f) = b.alloc(0) {
            assert!(
                !(16..32).contains(&f),
                "allocated a reserved hole frame {f}"
            );
            got.push(f);
        }
        assert_eq!(got.len(), 48);
    }

    #[test]
    fn soak_no_leak() {
        // Deterministic LCG-driven alloc/free churn; free_frames must stay consistent
        // and return to the original total at the end.
        let mut d = descs(4096);
        let mut b = Buddy::new(&mut d);
        b.add_frames(0, 4096);
        let total = b.free_frames();
        let mut live: Vec<(Pfn, u8)> = Vec::new();
        let mut rng: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng >> 33) as u32
        };
        for _ in 0..50_000 {
            let order = (next() % 4) as u8; // 0..=3
            if next() % 2 == 0 || live.is_empty() {
                if let Some(f) = b.alloc(order) {
                    live.push((f, order));
                }
            } else {
                let i = (next() as usize) % live.len();
                let (f, _) = live.swap_remove(i);
                b.free(f);
            }
            // Free frames never exceed the total and never underflow.
            assert!(b.free_frames() <= total);
        }
        for (f, _) in live {
            b.free(f);
        }
        assert_eq!(b.free_frames(), total, "leak: free frames not restored");
    }
}
