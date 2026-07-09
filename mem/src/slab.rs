//! Fixed-size object slab cache over a caller-provided byte region.
//!
//! A [`SlabCache`] carves a contiguous region into equal object slots and threads a
//! singly-linked free list **through the free slots themselves** (each free slot's
//! first `usize` holds the next free slot's byte offset). The cache struct holds only
//! the head + counts; the links live in the backing memory, which the caller passes
//! in on each operation. In the kernel the region is HHDM-mapped frames from the
//! buddy; host tests pass a `Vec<u8>`. Allocation returns a byte **offset** into the
//! region — the caller adds it to the region base to get the object address.

use core::mem::size_of;

const NIL_OFF: usize = usize::MAX;

/// A cache of fixed-size, fixed-alignment object slots.
pub struct SlabCache {
    slot_size: usize,
    capacity: usize,
    free_head: usize,
    free_count: usize,
}

impl SlabCache {
    /// Build a cache for objects of `obj_size`/`align` over a `region_len`-byte region.
    /// The slot size is rounded up to hold the free link and to satisfy `align`.
    /// `align` must be a power of two.
    #[must_use]
    pub fn new(obj_size: usize, align: usize, region_len: usize) -> Self {
        debug_assert!(align.is_power_of_two());
        let align = align.max(size_of::<usize>());
        let slot_size = round_up(obj_size.max(size_of::<usize>()), align);
        let capacity = region_len.checked_div(slot_size).unwrap_or(0);
        Self {
            slot_size,
            capacity,
            free_head: NIL_OFF,
            free_count: 0,
        }
    }

    /// Byte size of one slot (object size rounded up for link + alignment).
    #[must_use]
    pub fn slot_size(&self) -> usize {
        self.slot_size
    }

    /// Number of object slots the region holds.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Objects currently free.
    #[must_use]
    pub fn free_count(&self) -> usize {
        self.free_count
    }

    /// Thread the free list through every slot. Must be called once before [`alloc`]
    /// (Self::alloc); `mem` must be at least `capacity * slot_size` bytes.
    pub fn init(&mut self, mem: &mut [u8]) {
        debug_assert!(mem.len() >= self.capacity * self.slot_size);
        self.free_head = NIL_OFF;
        self.free_count = 0;
        // Link in reverse so the free list runs low-to-high (head at slot 0).
        for i in (0..self.capacity).rev() {
            let off = i * self.slot_size;
            write_link(mem, off, self.free_head);
            self.free_head = off;
            self.free_count += 1;
        }
    }

    /// Pop a free slot, returning its byte offset, or `None` when exhausted.
    pub fn alloc(&mut self, mem: &[u8]) -> Option<usize> {
        if self.free_head == NIL_OFF {
            return None;
        }
        let off = self.free_head;
        self.free_head = read_link(mem, off);
        self.free_count -= 1;
        Some(off)
    }

    /// Return a slot (by the offset [`alloc`](Self::alloc) gave) to the free list.
    pub fn free(&mut self, mem: &mut [u8], off: usize) {
        debug_assert!(off.is_multiple_of(self.slot_size) && off / self.slot_size < self.capacity);
        write_link(mem, off, self.free_head);
        self.free_head = off;
        self.free_count += 1;
    }
}

fn round_up(x: usize, align: usize) -> usize {
    (x + align - 1) & !(align - 1)
}

fn write_link(mem: &mut [u8], off: usize, val: usize) {
    mem[off..off + size_of::<usize>()].copy_from_slice(&val.to_ne_bytes());
}

fn read_link(mem: &[u8], off: usize) -> usize {
    let mut b = [0u8; size_of::<usize>()];
    b.copy_from_slice(&mem[off..off + size_of::<usize>()]);
    usize::from_ne_bytes(b)
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;
    use std::vec;
    use std::vec::Vec;

    #[test]
    fn carves_and_serves_all_slots() {
        // 64-byte objects over a 4 KiB region -> 64 slots.
        let mut cache = SlabCache::new(64, 8, 4096);
        assert_eq!(cache.slot_size(), 64);
        assert_eq!(cache.capacity(), 64);
        let mut mem = vec![0u8; 4096];
        cache.init(&mut mem);
        assert_eq!(cache.free_count(), 64);

        let mut offs = Vec::new();
        while let Some(o) = cache.alloc(&mem) {
            assert_eq!(o % 64, 0, "slot must be slot-size aligned");
            assert!(!offs.contains(&o), "slot handed out twice: {o}");
            offs.push(o);
        }
        assert_eq!(offs.len(), 64);
        assert_eq!(cache.free_count(), 0);
        assert!(cache.alloc(&mem).is_none());
    }

    #[test]
    fn free_then_realloc() {
        let mut cache = SlabCache::new(32, 8, 1024);
        let mut mem = vec![0u8; 1024];
        cache.init(&mut mem);
        let a = cache.alloc(&mem).unwrap();
        let b = cache.alloc(&mem).unwrap();
        assert_ne!(a, b);
        cache.free(&mut mem, a);
        // Freed slot is reused (LIFO).
        let c = cache.alloc(&mem).unwrap();
        assert_eq!(c, a);
        assert_eq!(cache.free_count(), cache.capacity() - 2);
    }

    #[test]
    fn small_object_rounds_up_to_link_size() {
        // A 1-byte object still needs room for the free link (>= usize).
        let cache = SlabCache::new(1, 1, 256);
        assert_eq!(cache.slot_size(), size_of::<usize>());
        assert_eq!(cache.capacity(), 256 / size_of::<usize>());
    }

    #[test]
    fn alignment_is_respected() {
        let cache = SlabCache::new(48, 64, 4096);
        assert_eq!(cache.slot_size() % 64, 0);
        assert_eq!(cache.slot_size(), 64);
    }
}
