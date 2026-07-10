//! The kernel's internal heap: a `#[global_allocator]` over a region carved from the P1
//! buddy allocator (ADR-0003 P3a decision).
//!
//! The async executor (P3) needs `alloc` — `Box<dyn Future>` task control blocks, the run
//! queue, wakers. This heap is **kernel-internal and cap-accounted**: its backing frames come
//! from the buddy that owns all USABLE RAM (a kernel-owned Untyped in capability terms), not
//! an anonymous ambient pool — so SPEC-CAP §8 ("no ambient heap a *component* draws from")
//! still holds; the kernel TCB legitimately allocates from memory it owns. Components (P7)
//! never touch this; they get memory via RETYPE from their own Untyped.

use linked_list_allocator::LockedHeap;
use mem::frame::PAGE_SIZE;

use crate::memory;

/// Heap span: 2 MiB (2^9 frames) — ample for the executor's task blocks, wakers, and queue,
/// and a single buddy block (`MAX_ORDER` is 18). Fixed for the kernel's lifetime; never freed.
const HEAP_ORDER: u8 = 9;

#[global_allocator]
static ALLOCATOR: LockedHeap = LockedHeap::empty();

/// Carve the heap from the buddy and hand it to the global allocator. Call **once**, after
/// [`memory::init`] (the buddy must exist) and before the first allocation.
pub fn init() {
    let phys = memory::alloc_frames(HEAP_ORDER)
        .unwrap_or_else(|| fatal("no contiguous frames for the kernel heap"));
    let size = (1usize << HEAP_ORDER) * PAGE_SIZE as usize;
    let virt = memory::phys_to_virt(phys);
    // SAFETY: `[virt, virt+size)` is exactly `size` bytes of frames we just allocated from the
    // buddy and exclusively own, HHDM-mapped and writable, and never freed (the heap lives for
    // the whole kernel lifetime). `init` runs once, before any allocation touches ALLOCATOR.
    unsafe {
        ALLOCATOR.lock().init(virt as *mut u8, size);
    }
    kprintln!(
        "[praesidium] heap: {} KiB kernel heap @ {virt:#x} (phys {phys:#x})",
        size / 1024
    );
}

fn fatal(msg: &str) -> ! {
    kprintln!("[praesidium] FATAL: heap: {msg}");
    crate::arch::halt();
}
