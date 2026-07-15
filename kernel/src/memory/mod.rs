//! Kernel memory subsystem (P1): bootstrap the physical frame allocator from the
//! Warden memory map, expose the HHDM for physical access, and drive the `mem`
//! crate's buddy/slab/`Untyped`-retype logic over live frames.
//!
//! All handoff-derived values are HOSTILE (GC-03): the region array is bounds- and
//! alignment-checked, frame arithmetic is overflow-guarded, and an implausible span,
//! an unusable map, or a too-small descriptor region fails the boot **closed**
//! (`FATAL` + halt) rather than corrupting an index.
//!
//! (Own page tables + W^X — AC1.3/AC1.4 — build on this and land alongside.)

use core::mem::{align_of, size_of};
use core::sync::atomic::{AtomicU64, Ordering};

use mem::buddy::{Buddy, FrameDesc};
use mem::frame::{pfn_to_phys, phys_to_pfn, Pfn, PAGE_SHIFT, PAGE_SIZE};
use mem::retype::Untyped;
use mem::slab::SlabCache;

use crate::boot::handoff::{MemRegion, MemoryKind, WardenBootInfo};
use crate::sync::SpinLock;

// Linker-defined kernel image + per-section boundaries (page-aligned), used to map
// each section with its own W^X protection.
extern "C" {
    static __kernel_start: u8;
    static __kernel_end: u8;
    static __text_start: u8;
    static __text_end: u8;
    static __rodata_start: u8;
    static __rodata_end: u8;
    static __data_start: u8;
    static __data_end: u8;
}

/// Upper bound on regions scanned from the (hostile) memory map (GC-03).
const MAX_REGIONS: usize = 1024;
/// Upper bound on distinct USABLE frame ranges we accept; more → fail closed.
const MAX_USABLE: usize = 512;

/// Low physical memory to identity- and HHDM-map (covers RAM + low MMIO on the QEMU
/// targets; extend when a target has usable RAM above 4 GiB).
const IDENTITY_BYTES: u64 = 4 << 30;

/// HHDM offset from Warden, set once at [`init`]; [`phys_to_virt`] adds it.
static HHDM_OFFSET: AtomicU64 = AtomicU64::new(0);

/// Physical address → kernel-virtual address via the higher-half direct map.
#[inline]
#[must_use]
pub fn phys_to_virt(pa: u64) -> u64 {
    pa + HHDM_OFFSET.load(Ordering::Relaxed)
}

/// The global physical frame allocator; `None` until [`init`].
static FRAMES: SpinLock<Option<FrameAllocator>> = SpinLock::new(None);

struct FrameAllocator {
    buddy: Buddy<'static>,
    /// Physical frame corresponding to buddy relative index 0.
    base_pfn: Pfn,
}

impl FrameAllocator {
    fn alloc(&mut self, order: u8) -> Option<u64> {
        let rel = self.buddy.alloc(order)?;
        Some(pfn_to_phys(self.base_pfn + rel))
    }

    fn free(&mut self, pa: u64) {
        let rel = phys_to_pfn(pa) - self.base_pfn;
        self.buddy.free(rel);
    }
}

/// Allocate `2^order` contiguous frames; returns the physical base address.
pub fn alloc_frames(order: u8) -> Option<u64> {
    FRAMES.lock().as_mut()?.alloc(order)
}

/// Allocate a single zeroed frame (page tables, fresh objects — CAP-MEM-2 hygiene).
pub fn alloc_zeroed_frame() -> Option<u64> {
    let pa = alloc_frames(0)?;
    zero_frame(pa);
    Some(pa)
}

/// Return frames previously handed out by [`alloc_frames`].
pub fn free_frames(pa: u64) {
    if let Some(fa) = FRAMES.lock().as_mut() {
        fa.free(pa);
    }
}

/// The number of physical frames currently free in the buddy (0 before [`init`]). Used by the
/// bridge-substrate reaper to prove frame **conservation** — a spawn's buddy footprint equals what
/// the reaper returns on process death (no leak, no over-free).
#[must_use]
pub fn free_frame_count() -> u64 {
    FRAMES
        .lock()
        .as_ref()
        .map_or(0, |fa| u64::from(fa.buddy.free_frames()))
}

/// Zero exactly one frame through the HHDM.
pub fn zero_frame(pa: u64) {
    zero_range(pa, PAGE_SIZE);
}

/// Bootstrap the frame allocator from the Warden memory map, then run the P1 demo.
pub fn init(bi: &WardenBootInfo) {
    HHDM_OFFSET.store(bi.hhdm_offset, Ordering::Relaxed);
    let regions = region_slice(bi);
    if regions.is_empty() {
        fatal("empty or unreadable memory map");
    }

    // Collect validated USABLE frame ranges, clamped to the HHDM-mapped window (frames
    // above it have no HHDM virtual address, so they must never enter the allocator —
    // the buddy accesses every frame via the HHDM).
    let hhdm_max = IDENTITY_BYTES >> PAGE_SHIFT;
    let mut ranges: [(Pfn, Pfn); MAX_USABLE] = [(0, 0); MAX_USABLE];
    let mut n = 0usize;
    for r in regions {
        if r.kind != MemoryKind::USABLE {
            continue;
        }
        let Some((start, end)) = usable_frames(r) else {
            continue;
        };
        let end = end.min(hhdm_max);
        if end <= start {
            continue; // entirely above the mapped window
        }
        if n >= MAX_USABLE {
            fatal("too many usable regions in memory map");
        }
        ranges[n] = (start as Pfn, end as Pfn);
        n += 1;
    }
    if n == 0 {
        fatal("no usable RAM within the HHDM window");
    }
    let ranges = &mut ranges[..n];
    ranges.sort_unstable_by_key(|&(start, _)| start);
    // Reject overlapping/duplicate USABLE regions: a hostile map must not make the buddy
    // double-insert a frame and hand the same physical page to two owners (GC-03).
    for pair in ranges.windows(2) {
        if pair[1].0 < pair[0].1 {
            fatal("overlapping USABLE regions in memory map");
        }
    }

    let base_pfn = ranges[0].0;
    let span = ranges[n - 1].1 - base_pfn;
    let total_pages: u64 = ranges.iter().map(|&(s, e)| u64::from(e - s)).sum();

    // Descriptor array: `span` entries, placed at the front of the first range big
    // enough to hold it (ranges are disjoint, so its frames belong to no other range).
    let desc_bytes = u64::from(span).saturating_mul(size_of::<FrameDesc>() as u64);
    let desc_frames = desc_bytes.div_ceil(PAGE_SIZE) as Pfn;
    let desc_start = ranges
        .iter()
        .find(|&&(s, e)| e - s >= desc_frames)
        .map(|&(s, _)| s)
        .unwrap_or_else(|| fatal("no usable region fits the frame-descriptor array"));
    let desc_phys = pfn_to_phys(desc_start);

    // Zero the descriptor region so forming `&mut [FrameDesc]` over it is sound (an
    // all-zero FrameDesc is valid), then build the buddy over it.
    zero_range(desc_phys, desc_bytes);
    // SAFETY: `[desc_phys, +desc_bytes)` is reserved usable RAM we exclusively own,
    // HHDM-mapped, exactly `span * size_of::<FrameDesc>()` bytes, page-aligned (a frame
    // base) hence FrameDesc-aligned, and just zeroed. It backs the allocator for the
    // whole kernel lifetime ('static).
    let desc: &'static mut [FrameDesc] = unsafe {
        core::slice::from_raw_parts_mut(phys_to_virt(desc_phys) as *mut FrameDesc, span as usize)
    };
    let mut buddy = Buddy::new(desc);

    // Release usable frames (disjoint), carving the descriptor array off its host range.
    for &(start, end) in ranges.iter() {
        if start == desc_start {
            let after = start + desc_frames;
            if after < end {
                buddy.add_frames(after - base_pfn, end - after);
            }
        } else {
            buddy.add_frames(start - base_pfn, end - start);
        }
    }

    let free = buddy.free_frames();
    *FRAMES.lock() = Some(FrameAllocator { buddy, base_pfn });

    kprintln!(
        "[praesidium] mem: base={:#x} span={} frames, usable={} MiB, desc={} frames @ {:#x}",
        pfn_to_phys(base_pfn),
        span,
        mib(total_pages),
        desc_frames,
        desc_phys
    );
    kprintln!(
        "[praesidium] mem: buddy managing {} free frames ({} MiB)",
        free,
        mib(u64::from(free))
    );

    // Build and switch to Praesidium's own page tables with W^X (AC1.3/AC1.4), then
    // exercise the allocator on the new tables.
    setup_kernel_paging(bi);
    demo();
    verify_zero_on_retype();
    kprintln!("[praesidium] PRAESIDIUM-P1-OK");
}

/// Build and switch to Praesidium's own page tables (AC1.4), enforcing W^X on the
/// kernel image (AC1.3). The switch is safe because the new map mirrors Warden's
/// layout — the identity map keeps the boot stack valid and the kernel `.text` stays
/// mapped executable — so the CPU keeps running across the CR3/TTBR swap.
fn setup_kernel_paging(bi: &WardenBootInfo) {
    // addr_of! computes linker-symbol addresses without reading the symbols; these are
    // region boundaries, never dereferenced.
    let (kstart, kend, tstart, tend, rostart, roend, dstart, dend) = (
        core::ptr::addr_of!(__kernel_start) as u64,
        core::ptr::addr_of!(__kernel_end) as u64,
        core::ptr::addr_of!(__text_start) as u64,
        core::ptr::addr_of!(__text_end) as u64,
        core::ptr::addr_of!(__rodata_start) as u64,
        core::ptr::addr_of!(__rodata_end) as u64,
        core::ptr::addr_of!(__data_start) as u64,
        core::ptr::addr_of!(__data_end) as u64,
    );
    let kphys = crate::arch::translate(kstart)
        .unwrap_or_else(|| fatal("cannot resolve kernel physical base"));
    let km = crate::arch::KernelMap {
        hhdm_offset: bi.hhdm_offset,
        identity_bytes: IDENTITY_BYTES,
        kernel_vbase: kstart,
        kernel_vend: kend.next_multiple_of(PAGE_SIZE),
        kernel_phys: kphys,
        text: (tstart, tend),
        rodata: (rostart, roend),
        data: (dstart, dend),
    };
    // map_kernel maps the image with a single leaf table, which requires it to fit in
    // one 2 MiB window (Warden's loader enforces this); assert it and fail closed.
    if !km.kernel_vbase.is_multiple_of(0x20_0000) || km.kernel_vend - km.kernel_vbase > 0x20_0000 {
        fatal("kernel image exceeds 2 MiB or is not 2 MiB-aligned");
    }
    // Establish the arch control state W^X depends on before activating the NX-bearing /
    // read-only tables (x86: CR0.WP + EFER.NXE; aarch64: no-op — EL1 enforces AP/XN).
    crate::arch::enable_wx();
    let space = crate::arch::build_address_space(&km);
    // SAFETY: `space` maps the current PC (kernel .text, executable), the boot stack
    // (kernel .bss), and the HHDM, so execution continues across the switch.
    unsafe { crate::arch::activate_address_space(space) };
    kprintln!(
        "[praesidium] mem: own page tables active, root={:#x}, kernel_phys={kphys:#x} (AC1.4)",
        space.primary
    );
    verify_wx(&km);
}

/// Confirm W^X is actually in force in the active tables and that the mapping API
/// structurally refuses W+X (CAP-MEM-1).
fn verify_wx(km: &crate::arch::KernelMap) {
    if crate::arch::Prot::checked(true, true, true).is_some() {
        fatal("W+X protection was not refused");
    }
    let (text_w, text_x) = crate::arch::page_prot(km.text.0)
        .unwrap_or_else(|| fatal("kernel .text unmapped after switch"));
    if text_w || !text_x {
        fatal("kernel .text is not R-X after switch");
    }
    // .rodata (if present) must be read-only, non-executable.
    if km.rodata.0 < km.rodata.1 {
        let (ro_w, ro_x) = crate::arch::page_prot(km.rodata.0)
            .unwrap_or_else(|| fatal("kernel .rodata unmapped after switch"));
        if ro_w || ro_x {
            fatal("kernel .rodata is not R-- after switch");
        }
    }
    let (data_w, data_x) = crate::arch::page_prot(km.data.0)
        .unwrap_or_else(|| fatal("kernel .data unmapped after switch"));
    if !data_w || data_x {
        fatal("kernel .data is not RW-NX after switch");
    }
    kprintln!("[praesidium] mem: W^X verified — .text R-X, .data RW-NX, W+X refused (AC1.3)");
}

/// Verify no stale data survives a retype/realloc (CAP-MEM-2): poison a frame, free
/// it, re-allocate zeroed, and confirm it reads back zero.
fn verify_zero_on_retype() {
    let f = alloc_frames(0).unwrap_or_else(|| fatal("zero-check: alloc"));
    let fp = phys_to_virt(f) as *mut u64;
    // SAFETY: `f` is our frame, HHDM-mapped and writable.
    unsafe { fp.write_volatile(0xA5A5_A5A5_A5A5_A5A5) };
    free_frames(f);
    let g = alloc_zeroed_frame().unwrap_or_else(|| fatal("zero-check: realloc"));
    let gp = phys_to_virt(g) as *const u64;
    // SAFETY: `g` is our freshly-zeroed frame, HHDM-mapped.
    let val = unsafe { gp.read_volatile() };
    if val != 0 {
        fatal("zero-on-retype failed: stale data visible");
    }
    kprintln!("[praesidium] mem: zero-on-retype verified — reads 0, not stale (CAP-MEM-2)");
    free_frames(g);
}

/// Exercise the allocator at boot: prove alloc/free (AC1.1), slab + HHDM read/write
/// (AC1.2), and the `Untyped` retype seam (MEM-T5). Any broken invariant fails the
/// boot closed via `fatal`.
fn demo() {
    // AC1.1 — allocate two distinct, zeroed, writable frames.
    let a = alloc_zeroed_frame().unwrap_or_else(|| fatal("frame alloc failed"));
    let b = alloc_zeroed_frame().unwrap_or_else(|| fatal("frame alloc failed"));
    let ap = phys_to_virt(a) as *mut u64;
    // SAFETY: `a` is our freshly-allocated, zeroed, HHDM-mapped frame; sole access.
    let ok = unsafe {
        let zeroed = ap.read_volatile() == 0;
        ap.write_volatile(0xDEAD_BEEF_CAFE_F00D);
        zeroed && ap.read_volatile() == 0xDEAD_BEEF_CAFE_F00D
    };
    if !ok || a == b {
        fatal("frame alloc not distinct/zeroed/writable");
    }
    kprintln!("[praesidium] mem: frames {a:#x},{b:#x} distinct + zeroed + writable (AC1.1)");
    free_frames(a);
    free_frames(b);

    // AC1.2 — slab of fixed objects over one frame, addressed through the HHDM.
    let sf = alloc_zeroed_frame().unwrap_or_else(|| fatal("slab frame failed"));
    let sv = phys_to_virt(sf) as *mut u8;
    // SAFETY: `sf` is our zeroed frame; expose it as a page-sized byte region.
    let region = unsafe { core::slice::from_raw_parts_mut(sv, PAGE_SIZE as usize) };
    let mut cache = SlabCache::new(64, 8, PAGE_SIZE as usize);
    cache.init(region);
    let o1 = cache.alloc(region).unwrap_or_else(|| fatal("slab alloc"));
    let o2 = cache.alloc(region).unwrap_or_else(|| fatal("slab alloc"));
    if o1 == o2 {
        fatal("slab returned a slot twice");
    }
    kprintln!(
        "[praesidium] mem: slab {}x64B objs, allocs off {o1},{o2}, {} free (AC1.2)",
        cache.capacity(),
        cache.free_count()
    );
    free_frames(sf);

    // MEM-T5 — Untyped retype accounting seam.
    let uf = alloc_frames(4).unwrap_or_else(|| fatal("untyped block alloc")); // 16 frames
    let mut u = Untyped::new(phys_to_pfn(uf), 16);
    let r = u.retype(1, 4).unwrap_or_else(|| fatal("retype"));
    kprintln!(
        "[praesidium] mem: untyped {} frames, retype(4) -> {:#x}, {} remaining (MEM-T5)",
        u.frames(),
        pfn_to_phys(r),
        u.remaining()
    );
    free_frames(uf);
}

// ---- helpers ----

fn fatal(msg: &str) -> ! {
    kprintln!("[praesidium] FATAL: mem: {msg}");
    crate::arch::halt();
}

fn mib(pages: u64) -> u64 {
    pages * PAGE_SIZE / (1024 * 1024)
}

/// Bounded, alignment-checked view of the region array via the HHDM.
fn region_slice(bi: &WardenBootInfo) -> &'static [MemRegion] {
    let count = bi.memmap.count.min(MAX_REGIONS as u64) as usize;
    if bi.memmap.regions == 0 || count == 0 {
        return &[];
    }
    let virt = phys_to_virt(bi.memmap.regions);
    if !virt.is_multiple_of(align_of::<MemRegion>() as u64) {
        return &[];
    }
    // SAFETY: Warden guarantees `regions[0..memmap.count]` is a MemRegion array at
    // this physical base, HHDM-mapped; `count` is bounded and `virt`'s alignment is
    // checked. Read-only for the boot lifetime.
    unsafe { core::slice::from_raw_parts(virt as *const MemRegion, count) }
}

/// Frame range `[start, end)` of a region, or `None` if malformed / out of range.
fn usable_frames(r: &MemRegion) -> Option<(u64, u64)> {
    if !r.base.is_multiple_of(PAGE_SIZE) {
        return None; // must be page-aligned
    }
    let start = r.base >> PAGE_SHIFT;
    let end = start.checked_add(r.pages)?; // overflow-safe
    if end <= start {
        return None;
    }
    Some((start, end))
}

/// Zero `[phys, phys + bytes)` through the HHDM.
fn zero_range(phys: u64, bytes: u64) {
    let virt = phys_to_virt(phys) as *mut u8;
    // SAFETY: `[phys, +bytes)` is reserved usable RAM we own, HHDM-mapped and writable.
    unsafe {
        core::ptr::write_bytes(virt, 0, bytes as usize);
    }
}
