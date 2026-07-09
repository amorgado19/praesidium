//! Physical frame primitives shared across the allocator.

/// Page/frame size (4 KiB — matches Warden's `PAGE_SIZE` and the handoff ABI).
pub const PAGE_SIZE: u64 = 4096;
/// `log2(PAGE_SIZE)` — a physical address >> `PAGE_SHIFT` is its frame number.
pub const PAGE_SHIFT: u32 = 12;

/// Page frame number: a physical address divided by [`PAGE_SIZE`]. `u32` addresses
/// up to 2^32 frames = 16 TiB of physical RAM, far beyond any target Praesidium runs.
pub type Pfn = u32;

/// Sentinel meaning "no frame" (used as a null link in intrusive free lists).
pub(crate) const NIL: Pfn = Pfn::MAX;

/// Convert a frame number to a physical address.
#[inline]
#[must_use]
pub const fn pfn_to_phys(pfn: Pfn) -> u64 {
    (pfn as u64) << PAGE_SHIFT
}

/// Convert a physical address (assumed page-aligned) to a frame number.
#[inline]
#[must_use]
pub const fn phys_to_pfn(phys: u64) -> Pfn {
    (phys >> PAGE_SHIFT) as Pfn
}
