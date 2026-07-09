//! WardenBootInfo intake + memory-map/framebuffer dump (P0 / T0.1–T0.3).
//!
//! All handoff input is HOSTILE (GC-03): the magic + ABI version are validated
//! before any other field is trusted, the region array pointer and count are
//! bounds-checked, and arithmetic over region sizes saturates rather than
//! trapping — a corrupt `count` must never walk the kernel off a cliff or panic
//! the smoke run.

pub mod handoff;

use handoff::{MemRegion, MemoryKind, WardenBootInfo};

/// Cap on memmap regions we will walk, defending against a corrupted `count`
/// (GC-03). Warden's post-ExitBootServices map is a few hundred entries at most.
const MAX_REGIONS: u64 = 1024;
/// How many regions to print individually before summarizing (keep serial sane).
const MAX_DUMP_ROWS: u64 = 32;

/// Validate the warden-rich handoff and dump what Warden gave us. On a contract
/// violation this logs `FATAL` and halts — it never proceeds on bad input.
pub fn validate_and_dump(bootinfo: *const WardenBootInfo) -> &'static WardenBootInfo {
    if bootinfo.is_null() {
        kprintln!("[praesidium] FATAL: null bootinfo pointer");
        crate::arch::halt();
    }
    // Forming a `&WardenBootInfo` from a misaligned pointer is instantaneous UB in
    // Rust (before any load even runs), so validate alignment against the hostile
    // pointer *before* the deref — not just its contents afterward (GC-03).
    if (bootinfo as usize) & (core::mem::align_of::<WardenBootInfo>() - 1) != 0 {
        kprintln!("[praesidium] FATAL: misaligned bootinfo pointer");
        crate::arch::halt();
    }

    // SAFETY: `bootinfo` is non-null and 8-aligned (both checked above) and, per the
    // warden-rich contract (REF-001), points at a WardenBootInfo mapped at this
    // HHDM-virtual address. We only read it, and treat every field *value* as
    // hostile below.
    let bi: &'static WardenBootInfo = unsafe { &*bootinfo };

    kprintln!(
        "[praesidium] magic={:#018x} abi_version={}",
        bi.magic,
        bi.abi_version
    );
    if !bi.is_valid() {
        kprintln!(
            "[praesidium] FATAL: bad handoff (want magic={:#018x} abi_version={})",
            handoff::WARDEN_MAGIC,
            handoff::WARDEN_ABI_VERSION
        );
        crate::arch::halt();
    }
    kprintln!("[praesidium] CONTRACT OK: magic + abi_version valid");
    kprintln!(
        "[praesidium] hhdm_offset={:#018x} rsdp={:#018x}",
        bi.hhdm_offset,
        bi.rsdp
    );

    dump_framebuffer(bi);
    dump_memmap(bi);
    bi
}

fn dump_framebuffer(bi: &WardenBootInfo) {
    let fb = &bi.framebuffer;
    if fb.present != 0 {
        kprintln!(
            "[praesidium] framebuffer {}x{} bpp={} pitch={} base={:#018x}",
            fb.width,
            fb.height,
            fb.bpp,
            fb.pitch,
            fb.base
        );
    } else {
        kprintln!("[praesidium] framebuffer: none");
    }
}

fn dump_memmap(bi: &WardenBootInfo) {
    let count = bi.memmap.count;
    kprintln!("[praesidium] memmap regions={}", count);
    if bi.memmap.regions == 0 || count == 0 {
        kprintln!("[praesidium] memmap: empty or null region array");
        return;
    }

    let walk = if count > MAX_REGIONS {
        kprintln!(
            "[praesidium] WARN: region count {} exceeds cap {}; truncating",
            count,
            MAX_REGIONS
        );
        MAX_REGIONS
    } else {
        count
    };

    // The region-array pointer is a PHYSICAL address; reach it through the HHDM.
    // (Strictly validating that the whole array stays inside Warden's mapped HHDM
    // window lands in P1, when Praesidium owns the memory map and knows the window;
    // in P0 an out-of-window pointer simply faults, which fails the boot closed.)
    let base_virt = bi.memmap.regions.wrapping_add(bi.hhdm_offset) as *const MemRegion;
    // A misaligned region array would make `&*base_virt.add(i)` UB before any load.
    if (base_virt as usize) & (core::mem::align_of::<MemRegion>() - 1) != 0 {
        kprintln!("[praesidium] FATAL: misaligned memmap region array");
        crate::arch::halt();
    }

    let mut usable_pages: u64 = 0;
    for i in 0..walk {
        // SAFETY: `base_virt` is 8-aligned (checked above) and `i < walk <= count`;
        // Warden guarantees `regions[0..count]` is a valid MemRegion array at the
        // physical base, mapped via the HHDM. Read-only.
        let r = unsafe { &*base_virt.add(i as usize) };
        if r.kind == MemoryKind::USABLE {
            usable_pages = usable_pages.saturating_add(r.pages);
        }
        if i < MAX_DUMP_ROWS {
            kprintln!(
                "[praesidium]   [{:>3}] base={:#018x} pages={:<10} {}",
                i,
                r.base,
                r.pages,
                kind_name(r.kind)
            );
        } else if i == MAX_DUMP_ROWS {
            kprintln!("[praesidium]   ... ({} more regions)", walk - MAX_DUMP_ROWS);
        }
    }

    let bytes = usable_pages.saturating_mul(handoff::PAGE_SIZE);
    kprintln!(
        "[praesidium] usable RAM: {} pages ({} MiB), walked via HHDM",
        usable_pages,
        bytes / (1024 * 1024)
    );
}

/// Short label for a [`MemoryKind`]. Matches on the inner value so an out-of-range
/// kind coming across the ABI is handled, not UB.
fn kind_name(k: MemoryKind) -> &'static str {
    match k.0 {
        0 => "USABLE",
        1 => "RESERVED",
        2 => "ACPI_RECLAIM",
        3 => "ACPI_NVS",
        4 => "MMIO",
        5 => "BOOT_RECLAIM",
        6 => "KERNEL+MODULES",
        7 => "FRAMEBUFFER",
        8 => "BAD_MEMORY",
        _ => "UNKNOWN",
    }
}
