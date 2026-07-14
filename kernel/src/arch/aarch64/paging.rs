//! aarch64 page-table construction, activation, and walking (ADR-0007 `VSpace` seam, P1).
//!
//! Builds Praesidium's own VMSAv8-64 tables (4 KiB granule): **TTBR0** identity-maps
//! the low half (2 MiB blocks, Device below the 1 GiB RAM base for PL011 MMIO, Normal
//! above); **TTBR1** maps the HHDM (2 MiB blocks) and the kernel image (4 KiB pages,
//! per-section W^X). It **reuses Warden's `TCR_EL1`/`MAIR_EL1`** (same granule + attr
//! indices: 0 = Normal WB, 1 = Device), swapping only the TTBRs. The switch keeps the
//! kernel `.text` and the boot stack (both in the TTBR1 kernel mapping) valid, so the
//! CPU keeps running; explicit `dsb`/`isb`/`tlbi` make the change coherent
//! (DEC-0007-4).

use core::arch::asm;

use crate::arch::{AddressSpace, KernelMap, Prot};
use crate::memory::{alloc_zeroed_frame, phys_to_virt};

const PAGE: u64 = 4096;
const GIB: u64 = 1 << 30;
/// Output-address field of a descriptor (bits 12..=47, 4 KiB granule).
const ADDR_MASK: u64 = 0x0000_ffff_ffff_f000;
/// On the QEMU `virt` machine, RAM starts at 1 GiB; everything below is MMIO.
const RAM_BASE: u64 = 0x4000_0000;

const VALID: u64 = 1 << 0;
/// Low bits for a table descriptor (L0..L2 → next level) or an L3 page descriptor.
const TABLE: u64 = 0b11;
/// Low bits for a block descriptor (an L1/L2 leaf).
const BLOCK: u64 = 0b01;
const AF: u64 = 1 << 10; // access flag (or the access faults)
const SH_INNER: u64 = 0b11 << 8; // inner shareable
const ATTR_NORMAL: u64 = 0 << 2; // AttrIndx 0 → MAIR attr0 (Normal WB)
const ATTR_DEVICE: u64 = 1 << 2; // AttrIndx 1 → MAIR attr1 (Device nGnRnE)
const ATTR_TAGGED: u64 = 2 << 2; // AttrIndx 2 → MAIR attr2 (Normal WB Tagged / MTE, P5b)
const AP_RO: u64 = 1 << 7; // AP[2]: read-only (clear = read-write)
const AP_EL0: u64 = 1 << 6; // AP[1]: accessible at EL0 (clear = EL1-only) — user pages set this
const PXN: u64 = 1 << 53; // privileged execute-never (clear = EL1-executable)
const UXN: u64 = 1 << 54; // unprivileged (EL0) execute-never (kernel pages: always set)

/// Table index for `vaddr` at translation `level` (0 = L0 … 3 = L3).
fn lidx(vaddr: u64, level: u32) -> usize {
    ((vaddr >> (12 + 9 * (3 - level))) & 0x1ff) as usize
}

fn alloc_table() -> u64 {
    alloc_zeroed_frame().expect("out of frames building page tables")
}

fn write_entry(table: u64, index: usize, value: u64) {
    // SAFETY: `table` is a freshly-allocated, HHDM-mapped table frame we own, and
    // `index < 512`, so the write stays within the frame.
    unsafe {
        (phys_to_virt(table) as *mut u64)
            .add(index)
            .write_volatile(value)
    };
}

fn read_entry(table: u64, index: usize) -> u64 {
    // SAFETY: `table` is an HHDM-mapped page-table frame; `index < 512`; read-only.
    unsafe {
        (phys_to_virt(table) as *const u64)
            .add(index)
            .read_volatile()
    }
}

/// Descriptor for a 4 KiB kernel page, per protection (Normal memory, kernel-only).
fn page_desc(pa: u64, prot: Prot) -> u64 {
    let base = pa | AF | SH_INNER | ATTR_NORMAL | TABLE | UXN; // `TABLE` = valid page at L3
    match prot {
        Prot::Rx => base | AP_RO,       // read-only + EL1-executable (PXN clear)
        Prot::Ro => base | AP_RO | PXN, // read-only + non-executable
        Prot::Rw => base | PXN,         // read-write (AP clear) + non-executable
    }
}

/// Map `[0, bytes)` of low physical memory into the region rooted at `l0[l0_index]`
/// using 2 MiB blocks (Device below `RAM_BASE`, Normal above), non-executable.
fn map_low(l0: u64, l0_index: usize, bytes: u64) {
    let l1 = alloc_table();
    write_entry(l0, l0_index, l1 | TABLE);
    for gib in 0..bytes.div_ceil(GIB) {
        let l2 = alloc_table();
        write_entry(l1, gib as usize, l2 | TABLE);
        for j in 0..512u64 {
            let pa = (gib << 30) | (j << 21);
            let desc = if pa < RAM_BASE {
                pa | AF | ATTR_DEVICE | BLOCK | PXN | UXN
            } else {
                pa | AF | SH_INNER | ATTR_NORMAL | BLOCK | PXN | UXN
            };
            write_entry(l2, j as usize, desc);
        }
    }
}

/// Map the kernel image (`<= 2 MiB`, 2 MiB-aligned base → one L3 table) with W^X.
fn map_kernel(l0: u64, km: &KernelMap) {
    let l1 = alloc_table();
    write_entry(l0, lidx(km.kernel_vbase, 0), l1 | TABLE);
    let l2 = alloc_table();
    write_entry(l1, lidx(km.kernel_vbase, 1), l2 | TABLE);
    let l3 = alloc_table();
    write_entry(l2, lidx(km.kernel_vbase, 2), l3 | TABLE);

    let mut v = km.kernel_vbase;
    while v < km.kernel_vend {
        let pa = km.kernel_phys + (v - km.kernel_vbase);
        write_entry(l3, lidx(v, 3), page_desc(pa, section_prot(km, v)));
        v += PAGE;
    }
}

fn section_prot(km: &KernelMap, v: u64) -> Prot {
    if (km.text.0..km.text.1).contains(&v) {
        Prot::Rx
    } else if (km.data.0..km.data.1).contains(&v) {
        Prot::Rw
    } else {
        Prot::Ro // .rodata + any inter-section padding
    }
}

/// Build Praesidium's own address space: `primary` = TTBR1 (HHDM + kernel, high half),
/// `secondary` = TTBR0 (identity, low half, incl. the PL011 MMIO the serial driver uses).
#[must_use]
pub fn build_address_space(km: &KernelMap) -> AddressSpace {
    let ttbr0 = alloc_table();
    map_low(ttbr0, 0, km.identity_bytes);
    let ttbr1 = alloc_table();
    map_low(ttbr1, lidx(km.hhdm_offset, 0), km.identity_bytes);
    map_kernel(ttbr1, km);
    AddressSpace {
        primary: ttbr1,
        secondary: ttbr0,
    }
}

/// Switch to the built address space (loads TTBR0/TTBR1), reusing Warden's TCR/MAIR.
///
/// # Safety
/// The space must map the current PC (executable), the current stack, and the HHDM, or
/// the CPU faults immediately after the switch.
pub unsafe fn activate_address_space(space: AddressSpace) {
    // SAFETY: precondition delegated to the caller (see doc). The barrier sequence makes
    // the new table writes visible to the walker, applies the TTBRs, and flushes stale
    // TLB entries — an explicit arch-seam obligation (DEC-0007-4).
    unsafe {
        asm!(
            "dsb ish",
            "msr ttbr0_el1, {t0}",
            "msr ttbr1_el1, {t1}",
            "isb",
            "tlbi vmalle1",
            "dsb ish",
            "isb",
            t0 = in(reg) space.secondary,
            t1 = in(reg) space.primary,
            options(nostack, preserves_flags),
        );
    }
}

/// On aarch64 there is nothing to establish: EL1 read-only (`AP`) and execute-never
/// (`PXN`/`UXN`) are enforced unconditionally whenever the MMU is on — no CR0.WP-style
/// disable exists. Kept as a seam method so both backends expose the same surface.
pub fn enable_wx() {}

/// Physical root for `vaddr` in the active tables: high half → TTBR1, low → TTBR0.
fn current_root(vaddr: u64) -> u64 {
    let ttbr: u64;
    if vaddr & (1 << 63) != 0 {
        // SAFETY: reading TTBR1_EL1 is side-effect-free.
        unsafe {
            asm!("mrs {}, ttbr1_el1", out(reg) ttbr, options(nomem, nostack, preserves_flags))
        };
    } else {
        // SAFETY: reading TTBR0_EL1 is side-effect-free.
        unsafe {
            asm!("mrs {}, ttbr0_el1", out(reg) ttbr, options(nomem, nostack, preserves_flags))
        };
    }
    ttbr & ADDR_MASK
}

/// Walk the active tables for `vaddr`, returning `(leaf descriptor, page offset mask)`.
fn walk(vaddr: u64) -> Option<(u64, u64)> {
    let mut table = current_root(vaddr);
    // Offset mask if a block/page leaf terminates at each level (L0 has no block form).
    const MASK: [u64; 4] = [0, 0x3fff_ffff, 0x1f_ffff, 0xfff];
    for level in 0..4u32 {
        let e = read_entry(table, lidx(vaddr, level));
        if e & VALID == 0 {
            return None;
        }
        if level == 3 {
            return Some((e, 0xfff)); // L3 page
        }
        if e & 0b11 == BLOCK {
            return Some((e, MASK[level as usize])); // L1/L2 block leaf
        }
        table = e & ADDR_MASK;
    }
    None
}

/// Translate a virtual address to physical using the active tables, or `None`.
#[must_use]
pub fn translate(vaddr: u64) -> Option<u64> {
    let (leaf, mask) = walk(vaddr)?;
    Some((leaf & ADDR_MASK & !mask) + (vaddr & mask))
}

/// Report a mapped page's `(writable, executable-at-EL1)` protection, or `None`.
#[must_use]
pub fn page_prot(vaddr: u64) -> Option<(bool, bool)> {
    let (leaf, _) = walk(vaddr)?;
    Some((leaf & AP_RO == 0, leaf & PXN == 0))
}

/// Install a **guard page** (isolation Layer 3, ADR-0008): render the 4 KiB page at `vaddr`
/// invalid in the *active* tables, first **splitting** the covering 2 MiB block down to 4 KiB
/// (L3) granularity if necessary (the HHDM is block-mapped). A stack overflow into the guard
/// then takes a translation fault instead of silently corrupting the neighbouring allocation.
///
/// The split preserves every other translation in the 2 MiB window (same output base, same
/// attributes) — only the guard page changes. The L3 table is built in full and published with
/// a `dsb ishst` *before* the L2 block descriptor is swung to a table descriptor (a single
/// aligned store), so the walker never observes a half-built table. A `tlbi vaae1is` for the
/// guard VA evicts any stale block/page TLB entry — the DEC-0007-4 arch-seam obligation.
///
/// # Safety
/// `vaddr` must name a page the caller owns and intends to make inaccessible *at this virtual
/// address* — e.g. the HHDM page immediately below a task stack, so a downward overflow faults.
/// This unmaps only `vaddr`'s mapping; note a sub-4 GiB frame is *also* aliased in the low identity
/// map (TTBR0), so making the underlying *frame* unreachable requires unmapping every alias. The
/// caller must not free the underlying frame without first restoring this mapping.
pub unsafe fn install_guard_page(vaddr: u64) {
    let v = vaddr & !0xfff; // page-align
    let l3 = ensure_l3_table(v);
    write_entry(l3, lidx(v, 3), 0); // invalid descriptor ⇒ the guard
    flush_page(v);
}

/// Remap the 4 KiB page at `vaddr` as **Normal-Tagged** (MTE, `AttrIndx = 2`) — read-write,
/// non-executable, inner-shareable — splitting the covering 2 MiB block first if needed. The
/// physical backing is unchanged; only the memory *type* becomes Tagged, so MTE tag checks now
/// apply to every access of the page (the P5b aarch64 Layer-2 domain victim). The Tagged MAIR
/// attribute must already be installed at index 2 (see [`super::mte::enable`]).
///
/// # Safety
/// `vaddr` must be an HHDM page the caller owns. Once it is Tagged, every access is tag-checked,
/// so the caller MUST set an allocation tag (STG) matching the pointers it will use before access.
pub unsafe fn map_tagged(vaddr: u64) {
    let v = vaddr & !0xfff;
    let l3 = ensure_l3_table(v);
    let pa = read_entry(l3, lidx(v, 3)) & ADDR_MASK; // keep the page's current physical backing
    write_entry(
        l3,
        lidx(v, 3),
        pa | AF | SH_INNER | ATTR_TAGGED | TABLE | UXN | PXN,
    );
    flush_page(v);
}

/// Walk to `v`'s L2 entry and, if it is a 2 MiB block, split it into a fresh 512-entry L3 table
/// with identical attributes; returns the L3 table's physical address. The table is published
/// with `dsb ishst` before the L2 descriptor is swung to point at it (a single aligned store), so
/// the walker never observes a half-built table — the split preserves every other translation in
/// the window. Shared by [`install_guard_page`] and [`map_tagged`]. `v` must be page-aligned.
fn ensure_l3_table(v: u64) -> u64 {
    let root = current_root(v);
    let e0 = read_entry(root, lidx(v, 0));
    assert!(e0 & VALID != 0, "split: L0 entry absent");
    let e1 = read_entry(e0 & ADDR_MASK, lidx(v, 1));
    assert!(
        e1 & VALID != 0 && e1 & 0b11 == TABLE,
        "split: unexpected L1 block"
    );
    let l2 = e1 & ADDR_MASK;
    let e2 = read_entry(l2, lidx(v, 2));
    assert!(e2 & VALID != 0, "split: L2 entry absent");
    if e2 & 0b11 == BLOCK {
        let base = e2 & ADDR_MASK & !0x1f_ffff; // 2 MiB-aligned output base
        let attrs = e2 & !ADDR_MASK & !0b11; // AF|SH|AttrIndx|AP|PXN|UXN (drop the descriptor bits)
        let new_l3 = alloc_table();
        for j in 0..512u64 {
            write_entry(new_l3, j as usize, (base + j * PAGE) | attrs | TABLE);
        }
        // SAFETY: `dsb ishst` publishes the L3 stores before the L2 descriptor swing below, so a
        // walker that follows the new table pointer sees fully-initialised entries.
        unsafe { asm!("dsb ishst", options(nostack, preserves_flags)) };
        write_entry(l2, lidx(v, 2), new_l3 | TABLE);
        new_l3
    } else {
        e2 & ADDR_MASK
    }
}

/// Invalidate the last-level TLB entry (block or page) for page-aligned `v` and re-synchronise —
/// the explicit arch-seam sequence after a leaf edit (DEC-0007-4).
fn flush_page(v: u64) {
    // SAFETY: order prior table stores, invalidate the VA's last-level entry, then re-sync.
    unsafe {
        asm!(
            "dsb ishst",
            "tlbi vaae1is, {page}",
            "dsb ish",
            "isb",
            page = in(reg) v >> 12,
            options(nostack, preserves_flags),
        );
    }
}

/// Map the 4 KiB page at `vaddr` to physical `phys` with protection `prot` in the active tables
/// (splitting the covering 2 MiB block first if needed), enforcing W^X via [`page_desc`]. Used by
/// the P6 loader to place a `.pex` segment at its declared virtual address — a `Prot::Rx` code
/// page is mapped read-execute at EL1, never writable.
///
/// # Safety
/// `vaddr` must be a page the caller controls in the single address space; `phys` must be a frame
/// the caller owns. Overwrites any existing mapping of `vaddr` (e.g. its identity alias).
pub unsafe fn map_page(vaddr: u64, phys: u64, prot: Prot) {
    let v = vaddr & !0xfff;
    let l3 = ensure_l3_table(v);
    write_entry(l3, lidx(v, 3), page_desc(phys & ADDR_MASK, prot));
    flush_page(v);
}

/// Descriptor for a 4 KiB **user (EL0-accessible)** page, per protection. Sets `AP[1]` (EL0
/// access); a `Prot::Rx` code page clears `UXN` (EL0-executable) but always sets `PXN` — the
/// kernel must never execute userspace code (a defence against a bug jumping into user code at
/// EL1). W^X holds: `Prot` has no writable-and-executable variant.
fn user_page_desc(pa: u64, prot: Prot) -> u64 {
    let base = pa | AF | SH_INNER | ATTR_NORMAL | TABLE | AP_EL0 | PXN; // EL0-accessible, EL1-noexec
    match prot {
        Prot::Rx => base | AP_RO,       // read-only + EL0-executable (UXN clear)
        Prot::Ro => base | AP_RO | UXN, // read-only + non-executable
        Prot::Rw => base | UXN,         // read-write (AP[2] clear) + non-executable
    }
}

/// Map the 4 KiB page at `vaddr` to physical `phys` as an **EL0-accessible** page with protection
/// `prot` (splitting the covering 2 MiB block first if needed). Used by the P7 loader to place a
/// userspace process's `.pex` segments so the process can reach them at EL0, while everything else
/// (HHDM, kernel) stays supervisor-only — so an EL0 raw pointer into kernel memory faults.
///
/// `domain` is the process's isolation domain (P7b-ii). On x86 it becomes the page's PKU key; on
/// aarch64 it will drive MTE allocation-tagging (Normal-Tagged + granule tag), so a cross-domain
/// raw read tag-faults — wired in the aarch64-MTE increment. Accepted now to keep the seam
/// identical across arches.
///
/// # Safety
/// `vaddr` must be a page in the process's reserved VA window the caller controls; `phys` a frame
/// the caller owns. Overwrites any existing mapping of `vaddr`.
pub unsafe fn map_user_page(vaddr: u64, phys: u64, prot: Prot, domain: u64) {
    let _ = domain; // aarch64 MTE domain-tagging is wired in the aarch64-MTE increment (task #16)
    let v = vaddr & !0xfff;
    let l3 = ensure_l3_table(v);
    write_entry(l3, lidx(v, 3), user_page_desc(phys & ADDR_MASK, prot));
    flush_page(v);
}

/// Program the isolation domain for the task the scheduler is switching in (P7b-ii). On aarch64 the
/// MTE mechanism carries the domain in the process's *pointer tag* (set on EL0 entry), so there is
/// no per-switch domain register to load — this is a no-op, kept to mirror the x86 `PKRU` switch
/// (and a future per-domain-page-table fallback would swing `TTBR0` here).
pub fn set_domain(_domain: Option<u64>) {}

/// Make `len` bytes at `vaddr` coherent for instruction fetch after code has been written there
/// through the data path (ADR-0006 / GC-09 — the cache-maintenance seam owed since P0, now that
/// P6 first copies-then-executes code). **Load-bearing on aarch64:** the I- and D-caches are NOT
/// coherent, so freshly-written code is fetched stale unless the D-cache is cleaned to the Point
/// of Unification and the I-cache invalidated. The canonical sequence, per line size read from
/// `CTR_EL0` (a larger step would skip lines): `dc cvau` over the range → `dsb ish` → `ic ivau`
/// over the range → `dsb ish` → `isb`. Cache ops are by VA but act on the PA (PIPT), so cleaning
/// via the write alias covers execution through any alias of the same frame.
pub fn sync_instruction_cache(vaddr: u64, len: usize) {
    if len == 0 {
        return;
    }
    let ctr: u64;
    // SAFETY: reading CTR_EL0 is side-effect-free; it reports the cache line geometry.
    unsafe { asm!("mrs {}, ctr_el0", out(reg) ctr, options(nomem, nostack, preserves_flags)) };
    let dline = 4u64 << ((ctr >> 16) & 0xf); // DminLine: log2(words) in bits [19:16]
    let iline = 4u64 << (ctr & 0xf); // IminLine: log2(words) in bits [3:0]
    let end = vaddr + len as u64;

    // Clean D-cache to PoU across the range, then publish before invalidating the I-cache.
    let mut a = vaddr & !(dline - 1);
    while a < end {
        // SAFETY: `dc cvau` cleans the data cache line for `a` to the Point of Unification; `a`
        // is a mapped, readable address within the range the caller just wrote.
        unsafe { asm!("dc cvau, {}", in(reg) a, options(nostack, preserves_flags)) };
        a += dline;
    }
    // SAFETY: order the D-cache cleans before the I-cache invalidations.
    unsafe { asm!("dsb ish", options(nostack, preserves_flags)) };

    let mut a = vaddr & !(iline - 1);
    while a < end {
        // SAFETY: `ic ivau` invalidates the instruction cache line for `a` to the PoU.
        unsafe { asm!("ic ivau, {}", in(reg) a, options(nostack, preserves_flags)) };
        a += iline;
    }
    // SAFETY: complete the invalidations and re-synchronise the fetch stream.
    unsafe { asm!("dsb ish", "isb", options(nostack, preserves_flags)) };
}
