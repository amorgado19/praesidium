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
const AP_RO: u64 = 1 << 7; // read-only at EL1 (clear = read-write)
const PXN: u64 = 1 << 53; // privileged execute-never (clear = EL1-executable)
const UXN: u64 = 1 << 54; // unprivileged execute-never (kernel pages: always set)

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
