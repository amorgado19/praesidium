//! x86-64 page-table construction, activation, and walking (ADR-0007 `VSpace` seam, P1).
//!
//! Builds Praesidium's own 4-level (PML4) hierarchy — identity + HHDM via 2 MiB
//! pages, the kernel image via 4 KiB pages with per-section W^X — and switches CR3
//! to it. Table frames come from the buddy allocator (physical) and are edited
//! through the HHDM. The layout mirrors Warden's handoff so the boot stack (low
//! identity) and current RIP (kernel `.text`, high half) stay mapped across the
//! `mov cr3`; the only change is W^X on the kernel image.

use core::arch::asm;

use crate::arch::{AddressSpace, KernelMap, Prot};
use crate::memory::{alloc_zeroed_frame, phys_to_virt};

const PRESENT: u64 = 1 << 0;
const WRITABLE: u64 = 1 << 1;
const HUGE: u64 = 1 << 7;
const NX: u64 = 1 << 63;
/// Physical-address field of a page-table entry (bits 12..=51).
const ADDR_MASK: u64 = 0x000f_ffff_ffff_f000;
const PAGE: u64 = 4096;
const GIB: u64 = 1 << 30;

/// Table index for `vaddr` at `level` (0 = PT … 3 = PML4).
fn idx(vaddr: u64, level: u32) -> usize {
    ((vaddr >> (12 + 9 * level)) & 0x1ff) as usize
}

/// Allocate a zeroed frame for a page table (panics only on true OOM at boot).
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

/// Leaf-entry bits for a protection. NX is set for everything non-executable; the
/// `Prot` enum makes writable+executable unrepresentable (W^X).
fn leaf_bits(prot: Prot) -> u64 {
    match prot {
        Prot::Rx => PRESENT,
        Prot::Ro => PRESENT | NX,
        Prot::Rw => PRESENT | WRITABLE | NX,
    }
}

/// Map `[0, bytes)` of low physical memory into the region rooted at `pml4[pml4_index]`
/// using 2 MiB pages, writable + non-executable (identity map and HHDM both use this).
fn map_low(pml4: u64, pml4_index: usize, bytes: u64) {
    let pdpt = alloc_table();
    write_entry(pml4, pml4_index, pdpt | PRESENT | WRITABLE);
    for g in 0..bytes.div_ceil(GIB) {
        let pd = alloc_table();
        write_entry(pdpt, g as usize, pd | PRESENT | WRITABLE);
        for j in 0..512u64 {
            let pa = (g << 30) | (j << 21);
            write_entry(pd, j as usize, pa | PRESENT | WRITABLE | HUGE | NX);
        }
    }
}

/// Map the kernel image (`<= 2 MiB`, 2 MiB-aligned base → one PT) with per-section W^X.
fn map_kernel(pml4: u64, km: &KernelMap) {
    let pdpt = alloc_table();
    write_entry(pml4, idx(km.kernel_vbase, 3), pdpt | PRESENT | WRITABLE);
    let pd = alloc_table();
    write_entry(pdpt, idx(km.kernel_vbase, 2), pd | PRESENT | WRITABLE);
    let pt = alloc_table();
    write_entry(pd, idx(km.kernel_vbase, 1), pt | PRESENT | WRITABLE);

    let mut v = km.kernel_vbase;
    while v < km.kernel_vend {
        let pa = km.kernel_phys + (v - km.kernel_vbase);
        write_entry(pt, idx(v, 0), pa | leaf_bits(section_prot(km, v)));
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

/// Build Praesidium's own address space; `primary` is the PML4 physical address
/// (x86-64 uses a single root, so `secondary` is unused).
#[must_use]
pub fn build_address_space(km: &KernelMap) -> AddressSpace {
    let pml4 = alloc_table();
    map_low(pml4, 0, km.identity_bytes); // identity (low half)
    map_low(pml4, idx(km.hhdm_offset, 3), km.identity_bytes); // HHDM (higher half)
    map_kernel(pml4, km);
    AddressSpace {
        primary: pml4,
        secondary: 0,
    }
}

/// Switch to the built address space (loads CR3 from `space.primary`).
///
/// # Safety
/// The space must map the current RIP (executable), the current stack, and the HHDM,
/// or the CPU faults immediately after the switch. Loading CR3 flushes the TLB.
pub unsafe fn activate_address_space(space: AddressSpace) {
    // SAFETY: precondition delegated to the caller (see doc); the write to CR3 has no
    // other memory effects the compiler must preserve here.
    unsafe { asm!("mov cr3, {}", in(reg) space.primary, options(nostack, preserves_flags)) };
}

/// Establish the x86-64 control state W^X depends on — never inherit it from firmware.
/// Must run before activating the NX-bearing / read-only tables:
///  - **EFER.NXE=1** so the NX bit (PTE bit 63) is honored rather than reserved (a set
///    reserved bit would `#PF`); NX support is confirmed via CPUID first (fail closed).
///  - **CR0.WP=1** so ring-0 writes respect the read-only (`R/W=0`) bit — otherwise a
///    supervisor write to `.text`/`.rodata` succeeds and W^X is silently void (Intel
///    SDM Vol 3A §4.6).
pub fn enable_wx() {
    // CPUID.8000_0001h:EDX[20] = NX (execute-disable) support.
    let edx: u32;
    // SAFETY: leaf 0x8000_0001 exists on every long-mode CPU. `cpuid` clobbers rbx
    // (reserved by LLVM), so we preserve it across the instruction; the only stack use
    // is the push/pop pair (hence no `nostack`), and no flags are modified.
    unsafe {
        asm!(
            "push rbx",
            "mov eax, 0x80000001",
            "cpuid",
            "pop rbx",
            out("eax") _,
            out("ecx") _,
            out("edx") edx,
            options(preserves_flags),
        );
    }
    assert!(
        edx & (1 << 20) != 0,
        "CPU lacks NX support — W^X unenforceable"
    );

    // SAFETY: set EFER.NXE (MSR 0xC000_0080 bit 11) then CR0.WP (bit 16). rdmsr/wrmsr
    // use ecx (index) and edx:eax (value) for the MSR; `mov cr0` writes the control
    // register. These establish the W^X-enforcing control bits explicitly. The `or`
    // instructions clobber flags (so no `preserves_flags`); no memory is touched.
    unsafe {
        asm!(
            "mov ecx, 0xC0000080",
            "rdmsr",
            "or eax, {nxe}",
            "wrmsr",
            "mov {tmp}, cr0",
            "or {tmp}, {wp}",
            "mov cr0, {tmp}",
            nxe = const 1u32 << 11,
            wp = const 1u64 << 16,
            tmp = out(reg) _,
            out("eax") _,
            out("ecx") _,
            out("edx") _,
            options(nostack),
        );
    }
}

fn current_pml4() -> u64 {
    let cr3: u64;
    // SAFETY: reading CR3 is side-effect-free.
    unsafe { asm!("mov {}, cr3", out(reg) cr3, options(nomem, nostack, preserves_flags)) };
    cr3 & ADDR_MASK
}

/// Walk the *active* tables for `vaddr`, returning `(leaf entry, page offset mask)`.
fn walk(vaddr: u64) -> Option<(u64, u64)> {
    let e4 = read_entry(current_pml4(), idx(vaddr, 3));
    if e4 & PRESENT == 0 {
        return None;
    }
    let e3 = read_entry(e4 & ADDR_MASK, idx(vaddr, 2));
    if e3 & PRESENT == 0 {
        return None;
    }
    if e3 & HUGE != 0 {
        return Some((e3, 0x3fff_ffff)); // 1 GiB page
    }
    let e2 = read_entry(e3 & ADDR_MASK, idx(vaddr, 1));
    if e2 & PRESENT == 0 {
        return None;
    }
    if e2 & HUGE != 0 {
        return Some((e2, 0x1f_ffff)); // 2 MiB page
    }
    let e1 = read_entry(e2 & ADDR_MASK, idx(vaddr, 0));
    if e1 & PRESENT == 0 {
        return None;
    }
    Some((e1, 0xfff)) // 4 KiB page
}

/// Translate a virtual address to physical using the active tables, or `None` if
/// unmapped. Used to discover the kernel's physical base from its link address.
#[must_use]
pub fn translate(vaddr: u64) -> Option<u64> {
    let (leaf, mask) = walk(vaddr)?;
    Some((leaf & ADDR_MASK & !mask) + (vaddr & mask))
}

/// Report a mapped page's `(writable, executable)` protection in the active tables,
/// or `None` if unmapped. Used to verify W^X is actually in force.
#[must_use]
pub fn page_prot(vaddr: u64) -> Option<(bool, bool)> {
    let (leaf, _) = walk(vaddr)?;
    Some((leaf & WRITABLE != 0, leaf & NX == 0))
}
