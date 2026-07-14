//! x86-64 page-table construction, activation, and walking (ADR-0007 `VSpace` seam, P1).
//!
//! Builds Praesidium's own 4-level (PML4) hierarchy — identity + HHDM via 2 MiB
//! pages, the kernel image via 4 KiB pages with per-section W^X — and switches CR3
//! to it. Table frames come from the buddy allocator (physical) and are edited
//! through the HHDM. The layout mirrors Warden's handoff so the boot stack (low
//! identity) and current RIP (kernel `.text`, high half) stay mapped across the
//! `mov cr3`; the only change is W^X on the kernel image.

use core::arch::asm;
use core::sync::atomic::{AtomicU64, Ordering};

use crate::arch::{AddressSpace, KernelMap, Prot};
use crate::memory::{alloc_zeroed_frame, phys_to_virt};

/// The pristine kernel base PML4 (kernel image + identity + HHDM, **NO user process pages**),
/// captured when [`build_address_space`] builds it. Each userspace process runs on its OWN page
/// table — a clone of this base plus only its own pages ([`new_process_space`]) — so a process
/// cannot even NAME another's memory: the victim's pages are simply not mapped in the attacker's
/// table (P7b-ii hostile-isolation boundary, ADR-0008 DEC-0008-6). Kernel tasks run on this base.
static KERNEL_PML4: AtomicU64 = AtomicU64::new(0);

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
    KERNEL_PML4.store(pml4, Ordering::Relaxed); // the pristine base every process table clones
    AddressSpace {
        primary: pml4,
        secondary: 0,
    }
}

/// The pristine kernel base address space (kernel + identity + HHDM, no user pages) — what kernel
/// tasks run on, and what [`new_process_space`] clones. Valid after [`build_address_space`].
#[must_use]
pub fn kernel_space() -> AddressSpace {
    AddressSpace {
        primary: KERNEL_PML4.load(Ordering::Relaxed),
        secondary: 0,
    }
}

/// Build a fresh per-process page table (P7b-ii hostile isolation): a clone of the kernel base with a
/// PRIVATE sub-tree for the process VA window `[1 GiB, 2 GiB)` (PML4[0] → PDPT[1] → PD), so this
/// process's user pages — mapped into that PD by the loader — are invisible to every other table.
/// Everything else (identity below 1 GiB, HHDM, kernel high half) is shared read-only. A process
/// that forms a raw pointer to another process's VA finds, in ITS table, the shared **supervisor**
/// identity 2 MiB huge page (never a user page), so an EL0/ring-3 access permission-faults — a
/// boundary no userspace instruction can cross (unlike PKU's `WRPKRU` / MTE tag-forge). Deep-copies
/// exactly three tables; the loader then splits the huge pages in the private PD to place user pages.
#[must_use]
pub fn new_process_space() -> AddressSpace {
    let kbase = KERNEL_PML4.load(Ordering::Relaxed);
    let clone_table = |orig: u64| -> u64 {
        let new = alloc_table();
        for i in 0..512 {
            write_entry(new, i, read_entry(orig, i));
        }
        new
    };
    // PML4 clone; then a private low-half PDPT (PML4[0]); then a private PD for the process window
    // (PDPT[1] = [1 GiB, 2 GiB), identity-mapped as 2 MiB huge pages since IDENTITY_BYTES = 4 GiB).
    let new_pml4 = clone_table(kbase);
    let e0 = read_entry(kbase, 0); // PML4[0] -> low-half PDPT (always present: identity + window)
    let new_pdpt = clone_table(e0 & ADDR_MASK);
    write_entry(new_pml4, 0, new_pdpt | (e0 & !ADDR_MASK));
    let e1 = read_entry(e0 & ADDR_MASK, 1); // PDPT[1] -> PD covering [1 GiB, 2 GiB)
    let new_pd = clone_table(e1 & ADDR_MASK);
    write_entry(new_pdpt, 1, new_pd | (e1 & !ADDR_MASK));
    AddressSpace {
        primary: new_pml4,
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

/// Invalidate any TLB entry (4 KiB or 2 MiB) caching a translation for `vaddr`.
fn invlpg(vaddr: u64) {
    // SAFETY: `invlpg` flushes the TLB entry for the given linear address and has no
    // other effect; the operand is a memory reference (so no `nomem`).
    unsafe { asm!("invlpg [{}]", in(reg) vaddr, options(nostack, preserves_flags)) };
}

/// Install a **guard page** (isolation Layer 3, ADR-0008): render the 4 KiB page at `vaddr`
/// non-present in the *active* tables, first **splitting** the covering 2 MiB huge page down
/// to 4 KiB granularity if necessary (the HHDM is huge-page-mapped). A stack overflow into the
/// guard then `#PF`s loudly instead of silently corrupting the neighbouring allocation. The
/// TLB entry for `vaddr` is flushed so the change takes effect immediately.
///
/// The split preserves every other translation in the 2 MiB window (same physical base, same
/// attributes) — only the one guard page changes — so it is safe to apply to the live kernel
/// tables. The new PT is built in full *before* the PD entry is swung from huge-page to
/// table-pointer (a single aligned 64-bit store), so the page-table walker never observes a
/// half-built table.
///
/// # Safety
/// `vaddr` must name a page the caller owns and intends to make inaccessible *at this virtual
/// address* — e.g. the HHDM page immediately below a task stack, so a downward overflow faults.
/// This unmaps only `vaddr`'s mapping; note a sub-4 GiB frame is *also* aliased in the low identity
/// map, so to make the underlying *frame* unreachable (not just this VA) the caller must unmap every
/// alias. The caller must not later free the underlying frame without first restoring this mapping
/// (a re-allocated frame would otherwise inherit a non-present alias).
pub unsafe fn install_guard_page(vaddr: u64) {
    let v = vaddr & !0xfff; // page-align
    let pt = ensure_pt(v);
    write_entry(pt, idx(v, 0), 0); // non-present ⇒ the guard
    invlpg(v);
}

/// Walk to `v`'s PD entry and, if it is a 2 MiB huge page, split it into a fresh 512-entry PT with
/// identical attributes; returns the PT's physical address. The PT is built in full before the PD
/// entry is swung to point at it (a single aligned store), so the walker never sees a half-built
/// table — the split preserves every other translation in the window. Shared by
/// [`install_guard_page`] and [`map_page`]. `v` must be page-aligned.
fn ensure_pt(v: u64) -> u64 {
    let e4 = read_entry(current_pml4(), idx(v, 3));
    assert!(e4 & PRESENT != 0, "ensure_pt: PML4 entry absent");
    let e3 = read_entry(e4 & ADDR_MASK, idx(v, 2));
    assert!(
        e3 & PRESENT != 0 && e3 & HUGE == 0,
        "ensure_pt: unexpected 1 GiB mapping"
    );
    let pd = e3 & ADDR_MASK;
    let e2 = read_entry(pd, idx(v, 1));
    assert!(e2 & PRESENT != 0, "ensure_pt: PD entry absent");
    if e2 & HUGE != 0 {
        let base = e2 & ADDR_MASK & !0x1f_ffff; // 2 MiB-aligned physical base
        let flags = e2 & !ADDR_MASK & !HUGE; // PRESENT|WRITABLE|NX (drop the PS/huge bit)
        let new_pt = alloc_table();
        for j in 0..512u64 {
            write_entry(new_pt, j as usize, (base + j * PAGE) | flags);
        }
        write_entry(pd, idx(v, 1), new_pt | PRESENT | WRITABLE);
        new_pt
    } else {
        e2 & ADDR_MASK
    }
}

/// Map the 4 KiB page at `vaddr` to physical `phys` with protection `prot` in the active tables
/// (splitting the covering 2 MiB page first if needed), enforcing W^X via [`leaf_bits`]. Used by
/// the P6 loader to place a `.pex` segment at its declared virtual address — a `Prot::Rx` code
/// page is mapped read-execute, never writable (the writable HHDM alias the loader used to copy
/// the bytes is a separate mapping the loaded process cannot name).
///
/// # Safety
/// `vaddr` must be a page the caller owns/controls in the single address space; `phys` must be a
/// frame the caller owns. Overwrites any existing mapping of `vaddr` (e.g. its identity alias).
pub unsafe fn map_page(vaddr: u64, phys: u64, prot: Prot) {
    let v = vaddr & !0xfff;
    let pt = ensure_pt(v);
    write_entry(pt, idx(v, 0), (phys & ADDR_MASK) | leaf_bits(prot));
    invlpg(v);
}

/// Page-table entry bit granting ring-3 (user) access to a page (U/S).
const USER: u64 = 1 << 2;

/// Like [`ensure_pt`], but also sets the USER (U/S) bit on every intermediate entry along the path
/// (PML4E, PDPTE, PDE). **Load-bearing on x86:** a ring-3 access is permitted only if USER is set at
/// EVERY paging level, so setting it solely on the leaf makes the walk fault at the supervisor
/// intermediate the identity map created. Setting USER on an intermediate is *permissive* — the
/// leaf's own USER bit is the actual per-page gate, so kernel pages sharing these intermediates keep
/// leaf USER=0 and stay supervisor-only. When a covering 2 MiB huge page is split, the copied leaves
/// get NO USER bit (they stay supervisor); only the PDE pointing at the new PT gets USER. Returns
/// the PT's physical address. `v` must be page-aligned.
fn ensure_pt_user(v: u64) -> u64 {
    let pml4 = current_pml4();
    let e4 = read_entry(pml4, idx(v, 3));
    assert!(e4 & PRESENT != 0, "ensure_pt_user: PML4 entry absent");
    if e4 & USER == 0 {
        write_entry(pml4, idx(v, 3), e4 | USER);
    }
    let pdpt = e4 & ADDR_MASK;
    let e3 = read_entry(pdpt, idx(v, 2));
    assert!(
        e3 & PRESENT != 0 && e3 & HUGE == 0,
        "ensure_pt_user: unexpected 1 GiB mapping"
    );
    if e3 & USER == 0 {
        write_entry(pdpt, idx(v, 2), e3 | USER);
    }
    let pd = e3 & ADDR_MASK;
    let e2 = read_entry(pd, idx(v, 1));
    assert!(e2 & PRESENT != 0, "ensure_pt_user: PD entry absent");
    if e2 & HUGE != 0 {
        // Split the covering 2 MiB huge page: the copied 4 KiB leaves keep the huge page's attrs
        // WITHOUT USER (they stay supervisor); only the PDE pointing at the new PT gets USER.
        let base = e2 & ADDR_MASK & !0x1f_ffff;
        let flags = e2 & !ADDR_MASK & !HUGE; // PRESENT|WRITABLE|NX (drop PS/huge)
        let new_pt = alloc_table();
        for j in 0..512u64 {
            write_entry(new_pt, j as usize, (base + j * PAGE) | flags);
        }
        write_entry(pd, idx(v, 1), new_pt | PRESENT | WRITABLE | USER);
        new_pt
    } else {
        if e2 & USER == 0 {
            write_entry(pd, idx(v, 1), e2 | USER);
        }
        e2 & ADDR_MASK
    }
}

/// Map the 4 KiB page at `vaddr` to physical `phys` as a **ring-3 (user) accessible** page with
/// protection `prot` (W^X via [`leaf_bits`]), splitting the covering 2 MiB page first if needed and
/// setting the USER bit on every paging level (see [`ensure_pt_user`]). Used by the P7 loader to
/// place a userspace process's segments so ring 3 can reach them, while the kernel/HHDM stay
/// supervisor-only (a ring-3 raw pointer into kernel memory faults on the supervisor-only leaf).
///
/// `domain` is the process's isolation domain (P7b-ii): its low 4 bits become the page's PKU
/// **protection key** (PTE bits [62:59]), so — with `CR4.PKE` on and a per-process `PKRU` set on
/// switch-in ([`super::pku`]) — a *different* process's raw read of this page takes a PK `#PF`.
/// The key bits are ignored by hardware when PKU is absent, so tagging is harmless on the fallback.
///
/// # Safety
/// `vaddr` must be a page in the process's reserved VA window the caller controls; `phys` a frame
/// the caller owns. Overwrites any existing mapping of `vaddr`.
pub unsafe fn map_user_page(vaddr: u64, phys: u64, prot: Prot, domain: u64) {
    let v = vaddr & !0xfff;
    let pt = ensure_pt_user(v);
    write_entry(
        pt,
        idx(v, 0),
        (phys & ADDR_MASK) | leaf_bits(prot) | USER | super::pku::pkey_bits(domain),
    );
    invlpg(v);
}

/// Make `len` bytes at `vaddr` coherent for instruction fetch after the loader has written code
/// there through the data path (ADR-0006 / GC-09 — the cache-maintenance seam owed since P0, now
/// that P6 first copies-then-executes code).
///
/// On x86-64 the instruction and data caches are **coherent** for the same physical memory
/// (Intel SDM Vol 3 §11.6): a store made before an instruction fetch of newly-written code is
/// observed, so no explicit invalidation is required. We still emit a store fence to order the
/// segment writes before any later transfer of control into the code (the actual EL-transition /
/// jump into loaded code is a serializing event, added in P7). `vaddr`/`len` are unused here but
/// keep the seam identical to aarch64, where they are load-bearing.
pub fn sync_instruction_cache(_vaddr: u64, _len: usize) {
    // SAFETY: `mfence` orders prior stores (the copied code) ahead of subsequent memory ops; it
    // has no other effect. Absence of `nomem` makes it a compiler barrier too.
    unsafe { asm!("mfence", options(nostack, preserves_flags)) };
}

/// Build a **per-domain page table** (isolation Layer 2 fallback, DEC-0008-6): a clone of the
/// *active* PML4 in which the single 4 KiB page at `victim_vaddr` is unmapped, while EVERYTHING
/// else — the current code, stack, HHDM, MMIO — stays mapped identically (the lower tables are
/// shared). Switching CR3 to the returned root keeps executing, but a raw access to `victim_vaddr`
/// `#PF`s. Only the PML4→PDPT→PD→PT path covering the victim is deep-copied, so the unmap never
/// perturbs the original (kernel) tables. Returns the clone's PML4 physical address.
///
/// x86 has no in-kernel hardware protection-key domain — PKU/PKRU gates *user* pages only and
/// there is no userspace yet (P7), and this host has no PKS — so this per-domain-page-table
/// fallback is x86's Layer-2 isolation mechanism, exercised end to end here (AC5.4, honest per
/// DEC-0008-6). The in-address-space PKU fast path arrives with userspace in P7.
#[must_use]
pub fn build_domain_excluding(victim_vaddr: u64) -> u64 {
    let v = victim_vaddr & !0xfff;
    let orig_pml4 = current_pml4();

    // Deep-copy one table (all 512 entries) into a fresh frame; return the new frame's phys.
    let clone_table = |orig: u64| -> u64 {
        let new = alloc_table();
        for i in 0..512 {
            write_entry(new, i, read_entry(orig, i));
        }
        new
    };

    // Clone PML4, then deep-copy down the victim's path so edits stay local to this domain.
    let new_pml4 = clone_table(orig_pml4);

    let e4 = read_entry(orig_pml4, idx(v, 3));
    assert!(e4 & PRESENT != 0, "domain: victim PML4 entry absent");
    let new_pdpt = clone_table(e4 & ADDR_MASK);
    write_entry(new_pml4, idx(v, 3), new_pdpt | (e4 & !ADDR_MASK));

    let e3 = read_entry(e4 & ADDR_MASK, idx(v, 2));
    assert!(
        e3 & PRESENT != 0 && e3 & HUGE == 0,
        "domain: unexpected 1 GiB victim mapping"
    );
    let new_pd = clone_table(e3 & ADDR_MASK);
    write_entry(new_pdpt, idx(v, 2), new_pd | (e3 & !ADDR_MASK));

    // The victim's PD entry is a 2 MiB huge page (HHDM); split it into a fresh PT (or copy an
    // already-4 KiB PT), then unmap ONLY the victim page in this domain.
    let e2 = read_entry(e3 & ADDR_MASK, idx(v, 1));
    assert!(e2 & PRESENT != 0, "domain: victim PD entry absent");
    let new_pt = if e2 & HUGE != 0 {
        let base = e2 & ADDR_MASK & !0x1f_ffff;
        let flags = e2 & !ADDR_MASK & !HUGE;
        let pt = alloc_table();
        for j in 0..512u64 {
            write_entry(pt, j as usize, (base + j * PAGE) | flags);
        }
        pt
    } else {
        clone_table(e2 & ADDR_MASK)
    };
    write_entry(new_pd, idx(v, 1), new_pt | PRESENT | WRITABLE);
    write_entry(new_pt, idx(v, 0), 0); // the victim is unmapped in this domain

    new_pml4
}
