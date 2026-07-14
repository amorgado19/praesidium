//! aarch64 Memory Tagging Extension (MTE) — the P5b Layer-2 hardware isolation mechanism
//! (ADR-0008 DEC-0008-3). This is the **real** intra-address-space domain enforcement: a memory
//! granule carries a 4-bit *allocation tag*, a pointer carries a 4-bit *logical tag* in its top
//! byte, and — with synchronous tag checking on — an access whose pointer tag ≠ the granule's
//! allocation tag takes a synchronous Data Abort. So a raw pointer forged into another domain's
//! region **traps**, within one shared address space and one page-table root (the SASOS win).
//!
//! Scope (per the P5 ruling): MTE is enabled globally at EL1 but only the *victim* page is mapped
//! Normal-Tagged, so only accesses to it are tag-checked — the rest of the kernel (all `AttrIndx`
//! 0 Normal memory) is untouched. This proves the mechanism on a scoped region; whole-kernel /
//! per-domain-userspace tagging is P7. PAC (pointer authentication, DEC-0008-3's other half) is a
//! distinct anti-forgery concern and is deferred to a hardening pass — see TASKS.md.
//!
//! Requires FEAT_MTE2 (synchronous checking), present under QEMU `-cpu max -machine virt,mte=on`.

use core::arch::asm;

/// Set the 4-bit MTE logical tag in the top byte (bits [59:56]) of a pointer, preserving the rest
/// of the top byte (bits [63:60], which TBI leaves as the address's sign extension).
fn with_tag(addr: u64, tag: u8) -> u64 {
    (addr & !(0xf << 56)) | ((u64::from(tag) & 0xf) << 56)
}

/// Enable synchronous MTE tag checking at **both EL1 (P5b) and EL0 (P7b-ii)** and install the
/// Normal-Tagged memory attribute at `MAIR` index 2 — idempotent (safe to call more than once).
/// Read-modify-write throughout so only the MTE-relevant bits change:
///  - `MAIR_EL1[byte 2] = 0xF0` — Normal Inner/Outer WB, **Tagged** (leaves attr0/attr1 intact).
///  - `TCR_EL1.TBI1/TBI0 = 1` — top-byte-ignore for the TTBR1 (high-half/HHDM) AND TTBR0 (low-half/
///    userspace) regions, so bits [59:56] are the pointer's logical tag rather than translated
///    address (existing pointers, whose top byte is the sign extension / zero, still translate
///    identically). `TCMA1 = TCMA0 = 0` so tag 0 is *not* exempt — a raw tag-0 pointer into a tagged
///    region faults (the P7b-ii red-team: a hostile process forms an untagged cross-domain pointer).
///  - `SCTLR_EL1.ATA/ATA0 = 1` (EL1+EL0 allocation-tag access) and `TCF/TCF0 = 0b01` (synchronous
///    tag-check faults at EL1 and EL0). An EL0 tag-check fault surfaces as a Data Abort (EC 0x24),
///    which the EL0 trap handler already routes to killing the process.
///
/// The translation-affecting writes (MAIR/TCR) are made coherent with a `tlbi`+`isb` before the
/// `SCTLR` write arms checking, and before any tagged access follows (explicit barriers, DEC-0007-4).
pub fn enable() {
    let (mut mair, mut tcr, mut sctlr): (u64, u64, u64);
    // SAFETY: reading MAIR/TCR/SCTLR_EL1 is side-effect-free.
    unsafe {
        asm!(
            "mrs {mair}, mair_el1",
            "mrs {tcr}, tcr_el1",
            "mrs {sctlr}, sctlr_el1",
            mair = out(reg) mair,
            tcr = out(reg) tcr,
            sctlr = out(reg) sctlr,
            options(nomem, nostack, preserves_flags),
        );
    }
    mair = (mair & !(0xffu64 << 16)) | (0xf0u64 << 16); // attr2 = Normal WB Tagged
    tcr |= 1u64 << 38; // TBI1 (top-byte-ignore, TTBR1/high half — EL1 pointer tags)
    tcr &= !(1u64 << 58); // TCMA1 = 0 (tag 0 is still checked at EL1)
    tcr |= 1u64 << 37; // TBI0 (top-byte-ignore, TTBR0/low half — EL0/userspace pointer tags, P7b-ii)
    tcr &= !(1u64 << 57); // TCMA0 = 0 (tag 0 is still checked at EL0 — a tag-0 raw pointer faults)
    sctlr |= 1u64 << 43; // ATA (EL1 allocation-tag access)
    sctlr = (sctlr & !(0b11u64 << 40)) | (0b01u64 << 40); // TCF = synchronous (EL1 tag-check faults)
    sctlr |= 1u64 << 42; // ATA0 (EL0 allocation-tag access, P7b-ii)
    sctlr = (sctlr & !(0b11u64 << 38)) | (0b01u64 << 38); // TCF0 = synchronous (EL0 tag-check faults)

    // SAFETY: installs an unused MAIR attr (attr2) + TBI1 (does not change existing translations)
    // + EL1 tag checking, preserving all other control bits. The tlbi+isb make the MAIR/TCR
    // reinterpretation coherent before checking is armed and before any Tagged access.
    unsafe {
        asm!(
            ".arch_extension memtag",
            "msr mair_el1, {mair}",
            "msr tcr_el1, {tcr}",
            "isb",
            "tlbi vmalle1",
            "dsb ish",
            "isb",
            "msr sctlr_el1, {sctlr}",
            "msr tco, #0", // clear PSTATE.TCO — otherwise tag checks are overridden (disabled)
            "isb",
            mair = in(reg) mair,
            tcr = in(reg) tcr,
            sctlr = in(reg) sctlr,
            options(nostack, preserves_flags),
        );
    }
}

/// Set every 16-byte allocation-tag granule of the 4 KiB page at `hhdm_base` to `tag` (P7b-ii). The
/// page MUST already be a **writable Normal-Tagged** mapping the kernel owns — the frame's HHDM
/// alias remapped via [`super::paging::map_tagged`] — since `STG` writes tag memory (needs write
/// permission + `SCTLR.ATA`). Setting the physical granule tags here makes an EL0 access through the
/// *user* alias (also Normal-Tagged) tag-checked against `tag`: a matching-tag pointer reads it, a
/// tag-0 (or foreign-tag) raw pointer faults. The tag storage is per physical granule, so which
/// alias sets it is immaterial. Barriers order the tag stores before any subsequent tagged access.
pub fn tag_frame(hhdm_base: u64, tag: u8) {
    let base = with_tag(hhdm_base & !0xfff, tag);
    for g in 0..256u64 {
        // 4096 / 16 = 256 granules per page.
        let p = base + g * 16;
        // SAFETY: `STG` sets the allocation tag of the 16-byte granule at `[p]` to `p`'s logical tag
        // (`tag`); `p` lies within a writable Normal-Tagged page the kernel owns. It reads/writes no
        // data, only tag memory. `.arch_extension memtag` lets the assembler accept STG on soft-float.
        unsafe {
            asm!(
                ".arch_extension memtag",
                "stg {p}, [{p}]",
                p = in(reg) p,
                options(nostack, preserves_flags),
            );
        }
    }
    // SAFETY: order the tag stores (to tag memory) before the granules are accessed through any
    // alias — the explicit barrier seam (DEC-0007-4).
    unsafe { asm!("dsb ish", "isb", options(nostack, preserves_flags)) };
}

/// Isolation Layer-2 escape red-team (P5b, DEC-0008-7): tag a victim granule to one domain, then
/// attempt a raw read of it through a pointer bearing a *different* domain's tag, and report
/// whether the hardware contained it (a synchronous tag-check fault). Returns `true` iff the
/// mis-tagged access faulted (isolation held); `false` means the raw read *succeeded* — a breach.
///
/// aarch64's Layer-2 mechanism is real FEAT_MTE (DEC-0008-3), within one address space — no
/// page-table swap, unlike the x86 fallback. The deliberate fault is contained by the single-shot
/// recovery seam ([`super::interrupts::contains_raw_read`]).
pub fn domain_escape_contained() -> bool {
    /// A recognizable value written through the owning tag, confirmed intact after the probe.
    const SENTINEL: u64 = 0x5A5A_1508_D0D0_BEEF;

    enable();
    let phys = crate::memory::alloc_frames(0).expect("no frame for the MTE domain victim");
    let hhdm = crate::memory::phys_to_virt(phys); // the HHDM alias (high half, to be Tagged)
    let ident = phys; // the low-half identity alias — mapped Normal-untagged, so an MTE bypass
                      // SAFETY: remap the HHDM alias Normal-Tagged so MTE checks apply to it, and unmap the *untagged*
                      // identity alias so it cannot bypass the tag check (this SASOS double-maps every frame < 4 GiB).
                      // The kernel reaches heap frames only through the HHDM, so unmapping the identity alias is sound;
                      // the kernel sets the granule's allocation tag (below) before any tagged access.
    unsafe {
        super::paging::map_tagged(hhdm);
        super::paging::install_guard_page(ident);
    }

    let owner = with_tag(hhdm, 1); // the owning domain's pointer (logical tag 1)
    let foreign = with_tag(hhdm, 2); // another domain's pointer (logical tag 2) — must trap

    // SAFETY: STG sets the victim granule's allocation tag to the owner pointer's tag (1); the
    // barrier orders it before the tagged store. The store then matches (pointer tag 1 == alloc
    // tag 1) and writes the sentinel. `owner` addresses a granule-aligned page we just tagged.
    unsafe {
        asm!(
            ".arch_extension memtag", // let the assembler accept STG on this soft-float target
            "stg {p}, [{p}]",
            "dsb ish",
            "isb",
            p = in(reg) owner,
            options(nostack, preserves_flags),
        );
        (owner as *mut u64).write_volatile(SENTINEL);
    }

    kprintln!("[praesidium] isolation: entered MTE domain (aarch64 FEAT_MTE, DEC-0008-3); granule tagged 1; probing foreign-tag HHDM read + the (unmapped) identity alias");
    // The mis-tagged HHDM read must take a tag-check fault; the identity alias must be gone entirely.
    let c_mte = super::interrupts::contains_raw_read(foreign);
    let c_ident = super::interrupts::contains_raw_read(ident);

    // SAFETY: read back through the owning (tag-1) pointer; the sentinel must have survived — the
    // contained mis-tagged read must not have corrupted the granule.
    let survived = unsafe { (owner as *const u64).read_volatile() } == SENTINEL;
    if !survived {
        kprintln!(
            "[praesidium] FATAL: isolation: MTE victim data did not survive the escape probe"
        );
        crate::arch::halt();
    }
    c_mte && c_ident
}
