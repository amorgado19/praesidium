//! The in-kernel `.pex` loader (ADR-0006 P6). Reads a `.pex` image (HOSTILE input — parsed by the
//! fuzzed [`abi::pex`] decoder), lays out its W^X segments, and builds the new process's CSpace
//! containing **exactly** the capabilities its manifest declares — no more (there is no ambient
//! authority to inherit), no less. Every process capability is *derived monotonically from the
//! loader's own authority* (`cap-core` RETYPE/SPLIT/GRANT): a loader can never grant what it does
//! not hold, and the rights it grants can only narrow. No capability is fabricated here — the
//! loader holds primordial authority and delegates a subset (CAP-RUST-1).
//!
//! **Scope (Fork-1 ruling): in-kernel only.** The image is loaded, its segments mapped W^X, and
//! its CSpace populated + a `Sched` budget bound; the process is left *ready to run*. The actual
//! EL0/ring-3 dispatch of `entry` — and the syscall trap that carries an [`abi::invoke`]
//! invocation into [`crate::syscall::invoke`] — is P7. The isolation *domain* is assigned here
//! (a `domain_id` recorded on the process) but *enforced* at execution (P7, ADR-0008).

use abi::pex::{
    ManifestEntry, Pex, PexError, MANIFEST_ENDPOINT, MANIFEST_FRAME, MANIFEST_SCHED, MAX_SEGMENTS,
    PERM_W, PERM_X,
};
use cap_core::{grant, CSpace, CapError, CapType, Cptr, GrantMode, Rights};
use mem::frame::pfn_to_phys;

use crate::arch::{self, Prot};
use crate::memory;

/// Slots in the loader's authority CSpace.
pub const LOADER_SLOTS: usize = 32;
/// Slots in a loaded process's CSpace.
pub const PROCESS_SLOTS: usize = 16;

/// The loader's fixed authority layout: primordial Untyped, Sched, and an Endpoint to hand out.
const L_UNTYPED: Cptr = 0;
const L_SCHED: Cptr = 1;
const L_ENDPOINT: Cptr = 2;
/// Scratch slots (retyped segment frames + split Sched children) start here.
const L_SCRATCH: Cptr = 8;

/// The reserved virtual-address window a loaded process's segments must fall within
/// (`[1 GiB, 2 GiB)`). A process vaddr is attacker-controlled hostile input; confining it here is
/// what keeps [`arch::map_page`] from ever touching a live kernel mapping — the HHDM and the kernel
/// image live in the high half, and the aarch64 MMIO identity region is below 1 GiB, so this window
/// is disjoint from all of them while still lying inside the always-present identity map (so the
/// map never faults a missing table). It shadows only unused low identity-alias VAs of that range,
/// which nothing accesses (the kernel reaches memory through the HHDM). P7's per-process address
/// spaces supersede this single shared window.
const PROC_VA_BASE: u64 = 0x4000_0000;
const PROC_VA_END: u64 = 0x8000_0000;

/// Why a `.pex` failed to load. Every variant fails the load closed — a malformed or over-reaching
/// image is refused, never partially applied with UB (AC6.2).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LoadError {
    /// The `.pex` itself is malformed (from the decoder).
    Pex(PexError),
    /// A manifest entry named a capability type the loader cannot satisfy.
    UnknownManifestType(u8),
    /// A manifest entry requested rights with bits the model does not define.
    BadRights,
    /// A `FRAME` manifest entry referenced a segment index that does not exist.
    BadSegmentRef,
    /// A segment's `[vaddr, vaddr+mem_size)` escapes the reserved process VA window — refused so
    /// the loader never maps over a live kernel/HHDM/MMIO mapping.
    SegmentVaddrOutOfRange,
    /// A manifest `dest_slot` is outside the process CSpace.
    DestSlotOutOfRange,
    /// A `cap-core` operation refused the derivation (e.g. rights not a subset — monotonicity).
    Cap(CapError),
}

impl From<PexError> for LoadError {
    fn from(e: PexError) -> Self {
        LoadError::Pex(e)
    }
}
impl From<CapError> for LoadError {
    fn from(e: CapError) -> Self {
        LoadError::Cap(e)
    }
}

/// A loaded process, ready to run (EL0 dispatch is P7).
pub struct Loaded {
    /// The entry-point virtual address (in an executable segment).
    pub entry: u64,
    /// The CPU-time budget bound to the process (split from the loader's `Sched`).
    pub budget: u32,
    /// The isolation domain assigned to the process (enforced at execution, P7).
    pub domain_id: u64,
}

/// Map a validated `.pex` permission byte to a W^X protection. The decoder already guaranteed the
/// combination is valid (readable, not W+X), so this is total.
fn perm_to_prot(perm: u8) -> Prot {
    if perm & PERM_X != 0 {
        Prot::Rx
    } else if perm & PERM_W != 0 {
        Prot::Rw
    } else {
        Prot::Ro
    }
}

/// Load `image` into a fresh process. `loader` holds the loader's authority (Untyped/Sched/
/// Endpoint at the fixed slots above); `proc` is the process's initially-empty CSpace. On success
/// `proc` holds exactly the manifest's capabilities and `Loaded` describes the process.
pub fn load(
    image: &[u8],
    loader: &mut CSpace<LOADER_SLOTS>,
    proc: &mut CSpace<PROCESS_SLOTS>,
    domain_id: u64,
) -> Result<Loaded, LoadError> {
    let pex = Pex::parse(image, arch::PEX_ARCH)?;
    let mut scratch = L_SCRATCH;

    // --- Segments: retype owned frames, copy the file bytes, cache-maintain code, map W^X. ---
    let mut seg_frames: [Option<Cptr>; MAX_SEGMENTS] = [None; MAX_SEGMENTS];
    for i in 0..pex.segment_count() {
        let s = pex.segment(i);

        // Confine the (attacker-controlled) vaddr to the reserved process window, overflow-checked,
        // so a hostile `.pex` can never make `map_page` touch a live kernel/HHDM/MMIO mapping.
        let seg_end = s
            .vaddr
            .checked_add(s.mem_size)
            .ok_or(LoadError::SegmentVaddrOutOfRange)?;
        if s.vaddr < PROC_VA_BASE || seg_end > PROC_VA_END {
            return Err(LoadError::SegmentVaddrOutOfRange);
        }
        // `mem_size` is 4 KiB-aligned and bounded by MAX_SEGMENT_PAGES (parser-guaranteed), so this
        // conversion cannot truncate; keep it checked so the allocation size can never silently
        // shrink below the copy length (the OOB-write footgun).
        let pages = u32::try_from(s.mem_size >> 12)
            .map_err(|_| LoadError::Pex(PexError::SegmentTooLarge))?;
        let mapped_bytes = (pages as usize) << 12;

        let frame_slot = scratch;
        scratch += 1;
        // One Frame object of `pages` contiguous frames, carved + zeroed from the loader's Untyped
        // (CAP-MEM-2 zero-on-retype gives a clean .bss tail for free).
        loader.retype(L_UNTYPED, CapType::Frame, pages, 1, frame_slot)?;
        let base_pfn = loader.resolve(frame_slot)?.objref as u32;
        let hhdm = memory::phys_to_virt(pfn_to_phys(base_pfn));

        // Copy the file bytes through the (writable) HHDM alias. The frames are physically
        // contiguous and the HHDM is linear, so the whole segment is one contiguous copy. The
        // decoder guarantees `file_size <= mem_size == mapped_bytes`, so this stays in-bounds.
        let data = pex.segment_data(i);
        debug_assert!(
            data.len() <= mapped_bytes,
            "copy would overrun the retyped frames"
        );
        // SAFETY: `[hhdm, hhdm+data.len())` lies within `mapped_bytes` (`file_size <= mem_size`) of
        // freshly-retyped, HHDM-mapped, writable frames the loader owns; `data` is a disjoint
        // borrow of the image buffer.
        unsafe {
            core::ptr::copy_nonoverlapping(data.as_ptr(), hhdm as *mut u8, data.len());
        }
        // Executable segment: make the ENTIRE mapped extent coherent for instruction fetch — not
        // just the copied file bytes: the zeroed `.bss` tail is executable too (the entry may fall
        // in it), so it must be cleaned to PoU + I-cache-invalidated as well (load-bearing on
        // aarch64; a no-op-with-fence on x86). Done before the pages are mapped executable.
        if s.perm & PERM_X != 0 {
            arch::sync_instruction_cache(hhdm, mapped_bytes);
        }

        let prot = perm_to_prot(s.perm);
        for k in 0..u64::from(pages) {
            // SAFETY: map owned, in-range frames at the process's declared vaddr (confined to the
            // reserved window above) with W^X `prot`; this shadows only an unused identity-alias VA.
            unsafe {
                arch::map_page(s.vaddr + k * 0x1000, pfn_to_phys(base_pfn + k as u32), prot);
            }
        }
        seg_frames[i] = Some(frame_slot);
    }

    // --- Manifest: derive EXACTLY the declared caps into the process, monotonically. ---
    let mut budget = 0u32;
    for i in 0..pex.manifest_count() {
        let m: ManifestEntry = pex.manifest(i);
        let dest = m.dest_slot as Cptr;
        if dest >= PROCESS_SLOTS {
            return Err(LoadError::DestSlotOutOfRange);
        }
        // Reject rights with bits outside the model (hostile-manifest hardening); grant then
        // enforces the subset check against the loader's own authority (monotonic).
        let rights = Rights::from_bits(m.rights).ok_or(LoadError::BadRights)?;

        match m.cap_type {
            MANIFEST_SCHED => {
                // SPLIT the loader's Sched (debits the loader — monotonic), then MOVE the child
                // into the process. Sched is non-duplicable: it transfers, never forks.
                let child = scratch;
                scratch += 1;
                loader.split(L_SCHED, child, m.param0 as u32)?;
                grant(loader, child, proc, dest, rights, 0, GrantMode::Move)?;
                budget = m.param0 as u32;
            }
            MANIFEST_ENDPOINT => {
                // MINT a badged Endpoint from the loader's Endpoint authority (rights ⊆ loader's).
                grant(
                    loader,
                    L_ENDPOINT,
                    proc,
                    dest,
                    rights,
                    m.param0,
                    GrantMode::Mint,
                )?;
            }
            MANIFEST_FRAME => {
                let seg = m.param0 as usize;
                let src = seg_frames
                    .get(seg)
                    .copied()
                    .flatten()
                    .ok_or(LoadError::BadSegmentRef)?;
                grant(loader, src, proc, dest, rights, 0, GrantMode::Mint)?;
            }
            other => return Err(LoadError::UnknownManifestType(other)),
        }
    }

    Ok(Loaded {
        entry: pex.entry(),
        budget,
        domain_id,
    })
}

/// The number of occupied slots in a process CSpace (for the AC6.4 "exactly the manifest caps"
/// check — no ambient authority crept in).
pub fn occupied_slots(proc: &CSpace<PROCESS_SLOTS>) -> usize {
    (0..PROCESS_SLOTS)
        .filter(|&s| proc.resolve(s).is_ok())
        .count()
}

mod demo;
pub use demo::run;
