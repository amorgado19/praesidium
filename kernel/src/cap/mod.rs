//! Capability-core integration (P2).
//!
//! `cap-core` is the pure, host-tested trust root (the capability machinery); this module
//! wires it to the running kernel: it provides the CAP-MEM-2/CAP-REVOKE-2 zeroing hook
//! (frames zeroed through the HHDM), carves the primordial `Untyped` from P1's frame
//! allocator, and runs the P2 boot demo — RETYPE / MINT / COPY / REVOKE — asserting the
//! gates (AC2.1–AC2.3). No capability is fabricated here: every `Cap` comes by value from
//! cap-core (CAP-RUST-1).

use cap_core::cap::Frame;
use cap_core::{CSpace, CapError, CapType, Rights};
use mem::frame::{pfn_to_phys, phys_to_pfn};

use crate::memory;

/// Number of slots in the P2 demo CSpace (single root CNode).
const SLOTS: usize = 64;

/// Zero `frames` frames starting at frame number `frame`, through the HHDM. This is the
/// object-zeroing hook cap-core calls on RETYPE (before an object is observable, CAP-MEM-2)
/// and on destroying an object's last capability (CAP-REVOKE-2).
fn zero_frames(frame: u64, frames: u32) {
    for i in 0..u64::from(frames) {
        memory::zero_frame(pfn_to_phys((frame + i) as u32));
    }
}

fn fatal(msg: &str) -> ! {
    kprintln!("[praesidium] FATAL: cap: {msg}");
    crate::arch::halt();
}

fn fatal_err(op: &str, e: CapError) -> ! {
    kprintln!("[praesidium] FATAL: cap: {op} failed: {e:?}");
    crate::arch::halt();
}

/// Bootstrap a CSpace over a buddy-allocated `Untyped` region and exercise the capability
/// operations, asserting AC2.1–AC2.3. Any violation fails the boot closed.
pub fn run() {
    // Carve an Untyped region from the frame allocator (2^6 = 64 frames = 256 KiB).
    let phys = memory::alloc_frames(6).unwrap_or_else(|| fatal("no frames for the root Untyped"));
    let base_frame = u64::from(phys_to_pfn(phys));
    let frames: u32 = 64;

    let mut cs = CSpace::<SLOTS>::new(zero_frames);
    cs.set_root_untyped(base_frame, frames);
    kprintln!("[praesidium] cap: root Untyped = {frames} frames @ frame {base_frame:#x}");

    // AC2.1 — RETYPE Untyped -> 4 Frames into slots 1..4, charged to budget, zeroed.
    cs.retype(0, CapType::Frame, 1, 4, 1)
        .unwrap_or_else(|e| fatal_err("retype", e));
    let f1 = cs
        .get::<Frame>(1)
        .unwrap_or_else(|e| fatal_err("get frame 1", e));
    kprintln!(
        "[praesidium] cap: RETYPE 4 Frames -> slots 1..4; budget {}/{frames} used; frame objref {:#x} (AC2.1)",
        cs.resolve(0).unwrap_or_else(|e| fatal_err("resolve untyped", e)).watermark,
        f1.objref()
    );

    // AC2.2 — MINT narrows (read-only, badged), widening refused; COPY keeps rights.
    cs.mint(1, 5, Rights::READ, 0xF00D)
        .unwrap_or_else(|e| fatal_err("mint", e));
    if cs.mint(5, 6, Rights::READ | Rights::WRITE, 0) != Err(CapError::InsufficientRights) {
        fatal("rights-widening was NOT refused");
    }
    cs.copy(2, 7).unwrap_or_else(|e| fatal_err("copy", e));
    let m = cs
        .get::<Frame>(5)
        .unwrap_or_else(|e| fatal_err("get minted", e));
    kprintln!(
        "[praesidium] cap: MINT read-only (badge {:#x}) ok; widening refused; COPY ok (AC2.2)",
        m.badge()
    );

    // AC2.3 — REVOKE the Untyped destroys ALL descendants; a revoked cptr fails cleanly.
    cs.revoke(0).unwrap_or_else(|e| fatal_err("revoke", e));
    let survivors = (1..SLOTS).filter(|&s| cs.resolve(s).is_ok()).count();
    if survivors != 0 {
        fatal("REVOKE left descendants alive");
    }
    let revoked = cs
        .get::<Frame>(1)
        .err()
        .unwrap_or_else(|| fatal("revoked cap still resolves"));
    if revoked != CapError::EmptySlot {
        fatal("revoked cap did not fail cleanly");
    }
    kprintln!(
        "[praesidium] cap: REVOKE destroyed all descendants; revoked cptr -> {:?}; budget reclaimed to {} (AC2.3)",
        revoked,
        cs.resolve(0).unwrap_or_else(|e| fatal_err("resolve untyped", e)).watermark
    );

    memory::free_frames(phys);
    kprintln!("[praesidium] PRAESIDIUM-P2-OK");
}
