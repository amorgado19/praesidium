//! P6 boot demo: build a `.pex` in memory, load it, and assert the ADR-0006 gates —
//! `invoke` works + bad-rights refused (AC6.1), a malformed `.pex` is refused (AC6.2), the loader
//! mints exactly the manifest caps + maps W^X segments + binds a `Sched` (AC6.3), and the process
//! begins with exactly its manifest caps — no ambient authority (AC6.4). Any violation fails the
//! boot closed. EL0 dispatch of the loaded process is P7.

use abi::encode::{encode, encoded_len, ManifestSpec, SegmentSpec};
use abi::invoke::{op, Invocation, InvokeError};
use abi::pex::{MANIFEST_ENDPOINT, MANIFEST_FRAME, MANIFEST_SCHED, PERM_R, PERM_W, PERM_X};
use cap_core::{CSpace, CapType, Rights};
use mem::frame::{pfn_to_phys, phys_to_pfn};

use super::{load, occupied_slots, LoadError, LOADER_SLOTS, L_ENDPOINT, PROCESS_SLOTS};
use crate::syscall::invoke;
use crate::{arch, memory};

/// Process virtual addresses for the demo image's two segments — inside the loader's reserved
/// process window `[1 GiB, 2 GiB)` (see `loader::PROC_VA_BASE`).
const CODE_VA: u64 = 0x4000_0000;
const DATA_VA: u64 = 0x4000_1000;
/// A badge the loader stamps on the process's Endpoint (identifies it to a server).
const EP_BADGE: u64 = 0xBADD;

fn fatal(msg: &str) -> ! {
    kprintln!("[praesidium] FATAL: loader: {msg}");
    arch::halt();
}

/// Frame-zeroing hook for the demo CSpaces (RETYPE zeroes objects before they are observable —
/// CAP-MEM-2), through the HHDM.
fn zero_frames(frame: u64, frames: u32) {
    for i in 0..u64::from(frames) {
        memory::zero_frame(pfn_to_phys((frame + i) as u32));
    }
}

/// Build the demo `.pex`: an RX code segment (entry inside it) + an RW data segment, and a manifest
/// of four caps — a Sched budget, a badged Endpoint, a readable Frame to the code segment, and a
/// **rights-less** Frame to the data segment (to prove a bad-rights invoke is refused).
fn build_pex() -> alloc::vec::Vec<u8> {
    let code = [0x90u8; 16];
    let data = [0xA5u8; 16];
    let segs = [
        SegmentSpec {
            vaddr: CODE_VA,
            mem_size: 0x1000,
            perm: PERM_R | PERM_X,
            data: &code,
        },
        SegmentSpec {
            vaddr: DATA_VA,
            mem_size: 0x1000,
            perm: PERM_R | PERM_W,
            data: &data,
        },
    ];
    let man = [
        ManifestSpec {
            cap_type: MANIFEST_SCHED,
            dest_slot: 1,
            rights: Rights::DERIVE.bits(),
            param0: 100,  // budget
            param1: 1000, // period
        },
        ManifestSpec {
            cap_type: MANIFEST_ENDPOINT,
            dest_slot: 2,
            rights: Rights::SEND.bits(),
            param0: EP_BADGE,
            param1: 0,
        },
        ManifestSpec {
            cap_type: MANIFEST_FRAME,
            dest_slot: 3,
            rights: Rights::READ.bits(),
            param0: 0, // segment 0 (code)
            param1: 0,
        },
        ManifestSpec {
            cap_type: MANIFEST_FRAME,
            dest_slot: 4,
            rights: Rights::empty().bits(), // no READ — a rights-less Frame
            param0: 1,                      // segment 1 (data)
            param1: 0,
        },
    ];
    let mut buf = alloc::vec![0u8; encoded_len(&segs, &man).unwrap()];
    encode(arch::PEX_ARCH, CODE_VA, &segs, &man, &mut buf).expect("encode demo pex");
    buf
}

/// Seed a loader authority CSpace: primordial Untyped (buddy-carved), Sched, and an Endpoint.
fn loader_authority() -> CSpace<LOADER_SLOTS> {
    let phys = memory::alloc_frames(6).unwrap_or_else(|| fatal("no frames for loader Untyped"));
    let base = u64::from(phys_to_pfn(phys));
    let mut cs = CSpace::<LOADER_SLOTS>::new(zero_frames);
    cs.set_root_untyped(base, 64);
    cs.set_root_sched(super::L_SCHED, 1000, 1000);
    cs.retype(super::L_UNTYPED, CapType::Endpoint, 1, 1, L_ENDPOINT)
        .unwrap_or_else(|e| fatal_err("retype loader Endpoint", e));
    cs
}

fn fatal_err(op: &str, e: cap_core::CapError) -> ! {
    kprintln!("[praesidium] FATAL: loader: {op}: {e:?}");
    arch::halt();
}

/// Run the P6 loader demo, emitting `PRAESIDIUM-P6-OK` on success.
pub fn run() {
    kprintln!("[praesidium] loader: P6 — capability-invocation ABI + .pex loader (ADR-0006)");

    let image = build_pex();
    let mut loader = loader_authority();
    let mut proc = CSpace::<PROCESS_SLOTS>::new(zero_frames);

    // AC6.3: load — mint exactly the manifest caps (monotonic), map W^X segments, bind Sched.
    let loaded = load(&image, &mut loader, &mut proc, 0x50)
        .unwrap_or_else(|e| fatal_load("valid .pex failed to load", e));
    kprintln!(
        "[praesidium] loader: .pex loaded — entry {:#x}, budget {}, domain {:#x} (AC6.2/6.3)",
        loaded.entry,
        loaded.budget,
        loaded.domain_id
    );

    // AC6.3: W^X segments actually mapped at their vaddrs (code R-X, data RW-NX).
    match arch::page_prot(CODE_VA) {
        Some((false, true)) => {}
        other => fatal_prot("code segment not mapped R-X", other),
    }
    match arch::page_prot(DATA_VA) {
        Some((true, false)) => {}
        other => fatal_prot("data segment not mapped RW-NX", other),
    }
    kprintln!("[praesidium] loader: segments mapped W^X — code R-X, data RW-NX (AC6.3)");

    // AC6.4: the process holds EXACTLY the four manifest caps — no ambient authority.
    if occupied_slots(&proc) != 4 {
        fatal("process CSpace does not hold exactly the manifest caps");
    }
    for (slot, ty) in [
        (1usize, CapType::Sched),
        (2, CapType::Endpoint),
        (3, CapType::Frame),
        (4, CapType::Frame),
    ] {
        match proc.resolve(slot) {
            Ok(c) if c.cap_type == ty => {}
            _ => fatal("process cap slot has the wrong type or is empty"),
        }
    }
    // The budget bound to the process equals the manifest's declaration.
    if loaded.budget != 100 {
        fatal("bound Sched budget does not match the manifest");
    }
    kprintln!("[praesidium] loader: process holds EXACTLY its 4 manifest caps — no ambient authority (AC6.4)");

    // AC6.1: invoke works and a bad-rights invoke is refused.
    // CAP_IDENTIFY resolves a cptr (the Frame at slot 3).
    let frame_objref = proc.resolve(3).unwrap().objref;
    match invoke(&proc, &Invocation::new(3, op::CAP_IDENTIFY)) {
        Ok(id) if (id >> 32) as u8 == CapType::Frame as u8 => {}
        other => fatal_invoke("CAP_IDENTIFY on the Frame cap", other),
    }
    // FRAME_PROBE requires READ — slot 3 has it, slot 4 does not.
    match invoke(&proc, &Invocation::new(3, op::FRAME_PROBE)) {
        Ok(objref) if objref == frame_objref => {}
        other => fatal_invoke("FRAME_PROBE on a readable Frame", other),
    }
    match invoke(&proc, &Invocation::new(4, op::FRAME_PROBE)) {
        Err(InvokeError::InsufficientRights) => {}
        other => fatal_invoke(
            "FRAME_PROBE on a rights-less Frame should be refused",
            other,
        ),
    }
    // ENDPOINT_SEND requires SEND on an Endpoint — the syscall/IPC unification (DEC-0006-3).
    match invoke(&proc, &Invocation::new(2, op::ENDPOINT_SEND)) {
        Ok(badge) if badge == EP_BADGE => {}
        other => fatal_invoke("ENDPOINT_SEND on the Endpoint cap", other),
    }
    // Wrong type / empty slot / out-of-range cptr all fail closed.
    match invoke(&proc, &Invocation::new(3, op::ENDPOINT_SEND)) {
        Err(InvokeError::WrongType) => {}
        other => fatal_invoke("ENDPOINT_SEND on a Frame should be WrongType", other),
    }
    match invoke(&proc, &Invocation::new(0, op::CAP_IDENTIFY)) {
        Err(InvokeError::EmptySlot) => {}
        other => fatal_invoke("invoke on an empty slot should be EmptySlot", other),
    }
    match invoke(&proc, &Invocation::new(999, op::CAP_IDENTIFY)) {
        Err(InvokeError::BadCptr) => {}
        other => fatal_invoke("invoke on an out-of-range cptr should be BadCptr", other),
    }
    kprintln!("[praesidium] loader: invoke resolves cptrs + rights-checked; bad-rights/type/cptr refused (AC6.1)");

    // AC6.2: a malformed .pex is refused, not UB. (The decoder is fuzzed; here we prove the loader
    // fails closed on a corrupt image at boot too.)
    let mut bad = build_pex();
    bad[0] ^= 0xff; // corrupt the magic
    let mut throwaway = CSpace::<PROCESS_SLOTS>::new(zero_frames);
    match load(&bad, &mut loader, &mut throwaway, 0x51) {
        Err(LoadError::Pex(_)) => {}
        _ => fatal("a malformed .pex was not refused"),
    }
    kprintln!("[praesidium] loader: malformed .pex refused, not UB (AC6.2)");

    kprintln!("[praesidium] PRAESIDIUM-P6-OK");
}

fn fatal_load(msg: &str, e: LoadError) -> ! {
    kprintln!("[praesidium] FATAL: loader: {msg}: {e:?}");
    arch::halt();
}
fn fatal_prot(msg: &str, got: Option<(bool, bool)>) -> ! {
    kprintln!("[praesidium] FATAL: loader: {msg} (got {got:?})");
    arch::halt();
}
fn fatal_invoke(msg: &str, got: Result<u64, InvokeError>) -> ! {
    kprintln!("[praesidium] FATAL: loader: {msg} (got {got:?})");
    arch::halt();
}
