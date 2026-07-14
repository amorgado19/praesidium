//! refproc `evil` (P7b-ii) — the HOSTILE reference process for the AC7.3 isolation red-team. Unlike
//! `ping`/`pong` (well-behaved Rust components), `evil` is a real NATIVE binary that deliberately
//! attempts to read another process's memory with a RAW POINTER — holding no capability to it, only
//! a hardcoded virtual address. In a naive single-address-space OS this succeeds (one page table,
//! every mapped byte reachable); Praesidium's CAP-MEM-3 backstop (x86 PKU / aarch64 MTE) must make
//! it FAULT, so the kernel contains the breach by killing `evil` while the victim + kernel survive.
//! This is the existential proof the whole SASOS thesis rests on — distinct from P5b's *armed*
//! in-kernel proof: here the adversary is a hostile native binary, exactly as the threat model says.
#![no_std]
#![no_main]

/// The victim virtual address `evil` raw-reads: `pong`'s segment base (the loader maps `pong` here,
/// in `pong`'s distinct isolation domain). Must equal xtask's `--defsym __base` for `pong`
/// (`0x40300000`). `evil` links at a different base, so this is unambiguously *another* domain.
const VICTIM_VA: usize = 0x4030_0000;

/// Process entry — dropped to at EL0/ring-3 with GPRs zeroed. `evil` forms a raw pointer to the
/// victim's page (no capability, no grant) and reads it. The hardware isolation backstop must trap
/// the read; the kernel then kills `evil`. If the read RETURNS, isolation has FAILED — `evil`
/// reports the stolen word loudly and exits with a breach code so the smoke fails attributably.
#[no_mangle]
pub extern "C" fn _start() -> ! {
    // SAFETY: this is DELIBERATELY an isolation violation — the entire purpose of the red-team. The
    // read MUST be contained by PKU (x86) / MTE (aarch64); reaching the next line means it was not.
    let stolen = unsafe { core::ptr::read_volatile(VICTIM_VA as *const u64) };
    // Only reached on a BREACH (the raw cross-domain read was not contained). Report the stolen word
    // so the failure is loud + attributable, then exit with a distinctive breach code.
    refproc::debug(stolen);
    refproc::exit(0xB4EAC4) // "breach" — a non-killed exit; the kernel asserts evil was KILLED instead
}
