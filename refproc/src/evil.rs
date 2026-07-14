//! refproc `evil` (P7b-ii) — the HOSTILE reference process for the AC7.3 isolation red-team. Unlike
//! `ping`/`pong` (well-behaved Rust components), `evil` is a real NATIVE binary that deliberately
//! attempts to read another process's memory. The threat model (ADR-0008 R1) is a hostile native
//! binary, so `evil` does NOT play nice: it actively tries to DEFEAT the hardware domain before the
//! read — x86: rewrite PKRU via the UNPRIVILEGED `WRPKRU` to unlock all protection keys; aarch64:
//! forge the victim's MTE tag into the pointer (the tag is 4 bits the process itself controls). A
//! sound isolation backstop must contain the read anyway (a mechanism the process cannot defeat); if
//! the read RETURNS, isolation FAILED and `evil` reports the breach loudly.
//!
//! v1.1: `evil` is also a legitimate holder of a `SharedRo` capability (a read-only window into the
//! shared transfer region). It READS the region (allowed — that is the point of a RO window) and
//! then tries to use that foothold to reach the owner's OTHER memory (ping's segment, NOT part of
//! the region). Per-domain page tables map ONLY the region into evil's table, so the out-of-region
//! read faults regardless of WRPKRU/tag-forge — proving a shared window cannot be turned into a
//! general read of the owner.
#![no_std]
#![no_main]

/// The victim virtual address `evil` raw-reads: `ping`'s segment base (`ping`'s linker default
/// `__base = 0x40100000`). This VA is in the [1 GiB, 1 GiB+2 MiB) block the P6 loader demo splits on
/// the kernel base, so it is the case a shared-leaf-table bug in new_process_space would breach —
/// per-domain tables must isolate it too (not just the un-split blocks pong/evil live in).
const VICTIM_VA: usize = 0x4010_0000;
/// `ping`'s isolation domain (the loader assigns ping=1, pong=2, evil=3 — a predictable assignment
/// an attacker knows or brute-forces over 4 bits). Used to forge ping's MTE tag on aarch64.
#[cfg(target_arch = "aarch64")]
const VICTIM_DOMAIN: u64 = 1;

/// Process entry — dropped to at EL0/ring-3 with GPRs zeroed. `evil` attempts to defeat the hardware
/// isolation domain, then reads the victim's page. The backstop must trap the read regardless; the
/// kernel then kills `evil`. If the read RETURNS, isolation has FAILED — report the stolen word.
#[no_mangle]
pub extern "C" fn _start() -> ! {
    // v1.1: evil legitimately holds a SharedRo cap — it may READ the shared window (intended). Learn
    // the window VA from the cap (RI) and read it; this must SUCCEED (a RO holder reads the region).
    let region_va = refproc::shared_region();
    let shared = refproc::shared_read(region_va);
    refproc::debug(shared); // proves evil read the shared region (== SHARED_SENTINEL) — that's allowed

    // Active bypass: unlock all PKU protection keys. WRPKRU is UNPRIVILEGED — usable from ring 3 —
    // so a hostile process can clear its own PKRU and defeat protection-key isolation entirely.
    #[cfg(target_arch = "x86_64")]
    // SAFETY: WRPKRU writes PKRU from EAX with ECX=EDX=0; EAX=0 clears every access/write-disable
    // bit, unlocking all 16 keys. Deliberately defeats PKU — the point of the red-team.
    unsafe {
        core::arch::asm!(
            "mov ecx, 0",
            "mov edx, 0",
            "mov eax, 0",
            "wrpkru",
            out("eax") _,
            out("ecx") _,
            out("edx") _,
            options(nostack),
        );
    }

    // The pointer to the victim. aarch64: forge the victim's MTE tag into the top byte (the process
    // controls pointer tags) so a tag check would MATCH. x86: a plain pointer (PKRU already cleared).
    #[cfg(target_arch = "aarch64")]
    let victim = (VICTIM_VA as u64 | (VICTIM_DOMAIN << 56)) as *const u64;
    #[cfg(target_arch = "x86_64")]
    let victim = VICTIM_VA as *const u64;

    // SAFETY: the deliberate cross-domain read after actively defeating the hardware domain. A sound
    // backstop (one the process cannot defeat) must trap this; reaching the next line is a breach.
    let stolen = unsafe { core::ptr::read_volatile(victim) };
    // Only reached on a BREACH. Report the stolen word loudly, then exit with a distinctive code.
    refproc::debug(stolen);
    refproc::exit(0xB4EAC4) // "breach" — a non-killed exit; the kernel asserts evil was KILLED
}
