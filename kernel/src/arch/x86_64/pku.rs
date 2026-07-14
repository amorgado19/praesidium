//! x86-64 Protection Keys for Userspace (PKU/PKRU) — the P7b-ii Layer-2 hardware isolation
//! mechanism for process-vs-process isolation within the single address space (ADR-0008
//! DEC-0008-2). This is the **real** intra-address-space domain enforcement on x86: every
//! user (U/S=1) page carries a 4-bit *protection key* in PTE bits [62:59], and the per-thread
//! `PKRU` register gates access per key (2 bits each: Access-Disable + Write-Disable). A domain
//! switch is a single `WRPKRU` — **no TLB flush, no CR3 reload** (the SASOS win) — so two
//! processes share one address space yet a raw pointer into another domain's page takes a
//! `#PF` (error-code bit 5, PK) the kernel contains by killing the offending process.
//!
//! Unlike PKS (supervisor keys, absent on this host), PKRU gates only U=1 pages, so the kernel's
//! own supervisor/HHDM mappings are untouched — the kernel reaches process frames through the
//! U=0 HHDM alias, never the U=1 process VA, so `PKRU` never gates a kernel access.
//!
//! **Availability is probed, never assumed** (the anti-theater rule): [`init`] checks
//! `CPUID.(EAX=7,ECX=0):ECX.PKU` and sets `CR4.PKE`; only if that holds is PKU the enforcing
//! mechanism. On a CPU without PKU the per-domain-page-table fallback (DEC-0008-6) is the honest
//! Layer-2 mechanism instead — [`available`] reports which held. The isolation smoke pins QEMU to
//! `accel=tcg -cpu max`, which emulates PKU, so the primary is exercised deterministically.

use core::sync::atomic::{AtomicBool, Ordering};

/// Whether [`init`] confirmed PKU is present and enabled (`CR4.PKE` set). When false, the
/// per-domain-page-table fallback is the enforcing mechanism (isolation still holds; a different
/// mechanism does the work — logged honestly per ISO-AC4).
static PKU_ENABLED: AtomicBool = AtomicBool::new(false);

/// PTE bit position of the 4-bit protection key (bits [62:59]).
const PKEY_SHIFT: u64 = 59;

/// `CR4.PKE` — enable protection keys for user pages (bit 22).
const CR4_PKE: u64 = 1 << 22;

/// Report whether PKU is the enforcing isolation mechanism (probed + enabled by [`init`]).
#[must_use]
pub fn available() -> bool {
    PKU_ENABLED.load(Ordering::Relaxed)
}

/// The PTE bits carrying protection key `pkey` (low 4 bits used). OR into a user leaf entry; a
/// `pkey` of 0 (the kernel/default domain) contributes nothing, so untagged pages stay key 0.
#[must_use]
pub fn pkey_bits(pkey: u64) -> u64 {
    (pkey & 0xf) << PKEY_SHIFT
}

/// Probe PKU (`CPUID.(EAX=7,ECX=0):ECX[3]`) and, if present, enable it (`CR4.PKE`). Idempotent.
/// Returns whether PKU is now the enforcing mechanism. Called once at [`super::user::gdt_init`]
/// time (before any userspace), so a process's first switch-in can program `PKRU`.
pub fn init() -> bool {
    let ecx: u32;
    // SAFETY: CPUID leaf 7 sub-leaf 0 exists on every PKU-capable CPU (and returns 0 for the PKU
    // bit otherwise — fail closed). `cpuid` clobbers rbx (LLVM-reserved), so preserve it; the
    // push/pop is the only stack use (no `nostack`); no flags change.
    unsafe {
        core::arch::asm!(
            "push rbx",
            "mov eax, 7",
            "mov ecx, 0", // sub-leaf 0 (mov, not xor, so flags are preserved)
            "cpuid",
            "pop rbx",
            out("eax") _,
            out("ecx") ecx,
            out("edx") _,
            options(preserves_flags),
        );
    }
    let supported = ecx & (1 << 3) != 0; // ECX.PKU
    if !supported {
        PKU_ENABLED.store(false, Ordering::Relaxed);
        return false;
    }
    // SAFETY: set CR4.PKE (bit 22) to enable protection keys for user pages. Read-modify-write so
    // no other CR4 bit changes; `mov cr4` has no memory effects the compiler must order here.
    unsafe {
        core::arch::asm!(
            "mov {t}, cr4",
            "or {t}, {pke}", // `or` clobbers flags, so no `preserves_flags` here
            "mov cr4, {t}",
            t = out(reg) _,
            pke = const CR4_PKE,
            options(nostack),
        );
    }
    PKU_ENABLED.store(true, Ordering::Relaxed);
    true
}

/// Write `PKRU` so **only** key `allow` is accessible and every other key is Access-Disabled
/// (AD) — the domain mask for a process holding protection key `allow`. `PKRU` is 2 bits per key
/// (`bit 2k` = AD, `bit 2k+1` = WD); we set AD for all keys, then clear AD (and WD) for `allow`.
/// Key 0 (the kernel/default domain) is included in the disable when `allow != 0`, since a
/// process's pages are all tagged with its own key — it needs no key-0 user page.
fn write_pkru_allow_only(allow: u64) {
    // Access-Disable every key, then re-enable `allow` (clear its AD+WD pair).
    let mut pkru: u32 = 0x5555_5555; // AD=1 for all 16 keys, WD=0
    let k = (allow & 0xf) as u32;
    pkru &= !(0b01 << (2 * k)); // clear AD for the allowed key (WD already 0)
    write_pkru(pkru);
}

/// Set `PKRU` to `value` via `WRPKRU` (requires `CR4.PKE`; `ECX=EDX=0`).
fn write_pkru(value: u32) {
    // SAFETY: `WRPKRU` loads PKRU from EAX with ECX=EDX=0 (its required operand form). PKU is
    // enabled (callers gate on [`available`]); the instruction has no memory effects but does
    // change the current thread's data-access permission on user pages — ordered before the
    // subsequent ring-3 entry / syscall return by the privilege transition that follows.
    unsafe {
        core::arch::asm!(
            "mov ecx, 0", // WRPKRU requires ECX=EDX=0 (mov, not xor, so flags are preserved)
            "mov edx, 0",
            "wrpkru",
            in("eax") value,
            out("ecx") _,
            out("edx") _,
            options(nostack, preserves_flags),
        );
    }
}

/// Program `PKRU` for the task the scheduler is switching in: a userspace process → only its
/// protection key is accessible; a kernel task (`None`) → key 0 only (its natural key; the kernel
/// touches no foreign-keyed user page). A no-op if PKU is not the enforcing mechanism (the
/// fallback does the isolation via per-domain page tables instead). Called on every switch-in
/// (preemption-masked, single CPU), so a process resuming a syscall lands with its own domain mask.
pub fn set_domain(pkey: Option<u64>) {
    if !available() {
        return;
    }
    match pkey {
        Some(k) => write_pkru_allow_only(k),
        None => write_pkru(0), // kernel: all keys allowed (kernel accesses no U=1 foreign page)
    }
}
