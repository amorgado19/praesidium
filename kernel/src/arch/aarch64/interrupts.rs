//! aarch64 exception handling (ADR-0003 P3b): the EL1 vector table, GICv2 setup, and the IRQ
//! path that drives preemption.
//!
//! `VBAR_EL1` points at a 2 KiB-aligned table of 16 entries (128 bytes each). The kernel runs at
//! EL1 with SP_EL1 (SPx), so real exceptions land in the "current EL, SPx" group; the IRQ stub
//! saves the full integer context (x0–x30 + ELR + SPSR), calls the Rust handler, restores, and
//! `eret`s. Because the whole context is saved, the handler may `context_switch` to another task
//! (its callee-saved registers are pushed on top of this frame); resuming it later returns
//! through the restore + `eret`. Sync faults and any unexpected vector fail closed.

use core::arch::global_asm;
use core::ptr::{read_volatile, write_volatile};
use core::sync::atomic::{AtomicU64, Ordering};

use super::timer;
use crate::sched::scheduler;

// ---- GICv2 (QEMU `virt`) ----------------------------------------------------------------

/// GIC distributor base (QEMU `virt`).
const GICD: usize = 0x0800_0000;
/// GIC CPU-interface base (QEMU `virt`).
const GICC: usize = 0x0801_0000;

const GICD_CTLR: usize = 0x000;
const GICD_ISENABLER: usize = 0x100; // one bit per INTID
const GICC_CTLR: usize = 0x000;
const GICC_PMR: usize = 0x004;
const GICC_IAR: usize = 0x00C;
const GICC_EOIR: usize = 0x010;

/// The EL1 physical timer PPI (PPI 14 → GIC INTID 30 on the `virt` machine).
pub const TIMER_INTID: u32 = 30;
/// GICC_IAR spurious sentinel (no pending interrupt).
const SPURIOUS: u32 = 1023;

fn mmio_write(base: usize, off: usize, val: u32) {
    // SAFETY: `base+off` is a valid GICv2 register on the QEMU `virt` machine; the region is
    // identity-mapped Device memory (paging.rs maps everything below 1 GiB as Device). Volatile.
    unsafe { write_volatile((base + off) as *mut u32, val) };
}

fn mmio_read(base: usize, off: usize) -> u32 {
    // SAFETY: as `mmio_write`, a valid mapped GICv2 register; read-only.
    unsafe { read_volatile((base + off) as *const u32) }
}

/// Enable the GIC and route the timer PPI to this CPU.
fn gic_init() {
    // Enable the distributor and CPU interface; accept every priority.
    mmio_write(GICD, GICD_CTLR, 1);
    mmio_write(GICC, GICC_PMR, 0xFF);
    mmio_write(GICC, GICC_CTLR, 1);
    // Enable the timer INTID (PPI 30 → ISENABLER register 0, bit 30).
    let reg = (TIMER_INTID / 32) as usize * 4;
    mmio_write(GICD, GICD_ISENABLER + reg, 1 << (TIMER_INTID % 32));
}

/// Acknowledge the pending interrupt (read its INTID from GICC_IAR).
fn gic_acknowledge() -> u32 {
    mmio_read(GICC, GICC_IAR) & 0x3FF
}

/// Signal end-of-interrupt for `intid` (write GICC_EOIR). Called before switching tasks.
fn gic_eoi(intid: u32) {
    mmio_write(GICC, GICC_EOIR, intid);
}

// ---- vector table + entry stubs ---------------------------------------------------------

extern "C" {
    /// The 2 KiB-aligned vector table defined in the `global_asm!` below.
    static praesidium_vectors: u8;
}

global_asm!(
    r#"
.section .text
.balign 2048
praesidium_vectors:
    // Current EL, SP0 (kernel uses SPx — these should not fire).
    .balign 0x80
    b {unexpected}
    .balign 0x80
    b {unexpected}
    .balign 0x80
    b {unexpected}
    .balign 0x80
    b {unexpected}
    // Current EL, SPx — the kernel's own exceptions.
    .balign 0x80
    b el1_sync_stub          // 0x200 synchronous
    .balign 0x80
    b el1_irq_stub           // 0x280 IRQ (the timer)
    .balign 0x80
    b {unexpected}           // 0x300 FIQ
    .balign 0x80
    b {unexpected}           // 0x380 SError
    // Lower EL, aarch64 (no EL0 yet).
    .balign 0x80
    b {unexpected}
    .balign 0x80
    b {unexpected}
    .balign 0x80
    b {unexpected}
    .balign 0x80
    b {unexpected}
    // Lower EL, aarch32.
    .balign 0x80
    b {unexpected}
    .balign 0x80
    b {unexpected}
    .balign 0x80
    b {unexpected}
    .balign 0x80
    b {unexpected}

// Save x0-x30 + ELR_EL1 + SPSR_EL1 (272 bytes, 16-aligned), run a handler, restore, eret.
el1_irq_stub:
    sub sp, sp, #272
    stp x0, x1, [sp, #16*0]
    stp x2, x3, [sp, #16*1]
    stp x4, x5, [sp, #16*2]
    stp x6, x7, [sp, #16*3]
    stp x8, x9, [sp, #16*4]
    stp x10, x11, [sp, #16*5]
    stp x12, x13, [sp, #16*6]
    stp x14, x15, [sp, #16*7]
    stp x16, x17, [sp, #16*8]
    stp x18, x19, [sp, #16*9]
    stp x20, x21, [sp, #16*10]
    stp x22, x23, [sp, #16*11]
    stp x24, x25, [sp, #16*12]
    stp x26, x27, [sp, #16*13]
    stp x28, x29, [sp, #16*14]
    mrs x0, elr_el1
    mrs x1, spsr_el1
    stp x30, x0, [sp, #16*15]
    str x1, [sp, #16*16]
    bl {irq_handler}
    b restore_and_eret

el1_sync_stub:
    sub sp, sp, #272
    stp x0, x1, [sp, #16*0]
    stp x2, x3, [sp, #16*1]
    stp x4, x5, [sp, #16*2]
    stp x6, x7, [sp, #16*3]
    stp x8, x9, [sp, #16*4]
    stp x10, x11, [sp, #16*5]
    stp x12, x13, [sp, #16*6]
    stp x14, x15, [sp, #16*7]
    stp x16, x17, [sp, #16*8]
    stp x18, x19, [sp, #16*9]
    stp x20, x21, [sp, #16*10]
    stp x22, x23, [sp, #16*11]
    stp x24, x25, [sp, #16*12]
    stp x26, x27, [sp, #16*13]
    stp x28, x29, [sp, #16*14]
    mrs x0, elr_el1
    mrs x1, spsr_el1
    stp x30, x0, [sp, #16*15]
    str x1, [sp, #16*16]
    mov x0, sp               // pass the saved frame so the handler can redirect the saved ELR
    bl {sync_handler}
    b restore_and_eret       // the handler may return (recoverable path); resume via the epilogue

restore_and_eret:
    ldr x1, [sp, #16*16]
    ldp x30, x0, [sp, #16*15]
    msr elr_el1, x0
    msr spsr_el1, x1
    ldp x28, x29, [sp, #16*14]
    ldp x26, x27, [sp, #16*13]
    ldp x24, x25, [sp, #16*12]
    ldp x22, x23, [sp, #16*11]
    ldp x20, x21, [sp, #16*10]
    ldp x18, x19, [sp, #16*9]
    ldp x16, x17, [sp, #16*8]
    ldp x14, x15, [sp, #16*7]
    ldp x12, x13, [sp, #16*6]
    ldp x10, x11, [sp, #16*5]
    ldp x8, x9, [sp, #16*4]
    ldp x6, x7, [sp, #16*3]
    ldp x4, x5, [sp, #16*2]
    ldp x2, x3, [sp, #16*1]
    ldp x0, x1, [sp, #16*0]
    add sp, sp, #272
    eret
"#,
    unexpected = sym unexpected_exception,
    irq_handler = sym irq_handler,
    sync_handler = sym sync_handler,
);

/// Install the vector table and bring up the GIC. Interrupts stay masked until the scheduler
/// enables them per task.
pub fn interrupts_init() {
    let vbar = core::ptr::addr_of!(praesidium_vectors) as u64;
    // SAFETY: `praesidium_vectors` is the 2 KiB-aligned table above; writing VBAR_EL1 just
    // records where the CPU vectors exceptions. No exception is pending during the write.
    unsafe {
        core::arch::asm!("msr vbar_el1, {}", "isb", in(reg) vbar, options(nostack, preserves_flags))
    };
    gic_init();
    // Ensure the GIC configuration (Device MMIO writes) has completed before interrupts are ever
    // unmasked (task_enter, later) — an explicit barrier, never assumed (DEC-0007-4).
    crate::arch::memory_barrier();
}

/// The Rust IRQ handler: acknowledge, and if it's the timer, re-arm it, EOI (before any switch,
/// so the next tick can fire), then charge/​preempt.
extern "C" fn irq_handler() {
    let intid = gic_acknowledge();
    if intid == TIMER_INTID {
        timer::rearm();
        gic_eoi(intid);
        if scheduler::on_tick() {
            scheduler::preempt();
        }
    } else if intid < SPURIOUS {
        gic_eoi(intid); // acknowledge and ignore anything unexpected
    }
}

/// Synchronous EL1 exception. Reached via `el1_sync_stub`, which passes `frame` = the 34-slot
/// saved integer context (slot 31 = the ELR the epilogue will restore). A fault at exactly the
/// armed single-shot probe address is the deliberate isolation-escape access (an MTE tag-check
/// fault) — contain it and resume at the probe's recovery label; any other sync exception is a
/// real kernel bug and fails closed (loud, then halt).
extern "C" fn sync_handler(frame: *mut u64) {
    let (esr, elr, far): (u64, u64, u64);
    // SAFETY: reading ESR/ELR/FAR_EL1 is side-effect-free; they describe the fault.
    unsafe {
        core::arch::asm!(
            "mrs {e}, esr_el1", "mrs {l}, elr_el1", "mrs {f}, far_el1",
            e = out(reg) esr, l = out(reg) elr, f = out(reg) far,
            options(nomem, nostack, preserves_flags),
        );
    }
    if let Some(resume) = expected_fault_resume(far) {
        kprintln!(
            "[praesidium] isolation: CONTAINED raw cross-domain access — sync fault at {far:#x} esr={esr:#x} (aarch64)"
        );
        // SAFETY: `frame` is the saved integer frame; slot 31 holds the ELR the epilogue restores.
        // Rewriting it redirects `eret` to the probe's recovery label (a kernel `.text` address).
        unsafe { frame.add(31).write_volatile(resume) };
        return; // el1_sync_stub → restore_and_eret → eret to `resume`
    }
    kprintln!("[praesidium] FATAL: EL1 sync exception esr={esr:#x} elr={elr:#x} far={far:#x}");
    crate::arch::halt();
}

/// Any vector we do not expect to take (SP0 group, FIQ, SError, lower-EL). Fail closed.
extern "C" fn unexpected_exception() -> ! {
    let esr: u64;
    // SAFETY: reading ESR_EL1 is side-effect-free.
    unsafe {
        core::arch::asm!("mrs {}, esr_el1", out(reg) esr, options(nomem, nostack, preserves_flags))
    };
    kprintln!("[praesidium] FATAL: unexpected aarch64 exception esr={esr:#x}");
    crate::arch::halt();
}

// ---- single-shot recoverable expected-fault (P5b isolation red-team seam) ----------------
//
// The trusted core of the isolation escape test (mirrors the x86 backend). Exactly one deliberate
// faulting access is armed at a time, at one exact address; `sync_handler` contains ONLY a fault
// at that address and resumes at a fixed label, disarming immediately. Anything else fails closed.
// Deliberately NOT a general fixup facility — the smaller the trusted surface, the easier it is to
// prove a real isolation-escape fault can never be silently recovered.

/// Strips the top-byte pointer tag (MTE logical tag lives in bits [59:56]) so the armed address
/// and the hardware-reported `FAR_EL1` compare on the address bits alone, tag-agnostic.
const VA_TAG_MASK: u64 = 0x00ff_ffff_ffff_ffff;

/// The armed expected-fault address (tag-stripped), or 0 when disarmed (single-shot).
static EXPECT_FAULT_ADDR: AtomicU64 = AtomicU64::new(0);
/// The address to resume at when the armed fault fires (the probe's recovery label).
static EXPECT_FAULT_RESUME: AtomicU64 = AtomicU64::new(0);

extern "C" {
    /// Perform a single deliberate byte read of `addr` (in `x0`); returns 1 (via the recovery
    /// label) if the read faulted and was contained, or 0 if it did not fault.
    fn praesidium_a64_probe_read(addr: u64) -> u64;
    /// The recovery label inside `praesidium_a64_probe_read`; the handler resumes here.
    static praesidium_a64_probe_resume: u8;
}

global_asm!(
    r#"
.section .text
.global praesidium_a64_probe_read
.global praesidium_a64_probe_resume
praesidium_a64_probe_read:
    ldrb w1, [x0]        // the deliberate access — faults on a tag mismatch or unmapped page
    mov x0, #0           // no fault: return 0 (containment FAILED — the access succeeded)
    ret
praesidium_a64_probe_resume:
    mov x0, #1           // handler vectored us here after CONTAINED: return 1
    ret
"#
);

/// If a fault at `far` is the armed single-shot probe (compared tag-agnostically), disarm and
/// return its resume label; otherwise `None` (the caller fails closed). Only the sync handler
/// calls this.
fn expected_fault_resume(far: u64) -> Option<u64> {
    let expect = EXPECT_FAULT_ADDR.load(Ordering::Relaxed);
    if expect != 0 && (far & VA_TAG_MASK) == expect {
        EXPECT_FAULT_ADDR.store(0, Ordering::Relaxed); // single-shot: disarm before resuming
        return Some(EXPECT_FAULT_RESUME.load(Ordering::Relaxed));
    }
    None
}

/// Attempt a raw byte read of `addr` that is **expected to fault** (an isolation-escape probe),
/// containing the fault. Returns `true` iff the access faulted and was contained; `false` means
/// the access *succeeded* — isolation did NOT hold. `addr` may carry an MTE tag in its top byte.
///
/// Arms the single-shot continuation, performs the access, and disarms unconditionally.
pub fn contains_raw_read(addr: u64) -> bool {
    EXPECT_FAULT_RESUME.store(
        core::ptr::addr_of!(praesidium_a64_probe_resume) as u64,
        Ordering::Relaxed,
    );
    EXPECT_FAULT_ADDR.store(addr & VA_TAG_MASK, Ordering::Relaxed); // arm last (tag-stripped)
                                                                    // SAFETY: `praesidium_a64_probe_read` does a single byte read of `addr`. We expect it to fault
                                                                    // (tag mismatch / unmapped); the armed handler resumes at the recovery label returning 1. If
                                                                    // `addr` is in fact accessible, no fault occurs and it returns 0 — reported as "not contained".
    let contained = unsafe { praesidium_a64_probe_read(addr) } == 1;
    EXPECT_FAULT_ADDR.store(0, Ordering::Relaxed); // ensure disarmed on the no-fault path
    contained
}
