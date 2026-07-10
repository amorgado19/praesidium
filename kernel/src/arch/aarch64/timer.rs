//! aarch64 EL1 physical timer — the periodic preemption clock (ADR-0003 P3b).
//!
//! Uses the architected generic timer (`CNTP_*_EL0`), whose input frequency is reported by
//! `CNTFRQ_EL0`, so the tick rate is precise (no calibration needed, unlike the x86 LAPIC). The
//! timer fires the PPI wired to GIC INTID 30; the IRQ handler re-arms it each tick by rewriting
//! `CNTP_TVAL_EL0` (which both deasserts the level interrupt and schedules the next deadline).

use core::arch::asm;
use core::sync::atomic::{AtomicU64, Ordering};

/// Countdown reload value (in timer ticks) for one preemption period, set by [`timer_init`].
static INTERVAL: AtomicU64 = AtomicU64::new(0);

/// Start the EL1 physical timer firing at `hz`.
pub fn timer_init(hz: u32) {
    let freq: u64;
    // SAFETY: reading CNTFRQ_EL0 (the timer input frequency) is side-effect-free.
    unsafe { asm!("mrs {}, cntfrq_el0", out(reg) freq, options(nomem, nostack, preserves_flags)) };
    let interval = (freq / u64::from(hz.max(1))).max(1000);
    INTERVAL.store(interval, Ordering::Relaxed);
    // SAFETY: arm the down-counter (CNTP_TVAL_EL0 = interval) and enable it unmasked
    // (CNTP_CTL_EL0 = 1: ENABLE set, IMASK clear). These EL0 timer registers are accessible at
    // EL1 and only affect this CPU's physical timer.
    unsafe {
        asm!("msr cntp_tval_el0, {}", in(reg) interval, options(nomem, nostack, preserves_flags));
        asm!("msr cntp_ctl_el0, {}", in(reg) 1u64, options(nomem, nostack, preserves_flags));
    }
}

/// Re-arm the timer for the next period. Called from the IRQ handler each tick; rewriting
/// `CNTP_TVAL_EL0` deasserts the (level) interrupt and sets the next deadline. The `isb`
/// synchronizes that deassertion before the caller signals EOI to the GIC — otherwise the
/// still-asserted timer line could re-trigger a spurious tick (explicit barrier, DEC-0007-4).
pub fn rearm() {
    let interval = INTERVAL.load(Ordering::Relaxed);
    // SAFETY: rewriting CNTP_TVAL_EL0 re-arms this CPU's physical timer; the `isb` makes the
    // write (and the interrupt deassertion it causes) effective before subsequent instructions.
    unsafe {
        asm!("msr cntp_tval_el0, {}", "isb", in(reg) interval, options(nomem, nostack, preserves_flags))
    };
}
