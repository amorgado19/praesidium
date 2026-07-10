//! x86-64 LAPIC timer — the periodic preemption clock (ADR-0003 P3b).
//!
//! Uses the Local APIC in xAPIC (MMIO) mode: enable the APIC, mask the legacy 8259 PICs so only
//! the LAPIC timer reaches the CPU, and run the timer in periodic mode on [`TIMER_VECTOR`]. The
//! initial count is a nominal (uncalibrated) value — precise timekeeping needs calibration
//! against a reference clock (a later phase); for preemption we only need it to fire steadily.

use core::arch::asm;
use core::ptr::write_volatile;

use super::interrupts::TIMER_VECTOR;
use crate::memory::phys_to_virt;

const APIC_BASE_MSR: u32 = 0x1B;
const MSR_APIC_GLOBAL_ENABLE: u64 = 1 << 11;

// LAPIC register offsets.
const REG_EOI: usize = 0xB0;
const REG_SVR: usize = 0xF0;
const REG_TPR: usize = 0x80;
const REG_LVT_TIMER: usize = 0x320;
const REG_TIMER_INITIAL: usize = 0x380;
const REG_TIMER_DIVIDE: usize = 0x3E0;

const SVR_SOFT_ENABLE: u32 = 1 << 8;
const SPURIOUS_VECTOR: u32 = 0xFF;
const LVT_PERIODIC: u32 = 1 << 17;
const DIVIDE_BY_16: u32 = 0b0011;

/// Physical base of the LAPIC (from `IA32_APIC_BASE`, and set its global-enable bit).
fn lapic_phys_base() -> u64 {
    let (lo, hi): (u32, u32);
    // SAFETY: `rdmsr` of IA32_APIC_BASE (0x1B) is side-effect-free; it exists on every APIC CPU.
    unsafe {
        asm!("rdmsr", in("ecx") APIC_BASE_MSR, out("eax") lo, out("edx") hi, options(nomem, nostack, preserves_flags));
    }
    let val = ((hi as u64) << 32) | lo as u64;
    // Ensure the APIC is globally enabled (bit 11), then keep the same base.
    let enabled = val | MSR_APIC_GLOBAL_ENABLE;
    // SAFETY: `wrmsr` writes back IA32_APIC_BASE with the global-enable bit set and the base
    // unchanged — the architected way to guarantee the LAPIC is on.
    unsafe {
        asm!("wrmsr", in("ecx") APIC_BASE_MSR, in("eax") enabled as u32, in("edx") (enabled >> 32) as u32, options(nomem, nostack, preserves_flags));
    }
    enabled & 0x000f_ffff_ffff_f000
}

fn reg(off: usize) -> *mut u32 {
    (phys_to_virt(lapic_phys_base() + off as u64)) as *mut u32
}

fn write(off: usize, val: u32) {
    // SAFETY: `off` is a valid 32-bit LAPIC register; the LAPIC MMIO page is mapped (HHDM covers
    // the sub-4 GiB APIC base). QEMU dispatches the access to the device regardless of cacheability.
    unsafe { write_volatile(reg(off), val) };
}

/// Mask both 8259 PICs so no legacy IRQ reaches the CPU — the LAPIC timer is our only source.
fn mask_legacy_pic() {
    // SAFETY: writing 0xFF to each PIC's data port (0x21 master, 0xA1 slave) masks all its IRQ
    // lines; these are the standard 8259 I/O ports and have no other effect.
    unsafe {
        asm!("out dx, al", in("dx") 0x21u16, in("al") 0xFFu8, options(nomem, nostack, preserves_flags));
        asm!("out dx, al", in("dx") 0xA1u16, in("al") 0xFFu8, options(nomem, nostack, preserves_flags));
    }
}

/// Enable the LAPIC and start its timer firing periodically at (nominally) `hz` on the
/// preemption vector.
pub fn timer_init(hz: u32) {
    mask_legacy_pic();
    // Software-enable the LAPIC with a spurious-interrupt vector, and accept all priorities.
    write(REG_SVR, SVR_SOFT_ENABLE | SPURIOUS_VECTOR);
    write(REG_TPR, 0);
    // Periodic timer on TIMER_VECTOR, divide the APIC clock by 16.
    write(REG_TIMER_DIVIDE, DIVIDE_BY_16);
    write(REG_LVT_TIMER, LVT_PERIODIC | u32::from(TIMER_VECTOR));
    // Nominal count assuming a ~1 GHz APIC clock / 16 (uncalibrated — see module docs). Clamp to
    // a sane floor so a tiny `hz` cannot produce a zero (disabled) count.
    let count = (1_000_000_000u32 / 16)
        .checked_div(hz.max(1))
        .unwrap_or(0)
        .max(10_000);
    write(REG_TIMER_INITIAL, count);
}

/// Signal end-of-interrupt to the LAPIC — call once per handled timer tick, before switching.
pub fn eoi() {
    write(REG_EOI, 0);
}
