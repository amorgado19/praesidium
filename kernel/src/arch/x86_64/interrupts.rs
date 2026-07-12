//! x86-64 interrupt setup (ADR-0003 P3b): an IDT with fail-closed CPU-exception handlers and
//! the LAPIC timer vector that drives preemption.
//!
//! The IDT is heap-allocated and leaked to `'static` (it must outlive the `lidt`). CPU
//! exceptions print and halt (fail closed — an unexpected fault must be loud, not silent). The
//! timer handler uses the `x86-interrupt` ABI, which brackets it with a full register
//! save/restore + `iret`, so it is safe for the handler to `context_switch` to another task:
//! the outgoing task's callee-saved registers are pushed on top of this interrupt frame, and
//! resuming it later returns through the ABI epilogue's `iret`.

use alloc::boxed::Box;
use core::arch::global_asm;
use core::sync::atomic::{AtomicU64, Ordering};

use x86_64::structures::idt::{InterruptDescriptorTable, InterruptStackFrame, PageFaultErrorCode};
use x86_64::VirtAddr;

use super::timer;
use crate::sched::scheduler;

/// The LAPIC timer's IDT vector. 0–31 are CPU exceptions; 32 is the first free vector.
pub const TIMER_VECTOR: u8 = 32;

/// Build, install (`lidt`), and leave the IDT active. The table is leaked so it lives for the
/// kernel's lifetime, as `lidt` records a bare pointer.
pub fn interrupts_init() {
    let idt: &'static mut InterruptDescriptorTable =
        Box::leak(Box::new(InterruptDescriptorTable::new()));

    idt.divide_error.set_handler_fn(ex_divide);
    idt.invalid_opcode.set_handler_fn(ex_invalid_opcode);
    idt.general_protection_fault.set_handler_fn(ex_gpf);
    idt.page_fault.set_handler_fn(ex_page_fault);
    idt.double_fault.set_handler_fn(ex_double_fault);
    idt.stack_segment_fault.set_handler_fn(ex_stack_segment);
    idt[TIMER_VECTOR].set_handler_fn(timer_interrupt);

    idt.load();
}

// ---- CPU exceptions: fail closed (loud + halt) ------------------------------------------

fn fault(name: &str, frame: &InterruptStackFrame) -> ! {
    kprintln!(
        "[praesidium] FATAL: cpu exception {name} at rip={:#x} rsp={:#x}",
        frame.instruction_pointer.as_u64(),
        frame.stack_pointer.as_u64()
    );
    crate::arch::halt();
}

extern "x86-interrupt" fn ex_divide(frame: InterruptStackFrame) {
    fault("#DE (divide)", &frame);
}
extern "x86-interrupt" fn ex_invalid_opcode(frame: InterruptStackFrame) {
    fault("#UD (invalid opcode)", &frame);
}
extern "x86-interrupt" fn ex_gpf(frame: InterruptStackFrame, _err: u64) {
    fault("#GP (general protection)", &frame);
}
extern "x86-interrupt" fn ex_stack_segment(frame: InterruptStackFrame, _err: u64) {
    fault("#SS (stack segment)", &frame);
}
extern "x86-interrupt" fn ex_page_fault(mut frame: InterruptStackFrame, _err: PageFaultErrorCode) {
    let cr2: u64;
    // SAFETY: reading CR2 (the faulting address) is side-effect-free.
    unsafe {
        core::arch::asm!("mov {}, cr2", out(reg) cr2, options(nomem, nostack, preserves_flags))
    };
    // Single-shot recoverable path (P5b): if this is exactly the deliberate isolation-escape
    // probe we armed at this address, contain it (resume at the probe's recovery label) instead
    // of failing closed. Any other #PF — or a #PF at the wrong address — is a real bug: FATAL.
    if let Some(resume) = expected_fault_resume(cr2) {
        kprintln!(
            "[praesidium] isolation: CONTAINED raw cross-domain access — #PF at {cr2:#x} (x86)"
        );
        // SAFETY: overwrite only the return instruction pointer with the probe's recovery label
        // (a kernel `.text` address); the iret then resumes there. The faulting `mov` touched no
        // stack, so the frame is otherwise intact.
        unsafe {
            frame
                .as_mut()
                .update(|f| f.instruction_pointer = VirtAddr::new(resume));
        }
        return;
    }
    kprintln!(
        "[praesidium] FATAL: cpu exception #PF at rip={:#x} addr={cr2:#x}",
        frame.instruction_pointer.as_u64()
    );
    crate::arch::halt();
}
extern "x86-interrupt" fn ex_double_fault(frame: InterruptStackFrame, _err: u64) -> ! {
    fault("#DF (double fault)", &frame);
}

// ---- the preemption timer ---------------------------------------------------------------

extern "x86-interrupt" fn timer_interrupt(_frame: InterruptStackFrame) {
    // EOI FIRST, before any context switch: the LAPIC must be told the interrupt is serviced so
    // the next tick can fire while whatever task we switch to runs. EOI-after-switch would stall
    // preemption (the LAPIC would think we're still in-service until the switch returns).
    timer::eoi();
    if scheduler::on_tick() {
        scheduler::preempt();
    }
}

// ---- single-shot recoverable expected-fault (P5b isolation red-team seam) ----------------
//
// The trusted core of the isolation escape test. The whole discipline is: keep this TINY and
// fail-closed. Exactly one deliberate faulting access can be "armed" at a time, at one exact
// address; the fault handler contains ONLY a fault at that address and resumes at a fixed label,
// disarming immediately. Anything else — an unarmed fault, or a fault at any other address — is
// a real bug and still fails closed (FATAL). This is deliberately NOT a general fixup facility
// (no table, no registration): the smaller the trusted surface here, the easier it is to prove
// a real isolation-escape fault can never be silently "recovered".

/// The armed expected-fault address, or 0 when disarmed (single-shot).
static EXPECT_FAULT_ADDR: AtomicU64 = AtomicU64::new(0);
/// The instruction pointer to resume at when the armed fault fires (the probe's recovery label).
static EXPECT_FAULT_RESUME: AtomicU64 = AtomicU64::new(0);

extern "C" {
    /// Perform a single deliberate byte read of `addr` (in `rdi`); returns 1 (via the recovery
    /// label) if the read faulted and was contained, or 0 if it did not fault. See the asm below.
    fn praesidium_x86_probe_read(addr: u64) -> u64;
    /// The recovery label inside `praesidium_x86_probe_read`; the handler resumes here.
    static praesidium_x86_probe_resume: u8;
}

global_asm!(
    r#"
.section .text
.global praesidium_x86_probe_read
.global praesidium_x86_probe_resume
praesidium_x86_probe_read:
    mov al, byte ptr [rdi]      // the deliberate access — #PF if the byte is unmapped/forbidden
    xor eax, eax                // no fault: return 0 (containment FAILED — the access succeeded)
    ret
praesidium_x86_probe_resume:
    mov eax, 1                  // handler vectored us here after CONTAINED: return 1
    ret
"#
);

/// If a fault at `addr` is the armed single-shot probe, disarm and return its resume label;
/// otherwise return `None` (the caller must fail closed). Called only from the #PF handler.
fn expected_fault_resume(addr: u64) -> Option<u64> {
    let expect = EXPECT_FAULT_ADDR.load(Ordering::Relaxed);
    if expect != 0 && addr == expect {
        EXPECT_FAULT_ADDR.store(0, Ordering::Relaxed); // single-shot: disarm before resuming
        return Some(EXPECT_FAULT_RESUME.load(Ordering::Relaxed));
    }
    None
}

/// Attempt a raw byte read of `addr` that is **expected to fault** (an isolation-escape probe),
/// containing the fault. Returns `true` iff the access faulted and was contained by the hardware
/// isolation mechanism; `false` means the access *succeeded* — i.e. isolation did NOT hold.
///
/// Arms the single-shot continuation, performs the access, and disarms unconditionally, so the
/// mechanism can never be left armed. A fault at any other address while armed still fails closed.
pub fn contains_raw_read(addr: u64) -> bool {
    EXPECT_FAULT_RESUME.store(
        core::ptr::addr_of!(praesidium_x86_probe_resume) as u64,
        Ordering::Relaxed,
    );
    EXPECT_FAULT_ADDR.store(addr, Ordering::Relaxed); // arm last
                                                      // SAFETY: `praesidium_x86_probe_read` does a single byte read of `addr`. We expect it to #PF;
                                                      // the armed handler resumes at the recovery label returning 1. If `addr` is in fact mapped,
                                                      // no fault occurs and it returns 0 — reported as "not contained".
    let contained = unsafe { praesidium_x86_probe_read(addr) } == 1;
    EXPECT_FAULT_ADDR.store(0, Ordering::Relaxed); // ensure disarmed on the no-fault path
    contained
}
