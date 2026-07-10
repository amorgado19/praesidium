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

use x86_64::structures::idt::{InterruptDescriptorTable, InterruptStackFrame, PageFaultErrorCode};

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
extern "x86-interrupt" fn ex_page_fault(frame: InterruptStackFrame, _err: PageFaultErrorCode) {
    let cr2: u64;
    // SAFETY: reading CR2 (the faulting address) is side-effect-free.
    unsafe {
        core::arch::asm!("mov {}, cr2", out(reg) cr2, options(nomem, nostack, preserves_flags))
    };
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
