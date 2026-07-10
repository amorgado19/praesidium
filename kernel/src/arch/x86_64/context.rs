//! x86-64 stackful context switch (ADR-0003 P3b — the preemptive-fallback machinery).
//!
//! A [`Context`] is a task's saved kernel stack pointer; the task's callee-saved registers
//! live on its own stack. [`context_switch`] pushes the current task's callee-saved set, swaps
//! `rsp`, and pops the next task's — so it works identically whether the outgoing task yielded
//! cooperatively (a plain call) or was parked mid-interrupt (the timer ISR calls this too): in
//! both cases the saved `rsp` captures exactly where to resume. This is the one place the
//! kernel does register-file-swapping context switching (DEC-0003-1's Tier-2 fallback).

use core::arch::naked_asm;

/// A saved execution context: just the kernel stack pointer. Everything else the System V ABI
/// requires a callee to preserve (`rbx`, `rbp`, `r12`–`r15`) is pushed onto the task's stack by
/// [`context_switch`] and restored from it, so the stack pointer alone captures the context.
#[repr(C)]
pub struct Context {
    pub sp: usize,
}

impl Context {
    /// An empty context; filled by [`context_init`] before a task is first scheduled.
    pub const EMPTY: Self = Self { sp: 0 };
}

/// Switch from the task whose context is `*from` to the task whose context is `*to`: save the
/// current callee-saved registers, record `rsp` into `from`, load `to`'s `rsp`, restore its
/// callee-saved registers, and `ret` into wherever it left off.
///
/// # Safety
/// `from` must point to writable storage for the outgoing context; `to` must point to a context
/// previously produced by [`context_init`] or a prior `context_switch` (i.e. a valid saved
/// stack), and its stack + code must stay mapped. Interrupts must be masked across the call
/// (the caller's obligation) so the half-swapped state is never observed by the timer ISR.
#[unsafe(naked)]
pub unsafe extern "C" fn context_switch(from: *mut Context, to: *const Context) {
    // System V: rdi = from, rsi = to. Save callee-saved, swap stacks, restore, return.
    naked_asm!(
        "push rbp",
        "push rbx",
        "push r12",
        "push r13",
        "push r14",
        "push r15",
        "mov [rdi], rsp", // from->sp = current rsp (Context.sp is at offset 0)
        "mov rsp, [rsi]", // rsp = to->sp
        "pop r15",
        "pop r14",
        "pop r13",
        "pop r12",
        "pop rbx",
        "pop rbp",
        "ret", // resume the target: its trampoline (new task) or its context_switch caller
    );
}

/// The entry stub a freshly-created task lands on the first time it is scheduled. It force-
/// aligns the stack (16-byte ABI) and calls the generic task launcher, which runs the task's
/// body and never returns.
#[unsafe(naked)]
extern "C" fn task_trampoline() -> ! {
    naked_asm!(
        "and rsp, -16",   // defensive 16-byte alignment before entering Rust
        "call {enter}",   // enter is `-> !`; the call keeps the frame consistent
        "ud2",            // unreachable — task_enter never returns
        enter = sym crate::sched::task_enter,
    );
}

/// Build the initial [`Context`] for a new task whose stack occupies `[stack_base, stack_top)`.
/// The first [`context_switch`] into it pops six zeroed callee-saved slots and `ret`s into
/// [`task_trampoline`], which launches the task body.
///
/// # Safety
/// `stack_top` must be the (exclusive) top of a writable, mapped kernel stack of at least a few
/// hundred bytes; this writes the initial frame just below it.
pub unsafe fn context_init(stack_top: *mut u8) -> Context {
    // 7 u64 slots: the trampoline return address above 6 zeroed callee-saved registers.
    let mut sp = (stack_top as usize) & !0xf; // 16-align the top
    let mut push = |val: usize| {
        sp -= 8;
        // SAFETY: `sp` walks down from the 16-aligned stack top through the space the caller
        // guarantees is writable; each slot is 8 bytes and stays within the stack.
        unsafe { core::ptr::write(sp as *mut usize, val) };
    };
    let trampoline: extern "C" fn() -> ! = task_trampoline;
    push(trampoline as usize); // ret target of the first switch-in
    for _ in 0..6 {
        push(0); // r15, r14, r13, r12, rbx, rbp
    }
    Context { sp }
}
