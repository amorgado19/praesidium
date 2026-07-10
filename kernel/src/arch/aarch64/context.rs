//! aarch64 stackful context switch (ADR-0003 P3b — the preemptive-fallback machinery).
//!
//! A [`Context`] is a task's saved stack pointer; its callee-saved registers (`x19`–`x28`, the
//! frame pointer `x29`, the link register `x30`) live on its own stack. [`context_switch`]
//! saves them, swaps `sp`, restores the next task's, and `ret`s into `x30` — resuming the task
//! whether it yielded cooperatively or was parked mid-interrupt. SP is 16-byte aligned by
//! construction (every `stp`/`ldp` moves it by a multiple of 16), satisfying the SP-alignment
//! requirement without extra work.

use core::arch::naked_asm;

/// A saved execution context: just the stack pointer. The callee-saved registers the AAPCS64
/// requires a callee to preserve are saved on the task's own stack by [`context_switch`].
#[repr(C)]
pub struct Context {
    pub sp: usize,
}

impl Context {
    /// An empty context; filled by [`context_init`] before a task is first scheduled.
    pub const EMPTY: Self = Self { sp: 0 };
}

/// Switch from the task whose context is `*from` to the task whose context is `*to`.
///
/// # Safety
/// `from` must point to writable storage for the outgoing context; `to` must point to a context
/// from [`context_init`] or a prior `context_switch` (a valid saved stack), with its stack and
/// code mapped. Interrupts (`DAIF.I`) must be masked across the call so the half-swapped state is
/// never observed by the timer IRQ.
#[unsafe(naked)]
pub unsafe extern "C" fn context_switch(from: *mut Context, to: *const Context) {
    // AAPCS64: x0 = from, x1 = to. Save callee-saved (x19..x30) downward, swap sp, restore.
    naked_asm!(
        "stp x19, x20, [sp, #-16]!",
        "stp x21, x22, [sp, #-16]!",
        "stp x23, x24, [sp, #-16]!",
        "stp x25, x26, [sp, #-16]!",
        "stp x27, x28, [sp, #-16]!",
        "stp x29, x30, [sp, #-16]!", // x29 (fp) + x30 (lr) — the last pushed, lowest address
        "mov x9, sp",
        "str x9, [x0]", // from->sp = sp (Context.sp is at offset 0)
        "ldr x9, [x1]",
        "mov sp, x9", // sp = to->sp
        "ldp x29, x30, [sp], #16",
        "ldp x27, x28, [sp], #16",
        "ldp x25, x26, [sp], #16",
        "ldp x23, x24, [sp], #16",
        "ldp x21, x22, [sp], #16",
        "ldp x19, x20, [sp], #16",
        "ret", // resume the target via x30: its trampoline (new task) or its switch caller
    );
}

/// The entry stub a freshly-created task lands on the first time it is scheduled: launch the
/// generic task body (which never returns). SP is already 16-aligned by the ldp sequence.
#[unsafe(naked)]
extern "C" fn task_trampoline() -> ! {
    naked_asm!(
        "bl {enter}",
        "brk #0", // unreachable — task_enter never returns
        enter = sym crate::sched::task_enter,
    );
}

/// Build the initial [`Context`] for a new task whose stack top is `stack_top`. The first
/// [`context_switch`] into it restores twelve zeroed callee-saved slots — except `x30`, set to
/// [`task_trampoline`] — and `ret`s into the trampoline, launching the task body.
///
/// # Safety
/// `stack_top` must be the (exclusive) top of a writable, mapped, 16-byte-aligned kernel stack;
/// this writes the 96-byte initial frame just below it.
pub unsafe fn context_init(stack_top: *mut u8) -> Context {
    // 12 u64 slots (6 register pairs), matching context_switch's stp/ldp frame. Only the x30
    // slot (second of the first-restored pair, at sp+8) carries the trampoline; the rest are 0.
    let sp = ((stack_top as usize) & !0xf) - 96;
    let trampoline: extern "C" fn() -> ! = task_trampoline;
    // SAFETY: `[sp, sp+96)` lies within the caller-guaranteed writable stack, 16-aligned.
    unsafe {
        core::ptr::write_bytes(sp as *mut u8, 0, 96);
        core::ptr::write((sp + 8) as *mut usize, trampoline as usize); // x30 = trampoline
    }
    Context { sp }
}
