//! refproc `waiter` (bridge substrate.2) — proves the `Notification` async-signal path. It blocks in
//! `NOTIFY_WAIT` on a Notification the kernel holds the signal side of, and wakes when the kernel
//! raises it (the stand-in for a P9 in-kernel IRQ waking a userspace driver). The shape a real
//! userspace driver's main loop takes: sleep until a hardware event, capability-gated.
#![no_std]
#![no_main]

/// Process entry — dropped to at EL0/ring-3 with GPRs zeroed.
#[no_mangle]
pub extern "C" fn _start() -> ! {
    refproc::debug(0x5A17); // "about to WAIT" — logged before we block, so the wake provably follows
    refproc::notify_wait(); // sleep until the kernel raises the Notification (the P9 IRQ->driver wake)
    refproc::debug(0x0A1E); // "WOKE" — reaching here proves NOTIFY_WAIT returned after the signal
    refproc::exit(0)
}
