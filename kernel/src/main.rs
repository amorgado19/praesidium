//! Praesidium — capability-native, single-address-space, dual-arch Rust kernel.
//! Booted by Warden's rich handoff (ADR-0001). This is the P0 entry point: it
//! validates the `WardenBootInfo` contract, dumps the handed-over memory map and
//! framebuffer over serial, and halts cleanly — on both x86-64 and aarch64.
//!
//! P0 introduces **no authority**: no capability is fabricated or exercised, so
//! the SPEC-CAP Root Invariant holds vacuously (the §11 checklist is satisfied
//! trivially by code that reads a struct and stops).
#![no_std]
#![no_main]

#[macro_use]
mod serial;
mod arch;
mod boot;

use boot::handoff::WardenBootInfo;

/// Kernel entry. Warden's warden-rich loader jumps here after building the page
/// tables and exiting boot services, passing an **HHDM-virtual** pointer to
/// [`WardenBootInfo`] in the C first-argument register (`rdi` on x86-64, `x0` on
/// aarch64 — the `extern "C"` ABI covers both).
#[no_mangle]
extern "C" fn _start(bootinfo: *const WardenBootInfo) -> ! {
    arch::serial_init();
    kprintln!();
    kprintln!("[praesidium] warden-rich kernel entered");

    boot::init_and_dump(bootinfo);

    kprintln!("[praesidium] PRAESIDIUM-P0-OK");

    // Ensure all serial/MMIO writes have completed before we park the CPU.
    arch::memory_barrier();
    arch::halt();
}

/// Panic handler: serial-first, then halt (GC-01). Emitting `PANIC` is a smoke
/// test *forbidden* marker, so any panic fails the phase loudly rather than
/// hanging silently.
#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    kprintln!("[praesidium] PANIC: {info}");
    arch::halt();
}
