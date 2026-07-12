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
// The x86-64 IDT uses the `x86-interrupt` calling convention for CPU-exception + IRQ handlers
// (P3b). It is nightly-only; scoped to x86-64 since aarch64 hand-rolls its vector table.
#![cfg_attr(target_arch = "x86_64", feature(abi_x86_interrupt))]

extern crate alloc;

#[macro_use]
mod serial;
mod arch;
mod boot;
mod cap;
mod heap;
mod ipc;
mod isolation;
mod memory;
mod sched;
mod sync;

use boot::handoff::WardenBootInfo;

/// The kernel's own boot stack, in `.bss` — part of the kernel image, so the frame
/// allocator never hands out its frames. Warden's boot stack lives in memory the
/// allocator *does* manage (UEFI `BOOT_SERVICES_DATA` → `USABLE`), so building page
/// tables on it would clobber the running stack; the per-arch naked `_start` switches
/// to this stack before any Rust that allocates runs.
pub(crate) const BOOT_STACK_SIZE: usize = 64 * 1024;

#[repr(C, align(16))]
pub(crate) struct BootStack([u8; BOOT_STACK_SIZE]);

pub(crate) static mut BOOT_STACK: BootStack = BootStack([0; BOOT_STACK_SIZE]);

/// Kernel entry proper. The per-arch naked [`_start`](arch) sets up [`BOOT_STACK`] and
/// tail-calls this with the C first-argument register still holding the Warden handoff
/// pointer (`rdi` on x86-64, `x0` on aarch64 — the `extern "C"` ABI covers both). The
/// pointer is **HHDM-virtual**.
pub(crate) extern "C" fn kmain(bootinfo: *const WardenBootInfo) -> ! {
    arch::serial_init();
    kprintln!();
    kprintln!("[praesidium] warden-rich kernel entered");

    let bi = boot::validate_and_dump(bootinfo);
    kprintln!("[praesidium] PRAESIDIUM-P0-OK");

    // P1: bring up the physical memory subsystem (prints PRAESIDIUM-P1-OK on success).
    memory::init(bi);

    // P3a: stand up the kernel heap (carved from the P1 buddy) before anything allocates.
    heap::init();

    // P2: bring up the capability core (prints PRAESIDIUM-P2-OK on success).
    cap::run();

    // P3a: executor + capability scheduling (prints PRAESIDIUM-P3A-OK on success).
    sched::run();

    // P4: synchronous capability IPC (prints PRAESIDIUM-P4-OK on success).
    ipc::run();

    // P5a: SASOS isolation backstop foundation (prints PRAESIDIUM-P5A-OK on success).
    isolation::run();

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
