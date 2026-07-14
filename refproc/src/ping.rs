//! refproc `ping` (P7b) — the first reference userspace process. P7b-i milestone: prove the real
//! `.pex` toolchain end-to-end — a compiled NATIVE binary, loaded by the kernel from a `.pex`, run
//! at EL0/ring-3, making a capability-mediated syscall (this replaces the P7a in-kernel bring-up
//! blob). P7b-i.2 turns this into an `ENDPOINT_SEND` to `pong`; P7b-ii red-teams isolation.
#![no_std]
#![no_main]

/// Process entry — the `.pex` entry point. The kernel drops to it at EL0/ring-3 with all GPRs
/// zeroed (initial-register ABI) and SP at the top of the user stack.
#[no_mangle]
pub extern "C" fn _start() -> ! {
    refproc::debug(0x5049_4E47); // "PING" — proves real native EL0 code + a capability-mediated syscall
    refproc::exit(0)
}
