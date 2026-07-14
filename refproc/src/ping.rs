//! refproc `ping` (P7b) — the first reference userspace process. P7b-i milestone: prove the real
//! `.pex` toolchain end-to-end — a compiled NATIVE binary, loaded by the kernel from a `.pex`, run
//! at EL0/ring-3, making a capability-mediated syscall (this replaces the P7a in-kernel bring-up
//! blob). P7b-i.2 turns this into an `ENDPOINT_SEND` to `pong`; P7b-ii red-teams isolation.
#![no_std]
#![no_main]

/// Process entry — the `.pex` entry point. The kernel drops to it at EL0/ring-3 with all GPRs
/// zeroed (initial-register ABI) and SP at the top of the user stack. ping is the IPC CLIENT: it
/// CALLs a value over the shared capability Endpoint and logs the reply (AC7.2 round-trip).
#[no_mangle]
pub extern "C" fn _start() -> ! {
    let reply = refproc::call(0xCAFE); // cross-process capability CALL -> pong -> reply
    refproc::debug(reply); // the reply (pong returns msg + 1 => 0xcaff)
    refproc::exit(0)
}
