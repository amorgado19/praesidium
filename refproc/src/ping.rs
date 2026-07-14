//! refproc `ping` (P7b + v1.1) — the first reference userspace process: IPC CLIENT + shared-region
//! OWNER. Round 1 (P7b-i.3): a register capability IPC round-trip with `pong`. Round 2 (v1.1):
//! publish bulk data into the shared read-only transfer region (zero-copy, no per-message Frame map
//! / kernel copy), signal `pong` over IPC, and confirm `pong` read it back.
#![no_std]
#![no_main]

/// Process entry — the `.pex` entry point. The kernel drops to it at EL0/ring-3 with all GPRs
/// zeroed (initial-register ABI) and SP at the top of the user stack.
#[no_mangle]
pub extern "C" fn _start() -> ! {
    // Round 1 (P7b-i.3): register capability IPC round-trip — CALL 0xCAFE, get pong's reply 0xCAFF.
    let reply = refproc::call(0xCAFE);
    refproc::debug(reply); // 0xcaff

    // Round 2 (v1.1): publish a bulk sentinel into the shared region (owner RW, zero-copy), signal
    // pong over IPC, and confirm pong echoed the sentinel back (a round-trip through the region —
    // bulk via the co-mapped region, control via IPC, no page-table swap for the data itself).
    refproc::shared_write(refproc::SHARED_SENTINEL);
    let echo = refproc::call(0xB0_1C); // signal "region ready" + receive pong's confirmation
    refproc::debug(echo); // must equal SHARED_SENTINEL — pong read the region zero-copy and echoed it
    refproc::exit(0)
}
