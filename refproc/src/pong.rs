//! refproc `pong` (P7b) — the second reference userspace process. In P7b-i.2 it RECVs on the shared
//! capability endpoint and REPLYs (the cross-process IPC round-trip, AC7.2); for the P7b-i.1
//! toolchain milestone it just makes a capability-mediated syscall and exits, proving two distinct
//! real `.pex` binaries build + load.
#![no_std]
#![no_main]

/// Process entry — see [`ping`](../ping/index.html). Dropped to at EL0/ring-3 with GPRs zeroed.
/// pong is the IPC SERVER: it RECVs a message over the shared capability Endpoint, logs it, and
/// REPLYs `msg + 1` (consuming the single-use Reply cap) — the round-trip's other half (AC7.2).
#[no_mangle]
pub extern "C" fn _start() -> ! {
    let msg = refproc::recv(); // block for ping's CALL
    refproc::debug(msg); // the received message (0xcafe)
    refproc::reply(msg.wrapping_add(1)); // reply 0xcaff to ping, consuming the Reply cap
    refproc::exit(0)
}
