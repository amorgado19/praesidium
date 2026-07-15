//! refproc `restartcli` (bridge substrate.3) — the client that drives the crash-restart demo. It
//! CALLs the server around a crash: a normal request (served), the `CRASH` sentinel (acknowledged,
//! then the server dies), and a second request that BLOCKS until the kernel supervisor has restarted
//! the server — which then serves it transparently. The client never learns the server died; that
//! transparency is the whole point (a P8 FS client / P9 driver client survives a server restart).
#![no_std]
#![no_main]

/// Must match `crashd::CRASH` / `crashd::STOP`.
const CRASH: u64 = 0x0000_C7A5;
const STOP: u64 = 0xDEAD_5709;

/// Process entry — dropped to at EL0/ring-3 with GPRs zeroed.
#[no_mangle]
pub extern "C" fn _start() -> ! {
    let a = refproc::call(0x100); // served by instance 1 -> 0x101
    refproc::debug(a);
    let ack = refproc::call(CRASH); // instance 1 acks (CRASH+1), then faults + dies
    refproc::debug(ack);
    let b = refproc::call(0x200); // BLOCKS until the RESTARTED instance serves it -> 0x201
    refproc::debug(b);
    let done = refproc::call(STOP); // shut the restarted server down (STOP+1 = 0xdead570a)
    refproc::debug(done);
    refproc::exit(0)
}
