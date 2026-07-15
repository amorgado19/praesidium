//! refproc `echocli` (bridge substrate) — the client of the persistent `echod` server. It CALLs the
//! server several times to prove the server serves MANY requests from ONE long-lived task (not one
//! per process), logs each served reply, then STOPs the server.
#![no_std]
#![no_main]

/// Must match `echod::STOP`.
const STOP: u64 = 0xDEAD_5709;
/// How many requests to send — enough to prove the server is persistent, not one-shot.
const REQUESTS: u64 = 4;

/// Process entry — dropped to at EL0/ring-3 with GPRs zeroed.
#[no_mangle]
pub extern "C" fn _start() -> ! {
    let mut i = 0;
    while i < REQUESTS {
        let reply = refproc::call(0x100 + i); // each CALL is served by the SAME persistent server
        refproc::debug(reply); // the served echo: 0x101, 0x102, 0x103, 0x104
        i += 1;
    }
    let ack = refproc::call(STOP); // shut the server down; it replies STOP+1 then exits
    refproc::debug(ack); // 0xdead570a
    refproc::exit(0)
}
