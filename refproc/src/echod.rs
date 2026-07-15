//! refproc `echod` (bridge substrate) — a PERSISTENT userspace SERVER. Unlike v1's one-shot
//! ping/pong (which exit after a single round-trip), `echod` runs a `RECV`-serve loop that services
//! MANY requests over its lifetime — the shape every real server (the P8 FS server, a P9 driver
//! server) takes. It echoes each request (`msg + 1`) and lives until a client sends `STOP`.
#![no_std]
#![no_main]

/// The stop sentinel — a client's final message. The server replies to it (so the client's CALL
/// returns), then exits. Everything else is served and the loop continues.
const STOP: u64 = 0xDEAD_5709;

/// Process entry — dropped to at EL0/ring-3 with GPRs zeroed. The persistent serve loop.
#[no_mangle]
pub extern "C" fn _start() -> ! {
    loop {
        let msg = refproc::recv(); // block for the next client request (the SAME long-lived task)
        refproc::reply(msg.wrapping_add(1)); // serve it: echo msg+1, consuming the single-use Reply
        if msg == STOP {
            refproc::exit(0); // graceful shutdown after acknowledging the STOP
        }
    }
}
