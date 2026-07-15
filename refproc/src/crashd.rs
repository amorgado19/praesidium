//! refproc `crashd` (bridge substrate.3) — a persistent server that CRASHES on demand, to prove the
//! kernel supervisor detects the death, REAPS the process's frames, and RESTARTS a fresh instance
//! that transparently serves the client's blocked request. It echoes each request (`msg + 1`); on the
//! `CRASH` sentinel it acknowledges first (so the client's CALL returns) and then faults on a raw read
//! of a supervisor page — the kernel kills it via the EL0 fault path, exactly like a real driver bug.
#![no_std]
#![no_main]

/// Acknowledged, then the server deliberately faults and is killed. Must match the kernel demo.
const CRASH: u64 = 0x0000_C7A5;
/// Graceful-shutdown sentinel (shared with the client). Replied to, then the server exits.
const STOP: u64 = 0xDEAD_5709;
/// A supervisor-only VA inside the process window: an EL0 read of it permission-faults (it is the
/// shared EL1-only identity mapping, never a page crashd maps for itself), so the kernel kills the
/// process — the stand-in for a fatal driver bug.
const BAD_VA: usize = 0x4090_0000;

/// Process entry — dropped to at EL0/ring-3 with GPRs zeroed. The persistent serve loop.
#[no_mangle]
pub extern "C" fn _start() -> ! {
    loop {
        let msg = refproc::recv(); // block for the next client request (the SAME long-lived task)
        refproc::reply(msg.wrapping_add(1)); // acknowledge every request (incl. CRASH / STOP)
        if msg == CRASH {
            // The ack is delivered; now DIE. This raw read of a supervisor page faults at EL0, the
            // kernel kills this process, and the supervisor reaps + restarts it. Never returns.
            let _ = refproc::raw_read(BAD_VA);
        }
        if msg == STOP {
            refproc::exit(0); // graceful shutdown after acknowledging STOP
        }
    }
}
