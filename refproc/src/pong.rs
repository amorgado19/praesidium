//! refproc `pong` (P7b + v1.1) — the second reference userspace process: IPC SERVER + shared-region
//! PEER (reader). Round 1 (P7b-i.3): the register capability IPC round-trip. Round 2 (v1.1): read
//! bulk data from the shared read-only window (zero-copy) that ping published, and echo it back.
#![no_std]
#![no_main]

/// Process entry — see [`ping`](../ping/index.html). Dropped to at EL0/ring-3 with GPRs zeroed.
#[no_mangle]
pub extern "C" fn _start() -> ! {
    // Round 1 (P7b-i.3): register IPC — RECV 0xCAFE, REPLY 0xCAFF (consuming the single-use Reply).
    let msg = refproc::recv();
    refproc::debug(msg); // 0xcafe
    refproc::reply(msg.wrapping_add(1)); // 0xcaff

    // Round 2 (v1.1): ping's second CALL signals "region ready". Learn the read-only window VA from
    // the SharedRo cap (RI — not ambiently), read the bulk zero-copy through it, and echo it back so
    // ping can confirm the region round-trip. pong only ever READS the region (the type is RO).
    let _signal = refproc::recv();
    let va = refproc::shared_region(); // RI: the RO window VA, learned from the SharedRo capability
    let bulk = refproc::shared_read(va); // zero-copy read of ping's published bulk data
    refproc::debug(bulk); // the received sentinel (== SHARED_SENTINEL)
    refproc::reply(bulk); // echo it — proves pong read the right region
    refproc::exit(0)
}
