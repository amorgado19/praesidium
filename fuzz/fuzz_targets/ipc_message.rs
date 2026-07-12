//! Fuzz the IPC message descriptor decoder — the hostile-input parser on the IPC boundary
//! (CLAUDE.md: all IPC messages are HOSTILE; fuzz every hostile-input parser).
//!
//! Property: for ANY untrusted bytes, `MessageInfo::decode` is total and panic-free, never
//! returns counts that exceed the fixed limits (so the kernel can never be tricked into
//! indexing the register/cap arrays out of bounds), and any accepted descriptor round-trips.
#![no_main]

use ipc::{MessageInfo, MAX_CAPS, MSG_REGS};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    for chunk in data.chunks(8) {
        let mut w = [0u8; 8];
        w[..chunk.len()].copy_from_slice(chunk);
        let word = u64::from_le_bytes(w);

        // Must never panic, and must never accept out-of-range counts.
        if let Ok(mi) = MessageInfo::decode(word) {
            assert!(usize::from(mi.n_regs) <= MSG_REGS, "n_regs escaped the bound");
            assert!(usize::from(mi.n_caps) <= MAX_CAPS, "n_caps escaped the bound");
            // A validated descriptor round-trips through encode/decode.
            assert_eq!(MessageInfo::decode(mi.encode()), Ok(mi), "encode/decode not stable");
        }
    }
});
