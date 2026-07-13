//! Fuzz the `.pex` decoder — the hostile-input parser on the loader boundary (ADR-0006 R2,
//! CLAUDE.md GC-03: every `.pex` is HOSTILE input; fuzz every hostile-input parser).
//!
//! Property: for ANY untrusted bytes, `Pex::parse` is total and panic-free, and any *accepted*
//! image is fully self-consistent — every segment's file bytes and every manifest record are
//! in-bounds, so the loader can read them without a bounds check ever failing. If parse accepts
//! an image, exercising every accessor must not panic or read out of bounds.
#![no_main]

use abi::pex::{Pex, ARCH_AARCH64, ARCH_X86_64};
use libfuzzer_sys::fuzz_target;

fn exercise(data: &[u8], arch: u16) {
    if let Ok(pex) = Pex::parse(data, arch) {
        // Every advertised segment must be readable without panicking or escaping the buffer.
        for i in 0..pex.segment_count() {
            let s = pex.segment(i);
            let bytes = pex.segment_data(i);
            assert_eq!(bytes.len(), s.file_size as usize, "segment_data length mismatch");
            // A valid segment is non-W^X and readable (the parser guaranteed it).
            assert!(!(s.is_write() && s.is_exec()), "W^X escaped the parser");
            assert!(u64::from(s.file_size) <= s.mem_size, "file_size > mem_size escaped");
        }
        // Every manifest record must be readable without panicking.
        for i in 0..pex.manifest_count() {
            let _ = pex.manifest(i);
        }
        // The entry point was proven to fall inside an executable segment.
        let entry = pex.entry();
        let entry_ok = (0..pex.segment_count()).any(|i| {
            let s = pex.segment(i);
            s.is_exec() && s.vaddr <= entry && entry < s.vaddr + s.mem_size
        });
        assert!(entry_ok, "accepted image with entry outside every exec segment");
    }
}

fuzz_target!(|data: &[u8]| {
    // Parse under both arch tags — the parser must be total either way.
    exercise(data, ARCH_X86_64);
    exercise(data, ARCH_AARCH64);
});
