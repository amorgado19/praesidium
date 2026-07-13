//! Host tests for the `.pex` format: encode→decode round-trip + exhaustive malformed rejection
//! (ADR-0006 AC6.2 — a malformed `.pex` is refused, not UB). The decoder is additionally fuzzed
//! (`fuzz/fuzz_targets/pex_parse.rs`); these tests pin the specific shapes.

use abi::encode::{encode, encoded_len, ManifestSpec, SegmentSpec};
use abi::pex::{
    Pex, PexError, ARCH_X86_64, MANIFEST_SCHED, PERM_R, PERM_W, PERM_X, PEX_MAGIC, PEX_VERSION,
};

const ARCH: u16 = ARCH_X86_64;

/// A minimal valid image: one RX segment at 0x1000 (entry inside it) + one Sched manifest entry.
fn valid_pex() -> Vec<u8> {
    let code = [0x90u8; 16]; // filler "code"
    let segs = [SegmentSpec {
        vaddr: 0x1000,
        mem_size: 0x1000,
        perm: PERM_R | PERM_X,
        data: &code,
    }];
    let man = [ManifestSpec {
        cap_type: MANIFEST_SCHED,
        dest_slot: 1,
        rights: 0,
        param0: 100,
        param1: 1000,
    }];
    let mut buf = vec![0u8; encoded_len(&segs, &man).unwrap()];
    let n = encode(ARCH, 0x1000, &segs, &man, &mut buf).unwrap();
    assert_eq!(n, buf.len());
    buf
}

#[test]
fn round_trip() {
    let buf = valid_pex();
    let pex = Pex::parse(&buf, ARCH).expect("valid pex parses");
    assert_eq!(pex.entry(), 0x1000);
    assert_eq!(pex.segment_count(), 1);
    assert_eq!(pex.manifest_count(), 1);

    let s = pex.segment(0);
    assert_eq!(s.vaddr, 0x1000);
    assert_eq!(s.mem_size, 0x1000);
    assert!(s.is_exec() && !s.is_write());
    assert_eq!(pex.segment_data(0), &[0x90u8; 16]);

    let m = pex.manifest(0);
    assert_eq!(m.cap_type, MANIFEST_SCHED);
    assert_eq!(m.dest_slot, 1);
    assert_eq!(m.param0, 100);
    assert_eq!(m.param1, 1000);
}

#[test]
fn empty_buffer_and_short_header() {
    assert_eq!(Pex::parse(&[], ARCH), Err(PexError::TooShort));
    assert_eq!(Pex::parse(&[0u8; 10], ARCH), Err(PexError::TooShort));
}

#[test]
fn bad_magic() {
    let mut buf = valid_pex();
    buf[0] ^= 0xff;
    assert_eq!(Pex::parse(&buf, ARCH), Err(PexError::BadMagic));
}

#[test]
fn bad_version() {
    let mut buf = valid_pex();
    buf[4] = (PEX_VERSION + 1) as u8;
    assert_eq!(Pex::parse(&buf, ARCH), Err(PexError::BadVersion));
}

#[test]
fn arch_mismatch() {
    let buf = valid_pex();
    assert_eq!(Pex::parse(&buf, ARCH + 1), Err(PexError::ArchMismatch));
}

#[test]
fn len_mismatch_truncated() {
    let buf = valid_pex();
    // Drop the last byte: total_len (in the header) no longer matches the buffer.
    assert_eq!(
        Pex::parse(&buf[..buf.len() - 1], ARCH),
        Err(PexError::LenMismatch)
    );
}

#[test]
fn reserved_tail_nonzero() {
    let mut buf = valid_pex();
    buf[40] = 1; // reserved u64 at offset 40
    assert_eq!(Pex::parse(&buf, ARCH), Err(PexError::ReservedNonZero));
}

#[test]
fn wx_segment_refused() {
    // Encode is willing to write a W+X perm; the DECODER must refuse it (W^X, CAP-MEM-1).
    let code = [0x90u8; 16];
    let segs = [SegmentSpec {
        vaddr: 0x1000,
        mem_size: 0x1000,
        perm: PERM_R | PERM_W | PERM_X,
        data: &code,
    }];
    let mut buf = vec![0u8; encoded_len(&segs, &[]).unwrap()];
    encode(ARCH, 0x1000, &segs, &[], &mut buf).unwrap();
    assert_eq!(Pex::parse(&buf, ARCH), Err(PexError::BadPermission));
}

#[test]
fn non_readable_segment_refused() {
    let code = [0x90u8; 16];
    let segs = [SegmentSpec {
        vaddr: 0x1000,
        mem_size: 0x1000,
        perm: PERM_X, // execute-only, not readable
        data: &code,
    }];
    let mut buf = vec![0u8; encoded_len(&segs, &[]).unwrap()];
    encode(ARCH, 0x1000, &segs, &[], &mut buf).unwrap();
    assert_eq!(Pex::parse(&buf, ARCH), Err(PexError::BadPermission));
}

#[test]
fn unaligned_vaddr_refused() {
    let code = [0x90u8; 16];
    let segs = [SegmentSpec {
        vaddr: 0x1234, // not 4 KiB-aligned
        mem_size: 0x1000,
        perm: PERM_R | PERM_X,
        data: &code,
    }];
    let mut buf = vec![0u8; encoded_len(&segs, &[]).unwrap()];
    encode(ARCH, 0x1234, &segs, &[], &mut buf).unwrap();
    assert_eq!(Pex::parse(&buf, ARCH), Err(PexError::Unaligned));
}

#[test]
fn entry_outside_exec_segment_refused() {
    let code = [0x90u8; 16];
    let segs = [SegmentSpec {
        vaddr: 0x1000,
        mem_size: 0x1000,
        perm: PERM_R | PERM_X,
        data: &code,
    }];
    let mut buf = vec![0u8; encoded_len(&segs, &[]).unwrap()];
    // Entry past the only segment.
    encode(ARCH, 0x9000, &segs, &[], &mut buf).unwrap();
    assert_eq!(Pex::parse(&buf, ARCH), Err(PexError::EntryNotExecutable));
}

#[test]
fn entry_in_non_exec_segment_refused() {
    let data = [0u8; 16];
    let segs = [SegmentSpec {
        vaddr: 0x1000,
        mem_size: 0x1000,
        perm: PERM_R | PERM_W, // data, not executable
        data: &data,
    }];
    let mut buf = vec![0u8; encoded_len(&segs, &[]).unwrap()];
    encode(ARCH, 0x1000, &segs, &[], &mut buf).unwrap();
    assert_eq!(Pex::parse(&buf, ARCH), Err(PexError::EntryNotExecutable));
}

#[test]
fn corrupt_segment_offset_is_caught_not_ub() {
    // Hand-corrupt the segment file_off to point past the buffer; parse must reject, not panic.
    let mut buf = valid_pex();
    // Segment table starts at HEADER_LEN (48); file_off is the first u32 of the record.
    let seg_off = 48usize;
    buf[seg_off..seg_off + 4].copy_from_slice(&0xffff_ff00u32.to_le_bytes());
    let r = Pex::parse(&buf, ARCH);
    assert!(matches!(r, Err(PexError::SegmentOutOfBounds)), "got {r:?}");
}

#[test]
fn oversize_mem_size_refused() {
    // A hostile mem_size that would truncate a loader's u32 page count must be rejected at parse
    // (the OOB-write footgun's defense-in-depth). Hand-build: valid header/segment but a huge,
    // 4 KiB-aligned mem_size.
    let mut buf = valid_pex();
    // Segment record 0 is at HEADER_LEN (48); mem_size is the u64 at record offset +16.
    let mem_size_off = 48 + 16;
    buf[mem_size_off..mem_size_off + 8].copy_from_slice(&0x1000_0000_1000u64.to_le_bytes());
    assert_eq!(Pex::parse(&buf, ARCH), Err(PexError::SegmentTooLarge));
}

#[test]
fn magic_and_version_constants_are_stable() {
    // Trip-wire: these are the frozen wire contract; a change here is an ABI break.
    assert_eq!(PEX_MAGIC, 0x0158_4550);
    assert_eq!(PEX_VERSION, 1);
}
