//! The `.pex` (Praesidium EXecutable) format (ADR-0006 DEC-0006-5..7).
//!
//! A `.pex` is a flat, little-endian, self-describing image carrying: loadable **segments**
//! (with W^X-compatible permissions — no writable+executable segment is representable), an
//! **entry point**, a **capability manifest** (the declaration of the initial capabilities the
//! process must be granted — there is no ambient authority, so a process's whole authority is
//! exactly this list), and a **reserved integrity/signature section** so the Warden-style
//! verify-before-execute posture can extend to userspace later.
//!
//! **Every `.pex` is HOSTILE input** (GC-03): it is parsed with explicit, panic-free bounds
//! checks — no field is read without first proving it is in range, no `&[u8]` is transmuted to a
//! header struct (which would be UB on a misaligned or too-short buffer). [`Pex::parse`] returns
//! a typed [`PexError`] for every malformed shape; it never panics and never reads out of bounds.
//! The layout is versioned + magic-guarded, and the wire encoding is deliberately arch-tagged
//! (segment bytes are native code).

/// `"PEX\x01"` little-endian — magic + a format-generation byte.
pub const PEX_MAGIC: u32 = 0x0158_4550;
/// Current format version.
pub const PEX_VERSION: u16 = 1;

/// Architecture tags (segment bytes are native code, so a `.pex` is arch-specific).
pub const ARCH_X86_64: u16 = 1;
/// aarch64 architecture tag.
pub const ARCH_AARCH64: u16 = 2;

/// Fixed byte sizes of the on-wire records.
pub const HEADER_LEN: usize = 48;
/// Size of one segment-table record.
pub const SEGMENT_LEN: usize = 32;
/// Size of one manifest-table record.
pub const MANIFEST_LEN: usize = 32;

/// Sanity caps on hostile counts — a `.pex` far beyond these is rejected outright rather than
/// trusted to describe thousands of segments/caps.
pub const MAX_SEGMENTS: usize = 16;
/// Maximum manifest entries accepted.
pub const MAX_MANIFEST: usize = 32;
/// Maximum pages (4 KiB) a single segment may span (16 MiB). Bounding `mem_size` here is what keeps
/// a hostile huge `mem_size` from later truncating a loader's `u32` page count into an under-sized
/// allocation (defense in depth with the loader's own checked conversion).
pub const MAX_SEGMENT_PAGES: u64 = 4096;

/// Segment permission bits (a subset must form a valid, non-W^X protection).
pub const PERM_R: u8 = 1 << 0;
/// Segment is writable.
pub const PERM_W: u8 = 1 << 1;
/// Segment is executable.
pub const PERM_X: u8 = 1 << 2;

/// Wire capability-type tags for manifest entries (the subset a loader can satisfy from its own
/// authority in P6). Kept distinct from `cap-core`'s `CapType`: this crate is a pure wire format
/// with no cap-core dependency; the loader validates + maps these to real capability types.
pub const MANIFEST_SCHED: u8 = 1;
/// Manifest entry requesting an `Endpoint` capability (`param0` = badge).
pub const MANIFEST_ENDPOINT: u8 = 2;
/// Manifest entry requesting a `Frame` capability to one of the image's segments
/// (`param0` = segment index).
pub const MANIFEST_FRAME: u8 = 3;

/// Every way a `.pex` can be malformed. Parsing fails closed with one of these — never a panic,
/// never undefined behavior (AC6.2).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PexError {
    /// Buffer shorter than a field/table it must contain.
    TooShort,
    /// `magic` did not match [`PEX_MAGIC`].
    BadMagic,
    /// `version` is not one this parser understands.
    BadVersion,
    /// `arch` does not match the loader's architecture.
    ArchMismatch,
    /// `total_len` disagrees with the actual buffer length.
    LenMismatch,
    /// A count exceeds [`MAX_SEGMENTS`]/[`MAX_MANIFEST`].
    TooManyRecords,
    /// A table (segments/manifest/integrity) runs past the buffer.
    TableOutOfBounds,
    /// A segment's file bytes `[file_off, file_off+file_size)` run past the buffer.
    SegmentOutOfBounds,
    /// A segment's `mem_size` is smaller than its `file_size`.
    BadMemSize,
    /// A segment spans more than [`MAX_SEGMENT_PAGES`] pages (a hostile/absurd `mem_size`).
    SegmentTooLarge,
    /// A segment permission is empty, non-readable, or writable+executable (W^X).
    BadPermission,
    /// A segment `vaddr` or `mem_size` is not 4 KiB-aligned.
    Unaligned,
    /// The entry point does not fall inside any executable segment.
    EntryNotExecutable,
    /// A reserved field was non-zero (future-proofing tripwire).
    ReservedNonZero,
}

/// Read helpers: return an error (never panic / read OOB) for a hostile, too-short buffer.
fn rd_u16(b: &[u8], off: usize) -> Result<u16, PexError> {
    let s = b.get(off..off + 2).ok_or(PexError::TooShort)?;
    Ok(u16::from_le_bytes([s[0], s[1]]))
}
fn rd_u32(b: &[u8], off: usize) -> Result<u32, PexError> {
    let s = b.get(off..off + 4).ok_or(PexError::TooShort)?;
    Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}
fn rd_u64(b: &[u8], off: usize) -> Result<u64, PexError> {
    let s = b.get(off..off + 8).ok_or(PexError::TooShort)?;
    let mut a = [0u8; 8];
    a.copy_from_slice(s);
    Ok(u64::from_le_bytes(a))
}

/// A parsed, validated segment descriptor.
#[derive(Clone, Copy, Debug)]
pub struct Segment {
    /// Offset of the segment's file bytes within the `.pex`.
    pub file_off: u32,
    /// Number of file bytes present (may be `< mem_size`; the tail is zero-filled — `.bss`).
    pub file_size: u32,
    /// Virtual address to map the segment at (4 KiB-aligned).
    pub vaddr: u64,
    /// Total mapped size (4 KiB-aligned, `>= file_size`).
    pub mem_size: u64,
    /// Permission bits (a valid, non-W^X combination).
    pub perm: u8,
}

impl Segment {
    /// Is this segment executable?
    #[must_use]
    pub fn is_exec(&self) -> bool {
        self.perm & PERM_X != 0
    }
    /// Is this segment writable?
    #[must_use]
    pub fn is_write(&self) -> bool {
        self.perm & PERM_W != 0
    }
}

/// A parsed, validated manifest entry — one initial capability the process must be granted.
#[derive(Clone, Copy, Debug)]
pub struct ManifestEntry {
    /// Wire capability type (`MANIFEST_*`).
    pub cap_type: u8,
    /// Destination slot (cptr) in the new process's CSpace.
    pub dest_slot: u16,
    /// Requested rights (wire bits; the loader validates + subset-checks against its authority).
    pub rights: u32,
    /// Type-specific parameter 0 (Sched: budget; Endpoint: badge; Frame: segment index).
    pub param0: u64,
    /// Type-specific parameter 1 (Sched: period; otherwise unused).
    pub param1: u64,
}

/// A validated view over a `.pex` byte buffer. Holds no owned data — it borrows the input and
/// exposes the header, segments, and manifest through bounds-checked accessors. Constructing one
/// (via [`Pex::parse`]) has already proven every table and segment lies within the buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pex<'a> {
    bytes: &'a [u8],
    entry: u64,
    seg_off: usize,
    seg_count: usize,
    man_off: usize,
    man_count: usize,
}

impl<'a> Pex<'a> {
    /// Parse and fully validate a `.pex` for the given architecture. Every failure mode is a
    /// typed [`PexError`]; this never panics and never reads out of bounds, for any input.
    pub fn parse(bytes: &'a [u8], arch: u16) -> Result<Pex<'a>, PexError> {
        if bytes.len() < HEADER_LEN {
            return Err(PexError::TooShort);
        }
        if rd_u32(bytes, 0)? != PEX_MAGIC {
            return Err(PexError::BadMagic);
        }
        if rd_u16(bytes, 4)? != PEX_VERSION {
            return Err(PexError::BadVersion);
        }
        if rd_u16(bytes, 6)? != arch {
            return Err(PexError::ArchMismatch);
        }
        let entry = rd_u64(bytes, 8)?;
        let total_len = rd_u32(bytes, 16)? as usize;
        if total_len != bytes.len() {
            return Err(PexError::LenMismatch);
        }
        let seg_off = rd_u32(bytes, 20)? as usize;
        let seg_count = rd_u16(bytes, 24)? as usize;
        let man_count = rd_u16(bytes, 26)? as usize;
        let man_off = rd_u32(bytes, 28)? as usize;
        let integ_off = rd_u32(bytes, 32)? as usize;
        let integ_len = rd_u32(bytes, 36)? as usize;
        // Reserved tail (bytes 40..48) must be zero.
        if rd_u64(bytes, 40)? != 0 {
            return Err(PexError::ReservedNonZero);
        }
        if seg_count > MAX_SEGMENTS || man_count > MAX_MANIFEST {
            return Err(PexError::TooManyRecords);
        }
        // Tables must lie fully within the buffer (checked-arithmetic against overflow).
        if !table_fits(bytes.len(), seg_off, seg_count, SEGMENT_LEN)
            || !table_fits(bytes.len(), man_off, man_count, MANIFEST_LEN)
        {
            return Err(PexError::TableOutOfBounds);
        }
        // The reserved integrity section (empty at v1) must also lie within the buffer.
        if integ_len != 0 {
            let end = integ_off
                .checked_add(integ_len)
                .ok_or(PexError::TableOutOfBounds)?;
            if end > bytes.len() {
                return Err(PexError::TableOutOfBounds);
            }
        }

        let pex = Pex {
            bytes,
            entry,
            seg_off,
            seg_count,
            man_off,
            man_count,
        };

        // Validate every segment now, so callers can iterate without re-checking.
        let mut entry_ok = false;
        for i in 0..seg_count {
            let s = pex.segment_raw(i)?;
            validate_perm(s.perm)?;
            if s.vaddr & 0xfff != 0 || s.mem_size & 0xfff != 0 {
                return Err(PexError::Unaligned);
            }
            if u64::from(s.file_size) > s.mem_size {
                return Err(PexError::BadMemSize);
            }
            // Bound the mapped size so a hostile `mem_size` cannot overflow a loader's page count.
            if s.mem_size >> 12 > MAX_SEGMENT_PAGES {
                return Err(PexError::SegmentTooLarge);
            }
            // The file bytes must lie within the buffer.
            let file_end = (s.file_off as usize)
                .checked_add(s.file_size as usize)
                .ok_or(PexError::SegmentOutOfBounds)?;
            if file_end > bytes.len() {
                return Err(PexError::SegmentOutOfBounds);
            }
            // Entry must fall inside an executable segment (overflow-safe: `mem_size` is bounded,
            // but the segment's `vaddr + mem_size` end is computed with a checked add regardless).
            let seg_end = s.vaddr.checked_add(s.mem_size).ok_or(PexError::Unaligned)?;
            if s.is_exec() && s.vaddr <= entry && entry < seg_end {
                entry_ok = true;
            }
        }
        if !entry_ok {
            return Err(PexError::EntryNotExecutable);
        }
        // Validate manifest entry types are readable (unknown types are refused by the loader,
        // not here — the format allows forward-compatible types; the loader fails closed on any
        // it cannot satisfy). We only bounds-check the records here.
        for i in 0..man_count {
            let _ = pex.manifest_raw(i)?;
        }
        Ok(pex)
    }

    /// The entry-point virtual address (proven to fall in an executable segment).
    #[must_use]
    pub fn entry(&self) -> u64 {
        self.entry
    }

    /// Number of loadable segments.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.seg_count
    }

    /// Number of manifest entries.
    #[must_use]
    pub fn manifest_count(&self) -> usize {
        self.man_count
    }

    fn segment_raw(&self, i: usize) -> Result<Segment, PexError> {
        let base = self.seg_off + i * SEGMENT_LEN;
        Ok(Segment {
            file_off: rd_u32(self.bytes, base)?,
            file_size: rd_u32(self.bytes, base + 4)?,
            vaddr: rd_u64(self.bytes, base + 8)?,
            mem_size: rd_u64(self.bytes, base + 16)?,
            perm: *self.bytes.get(base + 24).ok_or(PexError::TooShort)?,
        })
    }

    /// The validated `i`th segment (panics only on an out-of-range index, a caller bug).
    #[must_use]
    pub fn segment(&self, i: usize) -> Segment {
        assert!(i < self.seg_count, "segment index out of range");
        self.segment_raw(i).expect("segment validated at parse")
    }

    /// The file bytes of segment `i` (`[file_off, file_off+file_size)`), already proven in-bounds.
    #[must_use]
    pub fn segment_data(&self, i: usize) -> &'a [u8] {
        let s = self.segment(i);
        &self.bytes[s.file_off as usize..s.file_off as usize + s.file_size as usize]
    }

    fn manifest_raw(&self, i: usize) -> Result<ManifestEntry, PexError> {
        let base = self.man_off + i * MANIFEST_LEN;
        Ok(ManifestEntry {
            cap_type: *self.bytes.get(base).ok_or(PexError::TooShort)?,
            dest_slot: rd_u16(self.bytes, base + 2)?,
            rights: rd_u32(self.bytes, base + 4)?,
            param0: rd_u64(self.bytes, base + 8)?,
            param1: rd_u64(self.bytes, base + 16)?,
        })
    }

    /// The validated `i`th manifest entry (panics only on an out-of-range index).
    #[must_use]
    pub fn manifest(&self, i: usize) -> ManifestEntry {
        assert!(i < self.man_count, "manifest index out of range");
        self.manifest_raw(i).expect("manifest validated at parse")
    }
}

/// Does a table of `count` records of `rec_len` bytes starting at `off` fit within `total`?
/// All arithmetic is overflow-checked so a hostile offset/count cannot wrap.
fn table_fits(total: usize, off: usize, count: usize, rec_len: usize) -> bool {
    match count.checked_mul(rec_len).and_then(|w| off.checked_add(w)) {
        Some(end) => end <= total,
        None => false,
    }
}

/// A permission byte must be readable and must not be writable+executable (W^X, CAP-MEM-1).
fn validate_perm(perm: u8) -> Result<(), PexError> {
    // Reject unknown bits, non-readable, and the W+X combination.
    if perm & !(PERM_R | PERM_W | PERM_X) != 0
        || perm & PERM_R == 0
        || (perm & PERM_W != 0 && perm & PERM_X != 0)
    {
        return Err(PexError::BadPermission);
    }
    Ok(())
}
