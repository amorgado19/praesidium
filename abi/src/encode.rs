//! Minimal `.pex` encoder (ADR-0006). Produces a byte-exact image the [`crate::pex`] decoder
//! accepts. `no_std` + allocation-free: it writes into a caller-provided buffer, so both the
//! kernel (building a test image in a heap buffer) and the host `tools/` producer (a `Vec`) use
//! the same code. A polished linker/objcopy-equivalent toolchain is deferred to P7 (ADR-0006
//! NEG-004) — this is the small producer P6 needs to exercise the loader.

use crate::pex::{
    HEADER_LEN, MANIFEST_LEN, MAX_MANIFEST, MAX_SEGMENTS, PEX_MAGIC, PEX_VERSION, SEGMENT_LEN,
};

/// One segment to encode. `data` are the file bytes (`file_size = data.len()`); the mapped size
/// is `mem_size` (`>= data.len()`, 4 KiB-aligned; the tail is zero-filled at load).
pub struct SegmentSpec<'a> {
    /// Virtual address to map at (4 KiB-aligned).
    pub vaddr: u64,
    /// Total mapped size (4 KiB-aligned, `>= data.len()`).
    pub mem_size: u64,
    /// Permission bits (`PERM_*`; must be a valid non-W^X combination).
    pub perm: u8,
    /// The file bytes.
    pub data: &'a [u8],
}

/// One manifest entry to encode (a declared initial capability).
pub struct ManifestSpec {
    /// Wire capability type (`MANIFEST_*`).
    pub cap_type: u8,
    /// Destination slot (cptr) in the new process's CSpace.
    pub dest_slot: u16,
    /// Requested rights (wire bits).
    pub rights: u32,
    /// Type-specific parameter 0.
    pub param0: u64,
    /// Type-specific parameter 1.
    pub param1: u64,
}

/// The exact encoded length of a `.pex` with these segments + manifest (header + tables + the
/// concatenated segment data). Returns `None` on overflow or if the counts exceed format limits.
#[must_use]
pub fn encoded_len(segs: &[SegmentSpec], man: &[ManifestSpec]) -> Option<usize> {
    if segs.len() > MAX_SEGMENTS || man.len() > MAX_MANIFEST {
        return None;
    }
    let mut len = HEADER_LEN
        .checked_add(segs.len().checked_mul(SEGMENT_LEN)?)?
        .checked_add(man.len().checked_mul(MANIFEST_LEN)?)?;
    for s in segs {
        len = len.checked_add(s.data.len())?;
    }
    Some(len)
}

fn wr_u16(b: &mut [u8], o: usize, v: u16) {
    b[o..o + 2].copy_from_slice(&v.to_le_bytes());
}
fn wr_u32(b: &mut [u8], o: usize, v: u32) {
    b[o..o + 4].copy_from_slice(&v.to_le_bytes());
}
fn wr_u64(b: &mut [u8], o: usize, v: u64) {
    b[o..o + 8].copy_from_slice(&v.to_le_bytes());
}

/// Encode a `.pex` into `out`, returning the number of bytes written. Returns `None` if `out` is
/// too small, the counts exceed format limits, or a size does not fit `u32`. The result parses
/// cleanly under [`crate::pex::Pex::parse`] for the matching `arch` (as long as the caller's
/// segments/entry are themselves valid — e.g. W^X perms, aligned vaddr, entry in an exec segment).
#[must_use]
pub fn encode(
    arch: u16,
    entry: u64,
    segs: &[SegmentSpec],
    man: &[ManifestSpec],
    out: &mut [u8],
) -> Option<usize> {
    let total = encoded_len(segs, man)?;
    if out.len() < total || u32::try_from(total).is_err() {
        return None;
    }
    let out = &mut out[..total];
    out.fill(0);

    let seg_off = HEADER_LEN;
    let man_off = seg_off + segs.len() * SEGMENT_LEN;
    let mut data_cursor = man_off + man.len() * MANIFEST_LEN;

    // Header.
    wr_u32(out, 0, PEX_MAGIC);
    wr_u16(out, 4, PEX_VERSION);
    wr_u16(out, 6, arch);
    wr_u64(out, 8, entry);
    wr_u32(out, 16, total as u32);
    wr_u32(out, 20, u32::try_from(seg_off).ok()?);
    wr_u16(out, 24, segs.len() as u16);
    wr_u16(out, 26, man.len() as u16);
    wr_u32(out, 28, u32::try_from(man_off).ok()?);
    // integ_off / integ_len (32,36) and the reserved tail (40) stay zero (fill above).

    // Segment table + appended segment data.
    for (i, s) in segs.iter().enumerate() {
        let rec = seg_off + i * SEGMENT_LEN;
        let file_size = u32::try_from(s.data.len()).ok()?;
        wr_u32(out, rec, u32::try_from(data_cursor).ok()?);
        wr_u32(out, rec + 4, file_size);
        wr_u64(out, rec + 8, s.vaddr);
        wr_u64(out, rec + 16, s.mem_size);
        out[rec + 24] = s.perm;
        out[data_cursor..data_cursor + s.data.len()].copy_from_slice(s.data);
        data_cursor += s.data.len();
    }

    // Manifest table.
    for (i, m) in man.iter().enumerate() {
        let rec = man_off + i * MANIFEST_LEN;
        out[rec] = m.cap_type;
        wr_u16(out, rec + 2, m.dest_slot);
        wr_u32(out, rec + 4, m.rights);
        wr_u64(out, rec + 8, m.param0);
        wr_u64(out, rec + 16, m.param1);
    }

    Some(total)
}
