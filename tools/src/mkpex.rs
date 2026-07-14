//! `mkpex` — the host-side `.pex` producer (ADR-0006 T6.8; the real ELF packager, P7b).
//!
//! Reads a compiled refproc ELF, extracts its `PT_LOAD` program segments + entry point, attaches a
//! capability manifest (the initial caps the process must be granted — a process has no ambient
//! authority, so its whole authority is exactly this list), and emits a `.pex` the kernel loader
//! accepts. It uses the SAME [`abi::encode`] the kernel uses to build its in-boot test image, so
//! the on-disk format and the in-kernel format are provably one codepath; the fuzzed
//! [`abi::pex::Pex::parse`] re-validates the result on load.
//!
//! Usage: `mkpex --arch <x86_64|aarch64> --in <elf> -o <out.pex>
//!               [--budget N] [--period N] [--badge N] [--endpoint-rights send|recv|sendrecv]`

use std::process::exit;

use abi::encode::{encode, encoded_len, ManifestSpec, SegmentSpec};
use abi::pex::{
    ARCH_AARCH64, ARCH_X86_64, MANIFEST_ENDPOINT, MANIFEST_SCHED, PERM_R, PERM_W, PERM_X,
};
use object::{Object, ObjectSegment, SegmentFlags};

// Wire rights bits — these mirror `cap-core`'s `Rights` layout (the loader maps them back via
// `Rights::from_bits`). Kept as literals here so the host tool need not depend on cap-core.
const R_SEND: u32 = 1 << 6;
const R_RECV: u32 = 1 << 7;
const R_DERIVE: u32 = 1 << 9;

/// 4 KiB page — the `.pex` requires 4 KiB-aligned vaddr AND mem_size.
const PAGE: u64 = 4096;
/// The `Sched` manifest slot (the process's CPU-time budget).
const SCHED_SLOT: u16 = 1;
/// The `Endpoint` manifest slot — must match `refproc::EP` (the cptr the refproc runtime invokes).
const EP_SLOT: u16 = 2;

fn usage() -> ! {
    eprintln!(
        "usage: mkpex --arch <x86_64|aarch64> --in <elf> -o <out.pex> \
         [--budget N] [--period N] [--badge N] [--endpoint-rights send|recv|sendrecv]"
    );
    exit(2);
}

fn die(msg: &str) -> ! {
    eprintln!("mkpex: {msg}");
    exit(1);
}

fn parse_u64(s: &str) -> u64 {
    let s = s.trim();
    let r = s
        .strip_prefix("0x")
        .map_or_else(|| s.parse::<u64>(), |h| u64::from_str_radix(h, 16));
    r.unwrap_or_else(|_| die(&format!("bad number: {s}")))
}

/// Map an ELF `PT_LOAD` segment's `p_flags` to `.pex` permission bits (PF_X=1, PF_W=2, PF_R=4).
fn seg_perm(flags: SegmentFlags) -> u8 {
    match flags {
        SegmentFlags::Elf { p_flags } => {
            let mut p = 0u8;
            if p_flags & 0x4 != 0 {
                p |= PERM_R;
            }
            if p_flags & 0x2 != 0 {
                p |= PERM_W;
            }
            if p_flags & 0x1 != 0 {
                p |= PERM_X;
            }
            p
        }
        _ => die("input is not an ELF (unexpected segment flags)"),
    }
}

fn main() {
    let mut arch_name = String::new();
    let mut infile: Option<String> = None;
    let mut out: Option<String> = None;
    let mut budget: u64 = 100;
    let mut period: u64 = 1000;
    let mut badge: u64 = 0;
    let mut ep_rights = R_SEND; // default: SEND (DEBUG_EMIT/PROC_EXIT + ping's ENDPOINT_SEND)

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--arch" => arch_name = args.next().unwrap_or_else(|| usage()),
            "--in" => infile = Some(args.next().unwrap_or_else(|| usage())),
            "-o" | "--output" => out = Some(args.next().unwrap_or_else(|| usage())),
            "--budget" => budget = parse_u64(&args.next().unwrap_or_else(|| usage())),
            "--period" => period = parse_u64(&args.next().unwrap_or_else(|| usage())),
            "--badge" => badge = parse_u64(&args.next().unwrap_or_else(|| usage())),
            "--endpoint-rights" => {
                ep_rights = match args.next().as_deref() {
                    Some("send") => R_SEND,
                    Some("recv") => R_RECV,
                    Some("sendrecv") => R_SEND | R_RECV,
                    _ => usage(),
                }
            }
            "-h" | "--help" => usage(),
            other => die(&format!("unknown argument: {other}")),
        }
    }
    let arch = match arch_name.as_str() {
        "x86_64" => ARCH_X86_64,
        "aarch64" => ARCH_AARCH64,
        _ => usage(),
    };
    let infile = infile.unwrap_or_else(|| usage());
    let out = out.unwrap_or_else(|| usage());

    let elf_bytes = std::fs::read(&infile).unwrap_or_else(|e| die(&format!("read {infile}: {e}")));
    let obj =
        object::File::parse(&*elf_bytes).unwrap_or_else(|e| die(&format!("parse ELF {infile}: {e}")));
    let entry = obj.entry();

    // Collect the PT_LOAD segments. `SegmentSpec` borrows `data`, so keep the owned bytes alive in
    // `seg_data` for the lifetime of `segs`/`encode`.
    let mut seg_data: Vec<Vec<u8>> = Vec::new();
    let mut seg_meta: Vec<(u64, u64, u8)> = Vec::new(); // (vaddr, mem_size, perm)
    for seg in obj.segments() {
        let memsz = seg.size();
        if memsz == 0 {
            continue; // not loadable
        }
        let vaddr = seg.address();
        if vaddr % PAGE != 0 {
            die(&format!(
                "segment vaddr {vaddr:#x} is not 4 KiB-aligned (fix the userspace linker script)"
            ));
        }
        let mem_size = memsz.div_ceil(PAGE) * PAGE; // round p_memsz up to a whole page
        let data = seg
            .data()
            .unwrap_or_else(|e| die(&format!("read segment data: {e}")))
            .to_vec();
        seg_meta.push((vaddr, mem_size, seg_perm(seg.flags())));
        seg_data.push(data);
    }
    if seg_meta.is_empty() {
        die("ELF has no loadable (PT_LOAD) segments");
    }

    let segs: Vec<SegmentSpec> = seg_meta
        .iter()
        .zip(seg_data.iter())
        .map(|(&(vaddr, mem_size, perm), data)| SegmentSpec {
            vaddr,
            mem_size,
            perm,
            data,
        })
        .collect();

    // Manifest: exactly the caps the reference process needs — a Sched budget + one Endpoint (the
    // shared IPC endpoint; the loader mints a badged derivation from its own Endpoint authority).
    let man = [
        ManifestSpec {
            cap_type: MANIFEST_SCHED,
            dest_slot: SCHED_SLOT,
            rights: R_DERIVE,
            param0: budget,
            param1: period,
        },
        ManifestSpec {
            cap_type: MANIFEST_ENDPOINT,
            dest_slot: EP_SLOT,
            rights: ep_rights,
            param0: badge,
            param1: 0,
        },
    ];

    let total = encoded_len(&segs, &man).unwrap_or_else(|| die("image exceeds .pex format limits"));
    let mut buf = vec![0u8; total];
    let n = encode(arch, entry, &segs, &man, &mut buf).unwrap_or_else(|| die("encode .pex failed"));
    buf.truncate(n);

    std::fs::write(&out, &buf).unwrap_or_else(|e| die(&format!("write {out}: {e}")));
    println!(
        "mkpex: wrote {out} ({n} bytes, arch={arch_name}, entry={entry:#x}, {} segment(s))",
        segs.len()
    );
}
