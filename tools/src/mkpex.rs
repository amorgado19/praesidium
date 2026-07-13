//! `mkpex` — the minimal host-side `.pex` producer (ADR-0006 T6.8).
//!
//! P6 needs a producer that emits a valid `.pex` the kernel loader accepts; that is all this is.
//! It writes a small **reference image** (an RX code segment, an RW data segment, and a capability
//! manifest) for the requested architecture, using the same [`abi::encode`] the kernel uses to
//! build its in-boot test image — so the on-disk format and the in-kernel format are provably the
//! same code. A full linker/objcopy-equivalent that packages *compiled userspace programs* into a
//! `.pex` is deferred to P7 (its consumers — the reference processes — arrive then); the `.pex`
//! format itself is already locked + fuzzed in P6, so that producer is mechanical emission against
//! a fixed contract.
//!
//! Usage: `mkpex [--arch x86_64|aarch64] -o <out.pex>`

use std::io::Write;
use std::process::exit;

use abi::encode::{encode, encoded_len, ManifestSpec, SegmentSpec};
use abi::pex::{
    ARCH_AARCH64, ARCH_X86_64, MANIFEST_ENDPOINT, MANIFEST_FRAME, MANIFEST_SCHED, PERM_R, PERM_W,
    PERM_X,
};

// Wire rights bits — these mirror `cap-core`'s `Rights` layout (the loader maps them back via
// `Rights::from_bits`). Kept as literals here so the host tool need not depend on cap-core.
const R_READ: u32 = 1 << 0;
const R_SEND: u32 = 1 << 6;
const R_DERIVE: u32 = 1 << 9;

// Inside the loader's reserved process VA window [1 GiB, 2 GiB) (see kernel loader::PROC_VA_BASE).
const ENTRY: u64 = 0x4000_0000;

fn usage() -> ! {
    eprintln!("usage: mkpex [--arch x86_64|aarch64] -o <out.pex>");
    exit(2);
}

fn main() {
    let mut arch = ARCH_X86_64;
    let mut out: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--arch" => {
                arch = match args.next().as_deref() {
                    Some("x86_64") => ARCH_X86_64,
                    Some("aarch64") => ARCH_AARCH64,
                    _ => usage(),
                };
            }
            "-o" | "--output" => out = Some(args.next().unwrap_or_else(|| usage())),
            "-h" | "--help" => usage(),
            _ => usage(),
        }
    }
    let out = out.unwrap_or_else(|| usage());

    // A reference image: filler code + data (never executed by the P6 loader), and a manifest of
    // a Sched budget, a badged Endpoint, and a readable Frame to the code segment.
    let code = [0x90u8; 16];
    let data = [0xA5u8; 16];
    let segs = [
        SegmentSpec {
            vaddr: ENTRY,
            mem_size: 0x1000,
            perm: PERM_R | PERM_X,
            data: &code,
        },
        SegmentSpec {
            vaddr: ENTRY + 0x1000,
            mem_size: 0x1000,
            perm: PERM_R | PERM_W,
            data: &data,
        },
    ];
    let man = [
        ManifestSpec {
            cap_type: MANIFEST_SCHED,
            dest_slot: 1,
            rights: R_DERIVE,
            param0: 100,
            param1: 1000,
        },
        ManifestSpec {
            cap_type: MANIFEST_ENDPOINT,
            dest_slot: 2,
            rights: R_SEND,
            param0: 0xBADD,
            param1: 0,
        },
        ManifestSpec {
            cap_type: MANIFEST_FRAME,
            dest_slot: 3,
            rights: R_READ,
            param0: 0,
            param1: 0,
        },
    ];

    let mut buf =
        vec![0u8; encoded_len(&segs, &man).expect("reference image within format limits")];
    let n = encode(arch, ENTRY, &segs, &man, &mut buf).expect("encode reference .pex");
    buf.truncate(n);

    match std::fs::File::create(&out).and_then(|mut f| f.write_all(&buf)) {
        Ok(()) => println!("mkpex: wrote {out} ({n} bytes, arch={arch})"),
        Err(e) => {
            eprintln!("mkpex: failed to write {out}: {e}");
            exit(1);
        }
    }
}
