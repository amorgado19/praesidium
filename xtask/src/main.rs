//! Praesidium host-side build / boot-smoke harness (mirrors Warden's xtask).
//!
//! ```text
//! cargo xtask build --arch <x86_64|aarch64>
//! cargo xtask smoke --arch <x86_64|aarch64> [--scenario p0-rich|mem|cap|sched|preempt|ipc|isolation|loader|user|server|notify] [--no-tpm] [--timeout N]
//! ```
//!
//! `build` compiles the bare-metal kernel image (build-std, the linker script, and
//! the code model all come from `.cargo/config.toml`, so it is a plain
//! `cargo build`). `smoke` stages the prebuilt Warden `.efi`, the kernel ELF, and a
//! warden-rich fixture onto a virtual-FAT ESP, boots Warden+Praesidium under QEMU
//! (OVMF/AAVMF) with an optional swtpm vTPM, captures the serial log, and asserts
//! the required and forbidden markers with a watchdog. A halted kernel keeps QEMU
//! alive until the watchdog fires — the expected clean-halt success; the markers
//! decide pass/fail.

use std::env;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Per-architecture harness parameters.
struct ArchSpec {
    /// CLI/user name.
    name: &'static str,
    /// Bare-metal kernel target triple.
    target: &'static str,
    /// Warden's own UEFI target triple (for locating its prebuilt `.efi`).
    uefi_triple: &'static str,
    /// ESP staging directory, relative to the workspace root (gitignored).
    esp_dir: &'static str,
    /// Removable-media boot path inside the ESP.
    boot_file: &'static str,
    /// QEMU system binary.
    qemu: &'static str,
    /// QEMU machine/cpu args.
    machine: &'static [&'static str],
    /// Default smoke watchdog (seconds). aarch64 runs under TCG, so it is larger.
    default_timeout: u64,
    /// swtpm TPM device model for this machine type.
    tpm_device: &'static str,
    /// Env var overriding the Warden `.efi` path for this arch.
    warden_efi_env: &'static str,
    /// Firmware `(code, vars-template)` candidate pairs, most-specific first.
    firmware: &'static [(&'static str, &'static str)],
    /// Local writable VARS copy filename (kept per-checkout, gitignored).
    vars_local: &'static str,
    /// Env overrides for firmware `(code, vars)`.
    firmware_env: (&'static str, &'static str),
}

const X86_64: ArchSpec = ArchSpec {
    name: "x86_64",
    target: "x86_64-unknown-none",
    uefi_triple: "x86_64-unknown-uefi",
    esp_dir: "test-assets/esp",
    boot_file: "EFI/BOOT/BOOTX64.EFI",
    qemu: "qemu-system-x86_64",
    machine: &["-machine", "accel=kvm:tcg"],
    default_timeout: 60,
    tpm_device: "tpm-tis",
    warden_efi_env: "WARDEN_EFI_X64",
    firmware: &[
        (
            "/usr/share/edk2/x64/OVMF_CODE.4m.fd",
            "/usr/share/edk2/x64/OVMF_VARS.4m.fd",
        ),
        (
            "/usr/share/OVMF/OVMF_CODE_4M.fd",
            "/usr/share/OVMF/OVMF_VARS_4M.fd",
        ),
        (
            "/usr/share/OVMF/OVMF_CODE.fd",
            "/usr/share/OVMF/OVMF_VARS.fd",
        ),
        (
            "/usr/share/edk2-ovmf/x64/OVMF_CODE.fd",
            "/usr/share/edk2-ovmf/x64/OVMF_VARS.fd",
        ),
    ],
    vars_local: "OVMF_VARS.local.fd",
    firmware_env: ("OVMF_CODE", "OVMF_VARS"),
};

const AARCH64: ArchSpec = ArchSpec {
    name: "aarch64",
    // Soft-float: the P3b context switch preserves only integer registers, so the kernel must
    // emit no FP/SIMD (matches x86-64-unknown-none, which is soft-float by default).
    target: "aarch64-unknown-none-softfloat",
    uefi_triple: "aarch64-unknown-uefi",
    esp_dir: "test-assets/esp-a64",
    boot_file: "EFI/BOOT/BOOTAA64.EFI",
    qemu: "qemu-system-aarch64",
    // `-cpu max` (Armv8.5+) + `mte=on` so the machine emulates FEAT_MTE — the P5 hardware
    // isolation backstop (synchronous EL1 tag-check faults). cortex-a72 (Armv8.0) has no MTE/PAC.
    // Precondition (verified before P5 MTE plumbing): P0-P4 still boot green on this CPU model.
    machine: &["-machine", "virt,mte=on", "-cpu", "max"],
    default_timeout: 180,
    tpm_device: "tpm-tis-device",
    warden_efi_env: "WARDEN_EFI_A64",
    firmware: &[
        (
            "/usr/share/edk2/aarch64/QEMU_EFI.fd",
            "/usr/share/edk2/aarch64/QEMU_VARS.fd",
        ),
        (
            "/usr/share/AAVMF/AAVMF_CODE.fd",
            "/usr/share/AAVMF/AAVMF_VARS.fd",
        ),
        (
            "/usr/share/qemu-efi-aarch64/QEMU_EFI.fd",
            "/usr/share/qemu-efi-aarch64/QEMU_VARS.fd",
        ),
    ],
    vars_local: "AAVMF_VARS.local.fd",
    firmware_env: ("AAVMF_CODE", "AAVMF_VARS"),
};

/// A boot-smoke scenario: the serial markers to assert, keyed by `--scenario`.
struct Scenario {
    name: &'static str,
    /// All must appear.
    required: &'static [&'static str],
    /// None may appear.
    forbidden: &'static [&'static str],
    /// Headline success marker (the last required) — also the early-exit trigger.
    success: &'static str,
    /// Optional x86-64 QEMU machine/cpu override (replaces [`ArchSpec::machine`] on x86 only). The
    /// isolation-carrying scenario pins `accel=tcg -cpu max` so PKU is *emulated* deterministically,
    /// byte-identically everywhere — the x86 parallel to aarch64's `-cpu max -machine mte=on`, so the
    /// PKU isolation primary is actually exercised in CI rather than varying by host/KVM (anti-theater).
    x86_machine: Option<&'static [&'static str]>,
}

/// Forbidden markers shared by every scenario.
const FORBIDDEN: &[&str] = &[
    "PRAESIDIUM PANIC",
    "[praesidium] PANIC",
    "[praesidium] FATAL",
    "WARDEN PANIC",
    "could not obtain UEFI memory map",
];

/// P0: Warden handoff + serial + clean halt.
const P0_REQUIRED: &[&str] = &[
    "jumping to warden-rich kernel", // Warden loaded + launched us
    "[praesidium] warden-rich kernel entered", // our entry ran
    "[praesidium] CONTRACT OK",      // magic + abi_version validated
    "[praesidium] memmap regions=",  // memmap dumped
    "PRAESIDIUM-P0-OK",              // headline success
];

/// P1 (mem): P0 plus the memory subsystem on its own page tables with W^X.
const P1_REQUIRED: &[&str] = &[
    "[praesidium] CONTRACT OK",
    "PRAESIDIUM-P0-OK",
    "[praesidium] mem: buddy managing", // buddy over the Warden memmap
    "[praesidium] mem: own page tables active", // AC1.4
    "[praesidium] mem: W^X verified",   // AC1.3
    "distinct + zeroed + writable",     // AC1.1
    "[praesidium] mem: slab",           // AC1.2
    "zero-on-retype verified",          // CAP-MEM-2
    "PRAESIDIUM-P1-OK",
];

/// P2 (cap): P1 plus the capability core exercising RETYPE/MINT/COPY/REVOKE.
const P2_REQUIRED: &[&str] = &[
    "PRAESIDIUM-P1-OK",
    "[praesidium] cap: root Untyped",
    "RETYPE 4 Frames",                  // AC2.1
    "widening refused",                 // AC2.2
    "REVOKE destroyed all descendants", // AC2.3
    "revoked cptr -> EmptySlot",        // CAP-REVOKE-1: fails cleanly
    "PRAESIDIUM-P2-OK",
];

/// P3a (sched): P2 plus the kernel heap, Sched budget accounting, and the cooperative
/// executor. (Preemption / AC3.3 is P3b — not asserted here.)
const P3A_REQUIRED: &[&str] = &[
    "PRAESIDIUM-P2-OK",
    "KiB kernel heap",                // heap carved from the buddy is up
    "SPLIT/DELEGATE monotonic",       // AC3.4: budget conserved
    "Sched subtree revoked cleanly",  // Sched revoke destroys no frames
    "cooperative yields interleaved", // AC3.1: executor interleaves Futures
    "budget gated task A",            // AC3.2: depleted Sched ⇒ parked
    "replenishment resumed task A",   // CAP-SCHED-1 lifts on replenish
    "PRAESIDIUM-P3A-OK",
];

/// P3b (preempt): P3a plus interrupts, the timer, and the stackful scheduler preempting a
/// non-yielding task — the full P3 gate (AC3.3).
const PREEMPT_REQUIRED: &[&str] = &[
    "PRAESIDIUM-P3A-OK",
    "stackful scheduler live", // interrupts + timer up
    "cooperative Futures interleaved while preemptible", // Tier-1 cooperative under Tier-2
    "non-yielding hog preempted", // AC3.3: real hardware preemption
    "PRAESIDIUM-P3-OK",
];

/// P4 (ipc): P3 plus synchronous capability IPC — call/reply, single-use Reply, passive-server
/// budget, no-AS-swap fast path, and GRANT over IPC (AC4.1–AC4.5).
const IPC_REQUIRED: &[&str] = &[
    "PRAESIDIUM-P3-OK",
    "rendezvous is cap-gated", // RI: no ambient authority — Endpoint cap required
    "call/reply round-trip ok", // AC4.1
    "GRANT moved Frame",       // AC4.5
    "second reply on the consumed Reply", // AC4.2 (CAP-REPLY-1)
    "passive server ran on caller budget", // AC4.3
    "address-space root unchanged", // AC4.4 (SASOS: no page-table swap)
    "PRAESIDIUM-P4-OK",
];

/// P5 (isolation): P4 plus the full SASOS isolation backstop (ADR-0008). P5a — Layer 1 compile-time
/// nameability, Layer 3 guard pages + W^X + zero, capability-gated domain entry (AC5.1/AC5.3,
/// DEC-0008-5). P5b — the raw-pointer escape red-team (DEC-0008-7): guard pages actively fault and
/// a cross-domain raw access is contained on both arches (x86 per-domain page table / aarch64 MTE).
const ISOLATION_REQUIRED: &[&str] = &[
    "PRAESIDIUM-P4-OK",
    "Cap<T> unforgeable outside cap-core",   // Layer 1 (AC5.1)
    "guard page unmapped, neighbour intact", // Layer 3 guard pages (AC5.3)
    "W^X re-verified",                       // Layer 3 W^X + zero (AC5.3)
    "domain entry is capability-gated, never ambient", // DEC-0008-5
    "PRAESIDIUM-P5A-OK",
    // P5b (DEC-0008-7): the raw-pointer escape red-team is CONTAINED on both arches.
    "guard-page raw read CONTAINED", // Layer 3 guard pages actively fault (recovery seam)
    "cross-domain raw access CONTAINED", // Layer 2 domain escape held (x86 page-table / aarch64 MTE)
    "PRAESIDIUM-P5B-OK",
];

/// P6 (loader): P5 plus the capability-invocation ABI + `.pex` loader (ADR-0006). A `.pex` parses
/// (malformed refused), the loader mints exactly the manifest caps + maps W^X segments, `invoke`
/// resolves cptrs + rights-checks, and a bad-rights invoke is refused (AC6.1–AC6.4). EL0 dispatch
/// of the loaded process is P7.
const LOADER_REQUIRED: &[&str] = &[
    "PRAESIDIUM-P5B-OK",
    ".pex loaded — entry",                       // AC6.2/6.3: parsed + loaded
    "segments mapped W^X — code R-X",            // AC6.3
    "process holds EXACTLY its 4 manifest caps", // AC6.4: no ambient authority
    "invoke resolves cptrs + rights-checked",    // AC6.1: dispatch + bad-rights refused
    "malformed .pex refused",                    // AC6.2
    "PRAESIDIUM-P6-OK",
];

/// P7a (user): P6 plus the EL0 userspace transport — drop to EL0, run native code, service its
/// syscall trap (aarch64; x86-64 ring 3 is a follow-on). Bring-up validation gate.
const USER_REQUIRED: &[&str] = &[
    "PRAESIDIUM-P6-OK",
    "EL0 userspace transport",        // entered P7a
    "EL0 syscall DEBUG value=0xbeef", // real EL0 code made a capability-mediated syscall
    "process exited (code 0)",        // clean exit via a capability invocation
    "EL0 fault CONTAINED",            // an EL0 raw read of a supervisor page killed the process, kernel survived
    "PRAESIDIUM-P7A-OK",
    // P7b (i.3): TWO REAL refproc binaries do a CROSS-PROCESS capability IPC round-trip (AC7.2) —
    // ping CALLs 0xcafe over the shared Endpoint; pong RECVs it + REPLYs 0xcaff; ping gets the reply.
    "ping.pex loaded",                    // the loader accepted the real .pex + minted its manifest caps
    "pong.pex loaded",
    "EL0 syscall DEBUG value=0xcafe",     // pong received ping's message over the shared capability Endpoint
    "EL0 syscall DEBUG value=0xcaff",     // ping got pong's reply (the round-trip closed)
    "PRAESIDIUM-P7B-I3-OK",
    // P7b-ii (AC7.3): the isolation RED-TEAM — a hostile native .pex raw-reads pong's memory; the
    // hardware backstop (PKU/MTE) faults the read, the kernel kills evil, ping+pong+kernel survive.
    "isolation armed",                    // per-process domains assigned (PKU keys / MTE tags)
    "AC7.3 isolation red-team CONTAINED", // the hostile cross-domain raw read faulted; evil killed
    "PRAESIDIUM-P7B-II-OK",
    // v1.1 (Target A): the shared read-only transfer region — ping publishes bulk into a region
    // co-mapped RW in its table / RO in pong's; pong reads it zero-copy through its SharedRo window
    // and echoes it. Then a hostile RO-window holder proves it cannot reach beyond the region.
    "shared RO transfer region co-mapped", // the kernel co-mapped it (RI via caps, no EL0 map op)
    "EL0 syscall DEBUG value=0x5eedda7ad00df00d", // pong (+ ping's echo) read the bulk sentinel zero-copy
    "zero-copy transfer OK",              // the region round-trip closed
    "shared-region red-team CONTAINED",   // a hostile RO-window holder couldn't reach beyond the region
    "PRAESIDIUM-V1.1-A-OK",
];

/// Bridge substrate.1: a PERSISTENT userspace server (`echod`, a RECV-serve loop) serves a client
/// (`echocli`) MANY requests from ONE long-lived task — the shape P8/P9 servers take (vs one-shot).
const SERVER_REQUIRED: &[&str] = &[
    "PRAESIDIUM-V1.1-A-OK",             // prior phases boot
    "echod.pex loaded",                // the persistent server .pex loaded
    "echocli.pex loaded",              // its client loaded
    "EL0 syscall DEBUG value=0x101",   // request 1 served (echo of 0x100)
    "EL0 syscall DEBUG value=0x104",   // request 4 served — proves the SAME task served many, not one-shot
    "EL0 syscall DEBUG value=0xdead570a", // the STOP acknowledged (server shut down gracefully)
    "persistent server served",        // the kernel's summary of the RECV-serve loop
    "PRAESIDIUM-SERVER-1-OK",
];

/// Bridge substrate.2: the `Notification` async-signal runtime — a userspace waiter BLOCKS in
/// NOTIFY_WAIT, the kernel raises the notification (a P9 IRQ stand-in), the waiter WAKES. Ordered
/// markers (0x5a17 before the block, the kernel raise, 0xa1e after) prove the wake follows the signal.
const NOTIFY_REQUIRED: &[&str] = &[
    "PRAESIDIUM-SERVER-1-OK",           // prior substrate increment booted
    "waiter.pex loaded",
    "EL0 syscall DEBUG value=0x5a17",   // the waiter ran and is about to WAIT
    "waiter BLOCKED in NOTIFY_WAIT",    // it genuinely blocked; the kernel now raises the notification
    "EL0 syscall DEBUG value=0xa1e",    // it WOKE — NOTIFY_WAIT returned after the kernel signal
    "Notification WAIT/SIGNAL OK",
    "PRAESIDIUM-NOTIFY-OK",
];

const SCENARIOS: &[Scenario] = &[
    Scenario {
        name: "p0-rich",
        required: P0_REQUIRED,
        forbidden: FORBIDDEN,
        success: "PRAESIDIUM-P0-OK",
        x86_machine: None,
    },
    Scenario {
        name: "mem",
        required: P1_REQUIRED,
        forbidden: FORBIDDEN,
        success: "PRAESIDIUM-P1-OK",
        x86_machine: None,
    },
    Scenario {
        name: "cap",
        required: P2_REQUIRED,
        forbidden: FORBIDDEN,
        success: "PRAESIDIUM-P2-OK",
        x86_machine: None,
    },
    Scenario {
        name: "sched",
        required: P3A_REQUIRED,
        forbidden: FORBIDDEN,
        success: "PRAESIDIUM-P3A-OK",
        x86_machine: None,
    },
    Scenario {
        name: "preempt",
        required: PREEMPT_REQUIRED,
        forbidden: FORBIDDEN,
        success: "PRAESIDIUM-P3-OK",
        x86_machine: None,
    },
    Scenario {
        name: "ipc",
        required: IPC_REQUIRED,
        forbidden: FORBIDDEN,
        success: "PRAESIDIUM-P4-OK",
        x86_machine: None,
    },
    Scenario {
        name: "isolation",
        required: ISOLATION_REQUIRED,
        forbidden: FORBIDDEN,
        success: "PRAESIDIUM-P5B-OK",
        x86_machine: None,
    },
    Scenario {
        name: "loader",
        required: LOADER_REQUIRED,
        forbidden: FORBIDDEN,
        success: "PRAESIDIUM-P6-OK",
        x86_machine: None,
    },
    Scenario {
        name: "user",
        required: USER_REQUIRED,
        forbidden: FORBIDDEN,
        success: "PRAESIDIUM-V1.1-A-OK",
        // The isolation-carrying scenario: force TCG + `-cpu max` on x86 so PKU is emulated
        // deterministically (host-/KVM-independent), guaranteeing the PKU isolation primary is
        // exercised every run — not silently replaced by the fallback under a no-KVM CI.
        x86_machine: Some(&["-machine", "accel=tcg", "-cpu", "max"]),
    },
    Scenario {
        name: "server",
        required: SERVER_REQUIRED,
        forbidden: FORBIDDEN,
        success: "PRAESIDIUM-SERVER-1-OK",
        // Runs after the isolation scenario in the same boot; PKU/MTE + per-domain tables are live,
        // so pin the same deterministic TCG+max on x86 (the server processes are per-domain-isolated).
        x86_machine: Some(&["-machine", "accel=tcg", "-cpu", "max"]),
    },
    Scenario {
        name: "notify",
        required: NOTIFY_REQUIRED,
        forbidden: FORBIDDEN,
        success: "PRAESIDIUM-NOTIFY-OK",
        x86_machine: Some(&["-machine", "accel=tcg", "-cpu", "max"]),
    },
];

fn scenario(name: &str) -> Option<&'static Scenario> {
    SCENARIOS.iter().find(|s| s.name == name)
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("build") => cmd_build(&args[1..]),
        Some("smoke") => cmd_smoke(&args[1..]),
        _ => {
            usage();
            std::process::exit(2);
        }
    };
    match result {
        Ok(true) => std::process::exit(0),
        Ok(false) => std::process::exit(1),
        Err(e) => {
            eprintln!("[xtask] error: {e}");
            std::process::exit(1);
        }
    }
}

fn usage() {
    eprintln!(
        "usage:\n  \
         cargo xtask build --arch <x86_64|aarch64>\n  \
         cargo xtask smoke --arch <x86_64|aarch64> [--scenario p0-rich|mem|cap|sched|preempt|ipc|isolation|loader|user|server|notify] [--no-tpm] [--timeout <secs>]"
    );
}

// --------------------------------------------------------------------------
// Subcommands
// --------------------------------------------------------------------------

fn cmd_build(args: &[String]) -> Result<bool, String> {
    let arch = arch_from(args)?;
    build_kernel(arch)?;
    Ok(true)
}

fn cmd_smoke(args: &[String]) -> Result<bool, String> {
    let arch = arch_from(args)?;
    let use_tpm = !flag(args, "--no-tpm");
    let scenario_name = arg_value(args, "--scenario").unwrap_or_else(|| "loader".into());
    let sc = scenario(&scenario_name).ok_or_else(|| {
        format!("unknown --scenario {scenario_name} (have: p0-rich, mem, cap, sched, preempt, ipc, isolation, loader, user, server, notify)")
    })?;
    let (required, success): (&[&str], &str) = (sc.required, sc.success);
    // An x86 scenario pinned to TCG (the isolation `user` scenario) boots far slower than KVM, so
    // its default watchdog matches the TCG-sized aarch64 budget unless overridden explicitly.
    let tcg_x86 = arch.name == "x86_64" && sc.x86_machine.is_some();
    let default_timeout = if tcg_x86 {
        arch.default_timeout.max(180)
    } else {
        arch.default_timeout
    };
    let timeout = arg_value(args, "--timeout")
        .map(|s| s.parse::<u64>().map_err(|_| format!("bad --timeout: {s}")))
        .transpose()?
        .unwrap_or(default_timeout);

    let root = workspace_root();
    let kernel = build_kernel(arch)?;
    let warden_efi = resolve_warden_efi(arch)?;
    let esp = stage_esp(&root, arch, &kernel, &warden_efi)?;
    let (code, vars) = resolve_firmware(&root, arch)?;

    let tpm = if use_tpm { start_swtpm(&root) } else { None };
    let tpm_sock = tpm.as_ref().map(|(_, s)| s.clone());

    eprintln!(
        "[xtask] booting {} : Warden+QEMU (OVMF/AAVMF{}), watchdog {}s",
        arch.name,
        if tpm_sock.is_some() { "+swtpm" } else { "" },
        timeout
    );
    // Capture the run outcome WITHOUT `?` so swtpm is always cleaned up, even if
    // run_qemu errors (otherwise a leaked swtpm child lingers).
    // Per-scenario x86 machine override (e.g. the `user` isolation scenario pins `accel=tcg -cpu max`
    // for deterministic PKU); other scenarios and aarch64 use the arch default.
    let machine: &[&str] = match (arch.name, sc.x86_machine) {
        ("x86_64", Some(m)) => m,
        _ => arch.machine,
    };
    let run = run_qemu(
        &root,
        arch,
        machine,
        &esp,
        &code,
        &vars,
        tpm_sock.as_deref(),
        timeout,
        sc.forbidden,
        success,
    );

    if let Some((mut child, _)) = tpm {
        let _ = child.kill();
        let _ = child.wait();
    }
    let (log, exited_early) = run?;

    println!(
        "----- captured serial ({}, scenario {}) -----",
        arch.name, sc.name
    );
    print!("{log}");
    println!("\n----- end serial ({}) -----", arch.name);

    let markers_ok = assert_markers(&log, required, sc.forbidden);
    // Positive clean-halt check: a passing P0 kernel parks in hlt/wfi, so QEMU must
    // still have been running when the watchdog killed it. An early self-exit
    // (reset / triple-fault under -no-reboot) is a failure even if markers appear.
    if exited_early {
        eprintln!("[xtask] FAIL: guest exited on its own (reset/triple-fault) — not a clean halt");
    }
    Ok(markers_ok && !exited_early)
}

// --------------------------------------------------------------------------
// Build
// --------------------------------------------------------------------------

/// Build the kernel ELF for `arch` (release, `-D warnings` via the crates' own
/// `[lints]`). Returns the artifact path. build-std + linker script + code model
/// all come from `.cargo/config.toml`, so this is a plain `cargo build`.
fn build_kernel(arch: &ArchSpec) -> Result<PathBuf, String> {
    let root = workspace_root();
    let cargo = env::var("CARGO").unwrap_or_else(|_| "cargo".into());

    // P7b: build + package the refproc userspace binaries FIRST, so the kernel can `include_bytes!`
    // their `.pex` images (there is no FS; Warden hands empty modules — embed them in the image).
    // ping (SEND) and pong (SEND+RECV) link at distinct bases so they coexist in the shared address
    // space; both share ONE Endpoint (the loader derives each cap from one authority) for AC7.2 IPC.
    let ping_pex = build_refproc(arch, "ping", "send", "0x91", None)?;
    let pong_pex = build_refproc(arch, "pong", "sendrecv", "0x92", Some("0x40300000"))?;
    // P7b-ii red-team: the HOSTILE `evil` binary at its own base — raw-reads pong's segment
    // (0x40300000). It holds an Endpoint (SEND) only to report a breach; the read must fault first.
    let evil_pex = build_refproc(arch, "evil", "send", "0x93", Some("0x40500000"))?;
    // Bridge substrate: the persistent echo SERVER (`echod`) + its client (`echocli`), sharing one
    // Endpoint at distinct bases — proves a userspace RECV-serve loop serves many requests. echod is
    // SEND+RECV (RECV to serve; SEND because DEBUG/EXIT are modelled as Endpoint sends in bring-up).
    let echod_pex = build_refproc(arch, "echod", "sendrecv", "0x94", Some("0x40100000"))?;
    let echocli_pex = build_refproc(arch, "echocli", "send", "0x95", Some("0x40300000"))?;
    // Bridge substrate.2: `waiter` WAITs on a Notification the kernel signals (the P9 IRQ->driver
    // wake path). SEND for DEBUG/EXIT; the kernel installs its Notification (WAIT) cap at run time.
    let waiter_pex = build_refproc(arch, "waiter", "send", "0x96", None)?;

    eprintln!(
        "[xtask] building kernel for {} ({})",
        arch.name, arch.target
    );
    let status = Command::new(&cargo)
        .current_dir(&root)
        // The kernel `include_bytes!(env!("PRAESIDIUM_{PING,PONG,EVIL,ECHOD,ECHOCLI}_PEX"))` them.
        .env("PRAESIDIUM_PING_PEX", &ping_pex)
        .env("PRAESIDIUM_PONG_PEX", &pong_pex)
        .env("PRAESIDIUM_EVIL_PEX", &evil_pex)
        .env("PRAESIDIUM_ECHOD_PEX", &echod_pex)
        .env("PRAESIDIUM_ECHOCLI_PEX", &echocli_pex)
        .env("PRAESIDIUM_WAITER_PEX", &waiter_pex)
        .args([
            "build",
            // build-std is passed here (not in .cargo/config.toml) so it applies only to the
            // bare-metal kernel image and never to host builds — keeping `cargo test` working.
            "-Zbuild-std=core,alloc,compiler_builtins",
            "-Zbuild-std-features=compiler-builtins-mem",
            "-p",
            "kernel",
            "--release",
            "--target",
            arch.target,
        ])
        .status()
        .map_err(|e| format!("failed to run cargo: {e}"))?;
    if !status.success() {
        return Err(format!("kernel build failed for {}", arch.name));
    }
    let elf = root
        .join("target")
        .join(arch.target)
        .join("release/praesidium");
    if !elf.exists() {
        return Err(format!(
            "expected kernel artifact missing: {}",
            elf.display()
        ));
    }
    Ok(elf)
}

/// Build a refproc userspace binary `bin` for `arch` (P7b) with the low-VA userspace linker script
/// + build-std, then package the ELF into a `.pex` via `tools/mkpex`. Returns the `.pex` path,
/// which `build_kernel` feeds to the kernel via `PRAESIDIUM_*_PEX` for `include_bytes!`.
///
/// The userspace `RUSTFLAGS` override the kernel's high-half config (env takes precedence over the
/// `.cargo/config.toml` target rustflags) so refproc links into the process window [1 GiB, 2 GiB)
/// with `code-model=small` on x86 (the base is low, not the kernel's -2 GiB half). refproc is a
/// DETACHED workspace, so it has its own `refproc/target`.
fn build_refproc(
    arch: &ArchSpec,
    bin: &str,
    ep_rights: &str,
    badge: &str,
    base: Option<&str>,
) -> Result<PathBuf, String> {
    let root = workspace_root();
    let cargo = env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let ld = root.join("refproc/linker-user.ld");
    let mut rustflags = format!("-C relocation-model=static -C link-arg=-T{}", ld.display());
    // A distinct link base per process so two coexist in the shared address space (P7b i.2+).
    if let Some(b) = base {
        rustflags.push_str(&format!(" -C link-arg=--defsym=__base={b}"));
    }
    if arch.name == "x86_64" {
        rustflags.push_str(" -C code-model=small");
    }
    eprintln!(
        "[xtask] building refproc/{bin} for {} ({})",
        arch.name, arch.target
    );
    let status = Command::new(&cargo)
        .current_dir(&root)
        .env("RUSTFLAGS", &rustflags)
        .args([
            "build",
            "--manifest-path",
            "refproc/Cargo.toml",
            "--release",
            "--target",
            arch.target,
            "-Zbuild-std=core,compiler_builtins",
            "-Zbuild-std-features=compiler-builtins-mem",
            "--bin",
            bin,
        ])
        .status()
        .map_err(|e| format!("failed to run cargo (refproc): {e}"))?;
    if !status.success() {
        return Err(format!("refproc/{bin} build failed for {}", arch.name));
    }
    let elf = root
        .join("refproc/target")
        .join(arch.target)
        .join("release")
        .join(bin);
    if !elf.exists() {
        return Err(format!("refproc artifact missing: {}", elf.display()));
    }

    // Package ELF -> .pex via mkpex (host tool; a workspace member).
    let pex_dir = root.join("target/pex");
    fs::create_dir_all(&pex_dir).map_err(|e| format!("mkdir {}: {e}", pex_dir.display()))?;
    let pex = pex_dir.join(format!("{bin}-{}.pex", arch.name));
    let elf_s = elf.to_str().ok_or("refproc elf path not UTF-8")?;
    let pex_s = pex.to_str().ok_or("pex path not UTF-8")?;
    eprintln!("[xtask] packaging refproc/{bin} -> {}", pex.display());
    let status = Command::new(&cargo)
        .current_dir(&root)
        .args([
            "run",
            "-q",
            "-p",
            "tools",
            "--release",
            "--bin",
            "mkpex",
            "--",
            "--arch",
            arch.name,
            "--in",
            elf_s,
            "-o",
            pex_s,
            "--endpoint-rights",
            ep_rights,
            "--badge",
            badge,
        ])
        .status()
        .map_err(|e| format!("failed to run mkpex: {e}"))?;
    if !status.success() {
        return Err(format!("mkpex failed for refproc/{bin} ({})", arch.name));
    }
    Ok(pex)
}

// --------------------------------------------------------------------------
// ESP staging
// --------------------------------------------------------------------------

/// Assemble the ESP directory QEMU serves as a virtual FAT: the Warden bootloader
/// as the removable-media boot file, the kernel ELF, and the warden-rich fixture
/// staged as `\warden.toml`.
fn stage_esp(
    root: &Path,
    arch: &ArchSpec,
    kernel: &Path,
    warden_efi: &Path,
) -> Result<PathBuf, String> {
    let esp = root.join(arch.esp_dir);
    let boot = esp.join(arch.boot_file);
    let boot_dir = boot.parent().expect("boot file has a parent dir");
    fs::create_dir_all(boot_dir).map_err(|e| format!("mkdir {}: {e}", boot_dir.display()))?;

    copy(warden_efi, &boot)?;
    copy(kernel, &esp.join("praesidium"))?;
    copy(
        &root.join("fixtures/praesidium-rich.toml"),
        &esp.join("warden.toml"),
    )?;
    Ok(esp)
}

// --------------------------------------------------------------------------
// Warden binary + firmware resolution
// --------------------------------------------------------------------------

/// Locate the prebuilt Warden `.efi` (the user's chosen "consume prebuilt via env
/// path" model): arch-specific env override, else `$WARDEN_REPO/target/<uefi
/// triple>/release/warden.efi`, else the default sibling checkout.
fn resolve_warden_efi(arch: &ArchSpec) -> Result<PathBuf, String> {
    if let Ok(p) = env::var(arch.warden_efi_env) {
        let p = PathBuf::from(p);
        if p.exists() {
            return Ok(p);
        }
        return Err(format!(
            "{}={} does not exist",
            arch.warden_efi_env,
            p.display()
        ));
    }
    let repo = env::var("WARDEN_REPO").unwrap_or_else(|_| "/home/archmorgado/warden".into());
    let efi = PathBuf::from(&repo)
        .join("target")
        .join(arch.uefi_triple)
        .join("release/warden.efi");
    if efi.exists() {
        return Ok(efi);
    }
    Err(format!(
        "Warden bootloader not found at {}.\n  \
         Set {} to the prebuilt warden.efi, or WARDEN_REPO to the Warden checkout \
         (and build it there: `cargo xtask build-a64`/`build-x64`).",
        efi.display(),
        arch.warden_efi_env
    ))
}

/// Find the UEFI firmware `(code, vars-template)` as a MATCHED PAIR — CODE and VARS
/// must come from the same firmware build, or a 2 MiB/4 MiB size mismatch wedges
/// NVRAM init and the boot silently never starts. The writable VARS is refreshed
/// from the template each run into a local, gitignored file so a stale/corrupt copy
/// can't persist and the system template is never mutated.
fn resolve_firmware(root: &Path, arch: &ArchSpec) -> Result<(PathBuf, PathBuf), String> {
    let (code_env, vars_env) = arch.firmware_env;

    let (code, vars_tpl) = match (env::var(code_env).ok(), env::var(vars_env).ok()) {
        // Both overrides given: the operator is responsible for pairing them.
        (Some(c), Some(v)) => {
            if !Path::new(&c).exists() {
                return Err(format!("{code_env}={c} does not exist"));
            }
            if !Path::new(&v).exists() {
                return Err(format!("{vars_env}={v} does not exist"));
            }
            (PathBuf::from(c), PathBuf::from(v))
        }
        // Only one override: refuse rather than silently mix a hand-picked CODE
        // with an auto-resolved VARS (the size-mismatch trap).
        (Some(_), None) | (None, Some(_)) => {
            return Err(format!("set BOTH {code_env} and {vars_env}, or neither"));
        }
        // Neither: pick the first candidate PAIR where BOTH files exist.
        (None, None) => arch
            .firmware
            .iter()
            .find(|(c, v)| Path::new(c).exists() && Path::new(v).exists())
            .map(|(c, v)| (PathBuf::from(*c), PathBuf::from(*v)))
            .ok_or_else(|| {
                format!(
                    "no complete {} firmware pair found; tried: {}",
                    arch.name,
                    arch.firmware
                        .iter()
                        .map(|(c, _)| *c)
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })?,
    };

    let vars_local = root.join(arch.vars_local);
    copy(&vars_tpl, &vars_local)?; // always refresh from the template
    Ok((code, vars_local))
}

// --------------------------------------------------------------------------
// swtpm
// --------------------------------------------------------------------------

/// Launch a fresh swtpm vTPM socket, or `None` if swtpm is unavailable (Warden's
/// measured boot degrades gracefully without a TPM).
fn start_swtpm(root: &Path) -> Option<(Child, PathBuf)> {
    let dir = root.join("test-assets/tpm");
    let _ = fs::remove_dir_all(&dir);
    if fs::create_dir_all(&dir).is_err() {
        return None;
    }
    let sock = dir.join("swtpm-sock");
    let child = Command::new("swtpm")
        .args([
            "socket",
            "--tpm2",
            "--tpmstate",
            &format!("dir={}", dir.display()),
            "--ctrl",
            &format!("type=unixio,path={}", sock.display()),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn();
    let child = match child {
        Ok(c) => c,
        Err(_) => {
            eprintln!("[xtask] swtpm not available; booting without a vTPM");
            return None;
        }
    };
    // Wait for the control socket to appear so QEMU never races it.
    for _ in 0..100 {
        if sock.exists() {
            return Some((child, sock));
        }
        thread::sleep(Duration::from_millis(50));
    }
    eprintln!("[xtask] swtpm socket did not appear; booting without a vTPM");
    let mut child = child;
    let _ = child.kill();
    None
}

// --------------------------------------------------------------------------
// QEMU run + serial capture + marker assertion
// --------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn run_qemu(
    root: &Path,
    arch: &ArchSpec,
    machine: &[&str],
    esp: &Path,
    code: &Path,
    vars: &Path,
    tpm_sock: Option<&Path>,
    timeout: u64,
    fb: &[&str],
    success_marker: &str,
) -> Result<(String, bool), String> {
    let mut args: Vec<String> = Vec::new();
    for m in machine {
        args.push((*m).into());
    }
    args.push("-m".into());
    args.push(if arch.name == "x86_64" {
        "256M".into()
    } else {
        "512M".into()
    });
    args.push("-no-reboot".into());
    // Firmware: read-only code + writable local vars, both as pflash.
    args.push("-drive".into());
    args.push(format!(
        "if=pflash,format=raw,readonly=on,file={}",
        code.display()
    ));
    args.push("-drive".into());
    args.push(format!("if=pflash,format=raw,file={}", vars.display()));
    // The ESP as a virtual FAT (no image build needed).
    args.push("-drive".into());
    args.push(format!("format=raw,file=fat:rw:{}", esp.display()));
    // Optional vTPM.
    if let Some(sock) = tpm_sock {
        args.push("-chardev".into());
        args.push(format!("socket,id=chrtpm,path={}", sock.display()));
        args.push("-tpmdev".into());
        args.push("emulator,id=tpm0,chardev=chrtpm".into());
        args.push("-device".into());
        args.push(format!("{},tpmdev=tpm0", arch.tpm_device));
    }
    // Headless: serial on stdio, no display.
    args.push("-display".into());
    args.push("none".into());
    args.push("-serial".into());
    args.push("stdio".into());

    eprintln!("[xtask] {} {}", arch.qemu, args.join(" "));

    let mut child = Command::new(arch.qemu)
        .current_dir(root)
        .args(&args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| format!("failed to launch {}: {e}", arch.qemu))?;

    // Reader thread: drain serial to EOF into a shared buffer. It does NOT early-
    // break on any marker — we must keep reading so output produced *after* the
    // success marker (e.g. a late panic) is still captured and can fail the run.
    let mut stdout = child.stdout.take().expect("piped stdout");
    let captured = Arc::new(Mutex::new(Vec::<u8>::new()));
    let cap_t = Arc::clone(&captured);
    let reader = thread::spawn(move || {
        let mut chunk = [0u8; 4096];
        loop {
            match stdout.read(&mut chunk) {
                Ok(0) => break,
                Ok(n) => cap_t
                    .lock()
                    .expect("serial buffer lock")
                    .extend_from_slice(&chunk[..n]),
                Err(_) => break,
            }
        }
    });

    let forbidden: Vec<&[u8]> = fb.iter().map(|s| s.as_bytes()).collect();
    let success = success_marker.as_bytes();
    // Once the success marker lands, keep draining for a short grace window so a
    // forbidden marker emitted right *after* it is still caught (defeats a
    // "print OK then fault" mask). Forbidden is checked first each iteration, so if
    // it coexists with success the run still fails.
    let grace = Duration::from_millis(500);
    let deadline = Instant::now() + Duration::from_secs(timeout);
    let mut success_at: Option<Instant> = None;
    let mut exited_early = false;
    loop {
        {
            let c = captured.lock().expect("serial buffer lock");
            if forbidden.iter().any(|f| contains(&c, f)) {
                break; // decided: failure
            }
            if success_at.is_none() && contains(&c, success) {
                success_at = Some(Instant::now());
            }
        }
        if let Some(t) = success_at {
            if t.elapsed() >= grace {
                break; // success seen and grace-drained: decided
            }
        }
        match child.try_wait() {
            Ok(Some(_)) => {
                exited_early = true;
                break;
            }
            Ok(None) => {
                if Instant::now() >= deadline {
                    break;
                }
                thread::sleep(Duration::from_millis(50));
            }
            Err(e) => return Err(format!("waiting on QEMU failed: {e}")),
        }
    }
    let _ = child.kill();
    let _ = child.wait();
    let _ = reader.join();
    let log = String::from_utf8_lossy(&captured.lock().expect("serial buffer lock")).into_owned();
    Ok((log, exited_early))
}

/// True iff every required marker is present and no forbidden marker appears.
fn assert_markers(log: &str, required: &[&str], forbidden: &[&str]) -> bool {
    let mut ok = true;
    for m in required {
        let hit = log.contains(m);
        eprintln!(
            "[xtask] require {} {}",
            if hit { "OK  " } else { "MISS" },
            m
        );
        ok &= hit;
    }
    for m in forbidden {
        if log.contains(m) {
            eprintln!("[xtask] forbid HIT  {m}");
            ok = false;
        }
    }
    eprintln!("[xtask] result: {}", if ok { "PASS" } else { "FAIL" });
    ok
}

// --------------------------------------------------------------------------
// Small helpers
// --------------------------------------------------------------------------

fn workspace_root() -> PathBuf {
    // xtask's manifest dir is `<root>/xtask`; the workspace root is its parent.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask has a parent")
        .to_path_buf()
}

fn arch_from(args: &[String]) -> Result<&'static ArchSpec, String> {
    match arg_value(args, "--arch").as_deref() {
        Some("x86_64") => Ok(&X86_64),
        Some("aarch64") => Ok(&AARCH64),
        Some(other) => Err(format!("unknown --arch {other} (expected x86_64|aarch64)")),
        None => Err("missing --arch <x86_64|aarch64>".into()),
    }
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

fn copy(from: &Path, to: &Path) -> Result<(), String> {
    fs::copy(from, to)
        .map(|_| ())
        .map_err(|e| format!("copy {} -> {}: {e}", from.display(), to.display()))
}

/// Byte-substring search over accumulated serial output.
fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty() && haystack.windows(needle.len()).any(|w| w == needle)
}
