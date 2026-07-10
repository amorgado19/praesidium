//! Praesidium host-side build / boot-smoke harness (mirrors Warden's xtask).
//!
//! ```text
//! cargo xtask build --arch <x86_64|aarch64>
//! cargo xtask smoke --arch <x86_64|aarch64> [--scenario p0-rich|mem|cap|sched|preempt] [--no-tpm] [--timeout N]
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
    machine: &["-machine", "virt", "-cpu", "cortex-a72"],
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

const SCENARIOS: &[Scenario] = &[
    Scenario {
        name: "p0-rich",
        required: P0_REQUIRED,
        forbidden: FORBIDDEN,
        success: "PRAESIDIUM-P0-OK",
    },
    Scenario {
        name: "mem",
        required: P1_REQUIRED,
        forbidden: FORBIDDEN,
        success: "PRAESIDIUM-P1-OK",
    },
    Scenario {
        name: "cap",
        required: P2_REQUIRED,
        forbidden: FORBIDDEN,
        success: "PRAESIDIUM-P2-OK",
    },
    Scenario {
        name: "sched",
        required: P3A_REQUIRED,
        forbidden: FORBIDDEN,
        success: "PRAESIDIUM-P3A-OK",
    },
    Scenario {
        name: "preempt",
        required: PREEMPT_REQUIRED,
        forbidden: FORBIDDEN,
        success: "PRAESIDIUM-P3-OK",
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
         cargo xtask smoke --arch <x86_64|aarch64> [--scenario p0-rich|mem|cap|sched|preempt] [--no-tpm] [--timeout <secs>]"
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
    let scenario_name = arg_value(args, "--scenario").unwrap_or_else(|| "preempt".into());
    let sc = scenario(&scenario_name).ok_or_else(|| {
        format!("unknown --scenario {scenario_name} (have: p0-rich, mem, cap, sched, preempt)")
    })?;
    let timeout = arg_value(args, "--timeout")
        .map(|s| s.parse::<u64>().map_err(|_| format!("bad --timeout: {s}")))
        .transpose()?
        .unwrap_or(arch.default_timeout);

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
    let run = run_qemu(
        &root,
        arch,
        &esp,
        &code,
        &vars,
        tpm_sock.as_deref(),
        timeout,
        sc,
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

    let markers_ok = assert_markers(&log, sc.required, sc.forbidden);
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
    eprintln!(
        "[xtask] building kernel for {} ({})",
        arch.name, arch.target
    );
    let status = Command::new(&cargo)
        .current_dir(&root)
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
    esp: &Path,
    code: &Path,
    vars: &Path,
    tpm_sock: Option<&Path>,
    timeout: u64,
    sc: &Scenario,
) -> Result<(String, bool), String> {
    let mut args: Vec<String> = Vec::new();
    for m in arch.machine {
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

    let forbidden: Vec<&[u8]> = sc.forbidden.iter().map(|s| s.as_bytes()).collect();
    let success = sc.success.as_bytes();
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
