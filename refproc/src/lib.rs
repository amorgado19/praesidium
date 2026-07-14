//! refproc runtime — the tiny `no_std` userspace runtime shared by the P7b reference processes
//! (`ping`/`pong`). A process talks to the kernel ONLY through the capability-invocation ABI
//! (`abi`): every syscall is `sys::INVOKE` carrying an [`abi::invoke::Invocation`] in the arch
//! register ABI (the same seam the kernel's EL0/ring-3 trap handlers decode). Soft-float, no
//! allocation, no FS — the kernel grants the process exactly its `.pex` manifest capabilities.
#![no_std]

use abi::invoke::{op, sys};

/// The process's `Endpoint` capability cptr — the manifest `dest_slot` the loader mints it into.
/// The reference processes invoke it for their bring-up syscalls (`DEBUG_EMIT`/`PROC_EXIT`) and,
/// in P7b-i.2, cross-process IPC (`ENDPOINT_SEND`).
pub const EP: u32 = 2;

/// Perform a capability invocation: `sys::INVOKE` of `operation` on the capability at `cptr` with
/// `args`, returning the kernel's result word. The concrete register ABI is arch-specific (behind
/// the same seam the kernel trap handlers decode): aarch64 `x8=sel, x0=cptr, x1=op, x2..x5=args`;
/// x86-64 `rax=sel, rdi=cptr, rsi=op, rdx/r10/r8/r9=args` (`rcx`/`r11` clobbered by `syscall`).
#[inline]
#[must_use]
pub fn invoke(cptr: u32, operation: u16, args: [u64; 4]) -> u64 {
    let ret: u64;
    #[cfg(target_arch = "aarch64")]
    // SAFETY: `svc #0` traps into the kernel's capability-invocation dispatch with the aarch64
    // register ABI; the kernel restores x1..x30 and returns the result word in x0.
    unsafe {
        core::arch::asm!(
            "svc #0",
            in("x8") sys::INVOKE,
            inout("x0") u64::from(cptr) => ret,
            in("x1") u64::from(operation),
            in("x2") args[0],
            in("x3") args[1],
            in("x4") args[2],
            in("x5") args[3],
            options(nostack),
        );
    }
    #[cfg(target_arch = "x86_64")]
    // SAFETY: `syscall` traps into the kernel's LSTAR dispatch with the x86-64 register ABI; the
    // kernel returns the result in rax and clobbers rcx/r11 (the instruction) + the arg registers.
    unsafe {
        core::arch::asm!(
            "syscall",
            inout("rax") sys::INVOKE => ret,
            inout("rdi") u64::from(cptr) => _,
            inout("rsi") u64::from(operation) => _,
            inout("rdx") args[0] => _,
            inout("r10") args[1] => _,
            inout("r8") args[2] => _,
            inout("r9") args[3] => _,
            out("rcx") _,
            out("r11") _,
            options(nostack),
        );
    }
    ret
}

/// Bring-up: emit `val` to the kernel serial console via the Endpoint (`DEBUG_EMIT`, bring-up-only).
pub fn debug(val: u64) {
    let _ = invoke(EP, op::DEBUG_EMIT, [val, 0, 0, 0]);
}

/// Terminate this process with exit `code` (`PROC_EXIT`). Never returns — the kernel retires it.
pub fn exit(code: u64) -> ! {
    let _ = invoke(EP, op::PROC_EXIT, [code, 0, 0, 0]);
    // PROC_EXIT does not return to userspace; spin defensively if the kernel ever let it.
    loop {
        core::hint::spin_loop();
    }
}

/// The CSpace slot the kernel mints the single-use `Reply` capability into on [`recv`].
pub const REPLY: u32 = 3;

/// Cross-process CALL (AC7.2): send `msg` on the Endpoint and block for a one-word reply. The
/// kernel rendezvous delivers it to whoever holds a `RECV` cap to the same Endpoint. Requires the
/// process's Endpoint cap holds `SEND`.
#[must_use]
pub fn call(msg: u64) -> u64 {
    invoke(EP, op::ENDPOINT_CALL, [msg, 0, 0, 0])
}

/// Cross-process RECV: block until a caller's [`call`] arrives, returning its message word. The
/// kernel mints a single-use `Reply` capability at cptr [`REPLY`] for the ensuing [`reply`].
/// Requires the process's Endpoint cap holds `RECV`.
#[must_use]
pub fn recv() -> u64 {
    invoke(EP, op::ENDPOINT_RECV, [0, 0, 0, 0])
}

/// Reply `msg` to the blocked caller, consuming the single-use `Reply` cap at [`REPLY`].
pub fn reply(msg: u64) {
    let _ = invoke(REPLY, op::ENDPOINT_REPLY, [msg, 0, 0, 0]);
}

/// The CSpace slot holding the process's shared-region capability (v1.1): a `SharedRo` (readers) or
/// a `Frame` with WRITE (the owner). Free of the manifest slots (1=Sched, 2=Endpoint, 3=Reply).
pub const SHARED: u32 = 4;
/// The virtual address the kernel co-maps the shared transfer region at (a convention the OWNER
/// writes to; a reader learns it from its `SharedRo` cap via [`shared_region`], RI-clean).
pub const SHARED_VA: usize = 0x4070_0000;
/// The bulk sentinel the owner publishes into the shared region — a recognizable word so the peer's
/// zero-copy read of it is unambiguous in the serial log.
pub const SHARED_SENTINEL: u64 = 0x5EED_DA7A_D00D_F00D;

/// Ask the kernel where this process's read-only shared window is mapped — via its `SharedRo` cap
/// (RI: the reader learns the VA from the capability, not ambiently). Returns the read-only VA.
#[must_use]
pub fn shared_region() -> u64 {
    invoke(SHARED, op::SHARED_QUERY, [0, 0, 0, 0])
}

/// Read one word from the shared region at `va` (the co-mapped read-only window). Zero-copy: the
/// bulk lives in a region both processes' page tables see, so no kernel copy / per-message map.
#[must_use]
pub fn shared_read(va: u64) -> u64 {
    // SAFETY: `va` is the process's co-mapped shared window (from [`shared_region`] / [`SHARED_VA`]),
    // a mapped readable page; a volatile read observes whatever the owner published there.
    unsafe { core::ptr::read_volatile(va as *const u64) }
}

/// Write one word to the shared region at `SHARED_VA` — the OWNER's read-write path (the owner holds
/// a `Frame` cap with WRITE; the region is RW-mapped in its table). Publishes bulk data zero-copy.
pub fn shared_write(val: u64) {
    // SAFETY: the owner's shared region is RW-mapped at `SHARED_VA` (a `Frame` cap with WRITE
    // authorizes it); a volatile write publishes `val` for the peer to read.
    unsafe { core::ptr::write_volatile(SHARED_VA as *mut u64, val) };
}

/// Raw-read an arbitrary address (the red-team's foothold probe). Reaching a VA the process holds no
/// mapping for faults (per-domain page tables) — used to prove a shared-window holder cannot use it
/// to reach the owner's OTHER memory.
#[must_use]
pub fn raw_read(va: usize) -> u64 {
    // SAFETY: DELIBERATELY reads a VA this process may hold no capability/mapping for — the isolation
    // red-team. A sound boundary faults the read; reaching the return value is a breach.
    unsafe { core::ptr::read_volatile(va as *const u64) }
}

/// A panic in a reference process fails closed to `PROC_EXIT` with the sentinel error code — there
/// is no serial or unwinding in userspace, so relinquishing the CPU is the only sane action.
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    exit(u64::MAX)
}
