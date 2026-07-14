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

/// A panic in a reference process fails closed to `PROC_EXIT` with the sentinel error code — there
/// is no serial or unwinding in userspace, so relinquishing the CPU is the only sane action.
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    exit(u64::MAX)
}
