//! x86-64 ring-3 (userspace) entry + syscall/fault trap (P7a) — the arch mechanism behind
//! [`crate::user`], mirroring the aarch64 EL0 backend. Ring 3 needs machinery aarch64 gets from
//! its exception model for free: a GDT (ring-0 + ring-3 code/data), a TSS (RSP0 = the kernel stack
//! a ring3->ring0 trap lands on), and the `syscall`/`sysret` MSRs. [`gdt_init`] builds them (before
//! the IDT, so its gates capture the kernel CS); [`enter_user`] drops to ring 3 via `iretq`; the
//! `syscall` instruction traps to `praesidium_x86_syscall_entry` (LSTAR), which switches to the
//! task's kernel stack, saves a register frame, and dispatches through [`crate::user`] — the same
//! generic policy + P6 [`crate::syscall::invoke`] the aarch64 path uses. Every EL0 syscall is a
//! capability invocation: `rax = sys::INVOKE`, `rdi = cptr`, `rsi = op`, `rdx/r10/r8/r9 = args`.

use core::arch::{asm, global_asm};
use core::sync::atomic::{AtomicU64, Ordering};

use alloc::boxed::Box;

use abi::invoke::{sys, Invocation, MSG_REGS};
use x86_64::instructions::tables::load_tss;
use x86_64::registers::model_specific::{Efer, EferFlags, LStar, SFMask, Star};
use x86_64::registers::rflags::RFlags;
use x86_64::registers::segmentation::{Segment, CS, DS, ES, SS};
use x86_64::structures::gdt::{Descriptor, GlobalDescriptorTable, SegmentSelector};
use x86_64::structures::tss::TaskStateSegment;
use x86_64::VirtAddr;

/// Ring-3 code/data selectors (RPL 3), captured from the GDT by [`gdt_init`] for [`enter_user`].
static USER_CS: AtomicU64 = AtomicU64::new(0);
static USER_DS: AtomicU64 = AtomicU64::new(0);
/// Address of the leaked TSS's `privilege_stack_table[0]` (RSP0) field, so [`enter_user`] can set,
/// per process, the kernel stack a ring3->ring0 trap lands on.
static TSS_RSP0: AtomicU64 = AtomicU64::new(0);
/// The kernel stack the `syscall` trap switches to (the current process's task kernel stack). Set
/// by [`enter_user`]; read by `praesidium_x86_syscall_entry` — `syscall` leaves RSP on the USER
/// stack (unlike aarch64's hardware SP_EL1 switch), so the stub must reload a kernel stack itself.
static KERNEL_RSP: AtomicU64 = AtomicU64::new(0);
/// Scratch: the user RSP the syscall stub saves across the kernel-stack switch and restores before
/// `sysretq`. IF is masked in the handler (SFMASK) and this is single-CPU, so it is non-reentrant
/// and one slot suffices.
static USER_RSP: AtomicU64 = AtomicU64::new(0);

/// Bytes the syscall stub reserves for its saved register frame — 10 u64 slots (selector/result,
/// cptr, op, 4 args, user rip, user rflags, saved user rsp) = 0x50, exactly full. 0x50 is already
/// a 16-byte multiple, so the `call` into the handler is ABI-aligned with no padding slack: there
/// is NO spare slot — adding one requires growing this constant (else the write overruns the top
/// of the task kernel stack).
const SYSCALL_FRAME_BYTES: u64 = 0x50;

/// Whether this backend can run ring-3 userspace. True: [`gdt_init`] runs at `interrupts_init`
/// (P3b), before [`crate::user::run`] (P7a), so the ring-3 machinery is always up by then.
#[must_use]
pub fn el0_supported() -> bool {
    true
}

/// Set the kernel stack the `syscall` fast path (KERNEL_RSP) and a ring3->ring0 trap (TSS.RSP0) land
/// on — the CURRENT task's kernel stack. The scheduler calls this on switch-in to every task with a
/// kernel stack, so two concurrent processes' syscalls each use their OWN kernel stack: x86
/// `syscall` does not switch stacks, unlike aarch64's per-task-banked SP_EL1 (the aarch64 seam is a
/// no-op).
pub fn set_kernel_stack(top: u64) {
    KERNEL_RSP.store(top, Ordering::Relaxed);
    // SAFETY: TSS_RSP0 points at the leaked 'static TSS's privilege_stack_table[0], 4-byte aligned in
    // the #[repr(C, packed(4))] TSS (unaligned write). The scheduler runs this preemption-masked on a
    // single CPU, so no trap reads RSP0 mid-write.
    unsafe {
        (TSS_RSP0.load(Ordering::Relaxed) as *mut VirtAddr).write_unaligned(VirtAddr::new(top));
    }
}

/// Build the GDT (ring-0 + ring-3 code/data) + a TSS, load them, reload the segment registers, and
/// program the `syscall`/`sysret` MSRs. Called at the START of [`super::interrupts_init`] — BEFORE
/// the IDT is built — so `set_handler_fn` captures this kernel CS in the interrupt gates.
pub fn gdt_init() {
    let tss: &'static mut TaskStateSegment = Box::leak(Box::new(TaskStateSegment::new()));
    // RSP0 (the kernel stack a ring3->ring0 trap lands on) is set per process by `enter_user`.
    TSS_RSP0.store(
        core::ptr::addr_of_mut!(tss.privilege_stack_table[0]) as u64,
        Ordering::Relaxed,
    );
    let tss: &'static TaskStateSegment = tss;

    let gdt: &'static mut GlobalDescriptorTable = Box::leak(Box::new(GlobalDescriptorTable::new()));
    // Append order is load-bearing for `Star::write`/`sysret`: kernel_data = kernel_code + 8, and
    // user_code = user_data + 8 (the crate validates this ordering).
    let kernel_code = gdt.append(Descriptor::kernel_code_segment());
    let kernel_data = gdt.append(Descriptor::kernel_data_segment());
    let user_data = gdt.append(Descriptor::user_data_segment());
    let user_code = gdt.append(Descriptor::user_code_segment());
    let tss_sel = gdt.append(Descriptor::tss_segment(tss));
    let gdt: &'static GlobalDescriptorTable = gdt;
    gdt.load();

    // SAFETY: `gdt` is loaded and its selectors are valid flat 64-bit segments; reloading CS/SS/DS/ES
    // to the kernel selectors is transparent (flat addressing in 64-bit mode), and `tss_sel` names
    // the TSS just installed. Order: segment registers, then the task register.
    unsafe {
        CS::set_reg(kernel_code);
        SS::set_reg(kernel_data);
        DS::set_reg(kernel_data);
        ES::set_reg(kernel_data);
        load_tss(tss_sel);
    }

    USER_CS.store(u64::from(user_code.0), Ordering::Relaxed);
    USER_DS.store(u64::from(user_data.0), Ordering::Relaxed);

    // syscall/sysret MSRs.
    // SAFETY: enabling SCE (EFER) + programming STAR/LSTAR/SFMASK configures the `syscall` fast
    // path; the LSTAR entry is a valid kernel-text address and STAR's selectors were just installed.
    unsafe {
        Efer::write(Efer::read() | EferFlags::SYSTEM_CALL_EXTENSIONS);
    }
    Star::write(user_code, user_data, kernel_code, kernel_data)
        .expect("STAR selector layout (GDT append order guarantees it)");
    LStar::write(VirtAddr::new(praesidium_x86_syscall_entry as *const () as u64));
    // Clear IF on `syscall` entry — the handler runs non-preemptibly (matches the aarch64 DAIF mask;
    // preemptible userspace is P7b).
    SFMask::write(RFlags::INTERRUPT_FLAG);

    // Probe + enable PKU (protection keys for user pages) — the P7b-ii process-vs-process isolation
    // mechanism (ADR-0008 DEC-0008-2). Done before any userspace so a process's first switch-in can
    // program PKRU; if absent, the per-domain-page-table fallback isolates instead (logged at use).
    let pku = super::pku::init();
    kprintln!("[praesidium] user: x86 PKU {} (CR4.PKE)", if pku { "enabled — process-vs-process isolation via protection keys" } else { "UNAVAILABLE — per-domain-page-table fallback will isolate" });
}

/// Drop to ring 3 at `entry` with user stack `user_sp`. Never returns: the process runs until it
/// exits (a syscall) or faults, each of which retires its scheduler task and schedules away.
///
/// # Safety
/// `entry` must map ring-3-executable code and `user_sp` a ring-3-writable, 16-aligned stack. The
/// caller must be a scheduler task whose kernel stack can host the trap frames. [`gdt_init`] must
/// have run.
pub unsafe fn enter_user(entry: u64, user_sp: u64) -> ! {
    // The trap handlers' kernel stack (KERNEL_RSP for `syscall`, TSS.RSP0 for a ring3->ring0 fault)
    // is set PER-TASK by the scheduler on switch-in ([`set_kernel_stack`]) — NOT here — so two
    // concurrent processes' syscalls each land on their own kernel stack.
    let user_cs = USER_CS.load(Ordering::Relaxed);
    let user_ds = USER_DS.load(Ordering::Relaxed);
    // SAFETY: `user_ds` is a valid ring-3 data selector; DS/ES bases are ignored in 64-bit mode, so
    // loading them before the iretq is transparent to the kernel code that follows.
    unsafe {
        DS::set_reg(SegmentSelector(user_ds as u16));
        ES::set_reg(SegmentSelector(user_ds as u16));
    }

    // Drop to ring 3 via iretq. Frame (pushed high->low): SS, RSP, RFLAGS, CS, RIP. RFLAGS = only
    // reserved bit 1 set (IF clear): a cooperative, non-preemptible bring-up process (matches the
    // aarch64 DAIF-masked EL0). All GPRs are zeroed first so no kernel value leaks across the
    // privilege drop (initial-register ABI: a process starts with GPRs = 0).
    // SAFETY: builds a well-formed iretq frame on the kernel stack from validated selectors and the
    // caller's entry/stack; `options(noreturn)` — control transfers to ring 3.
    unsafe {
        asm!(
            "push {ss}",
            "push {rsp_u}",
            "push {rflags}",
            "push {cs}",
            "push {rip}",
            "xor rax, rax", "xor rbx, rbx", "xor rcx, rcx", "xor rdx, rdx",
            "xor rsi, rsi", "xor rdi, rdi", "xor rbp, rbp",
            "xor r8, r8", "xor r9, r9", "xor r10, r10", "xor r11, r11",
            "xor r12, r12", "xor r13, r13", "xor r14, r14", "xor r15, r15",
            "iretq",
            ss = in(reg) user_ds,
            rsp_u = in(reg) user_sp,
            rflags = in(reg) 0x2u64,
            cs = in(reg) user_cs,
            rip = in(reg) entry,
            options(noreturn),
        );
    }
}

extern "C" {
    fn praesidium_x86_syscall_entry();
    static praesidium_x86_user_blob: u8;
    static praesidium_x86_user_blob_end: u8;
    static praesidium_x86_fault_blob: u8;
    static praesidium_x86_fault_blob_end: u8;
}

// The `syscall` (LSTAR) entry. `syscall` leaves RSP on the USER stack, RCX = user RIP, R11 = user
// RFLAGS, and does NOT switch stacks. Save the user RSP, switch to the task's kernel stack, spill a
// register frame (the syscall ABI: rax=selector, rdi=cptr, rsi=op, rdx/r10/r8/r9=args), dispatch,
// restore rax(result)/rcx(rip)/r11(rflags), zero the other scratch regs (no kernel leak to ring 3),
// restore the user RSP, and `sysretq`. IF is masked (SFMASK), so this runs non-reentrantly.
global_asm!(
    r#"
.section .text
.global praesidium_x86_syscall_entry
praesidium_x86_syscall_entry:
    mov [rip + {user_rsp}], rsp        // TRANSIENT stash of the user rsp (moved into the frame below;
                                       // no yield happens before that + IF is masked, so this global
                                       // is safe as a single-entry transient even with 2 processes)
    mov rsp, [rip + {kernel_rsp}]      // switch to the CURRENT task's kernel stack (set per-task by
                                       // the scheduler on switch-in, so two processes never collide)
    and rsp, -16                       // 16-align (frame is a 16-multiple -> aligned at the call)
    sub rsp, {frame}
    mov [rsp + 0x00], rax              // selector (also the result slot)
    mov [rsp + 0x08], rdi              // cptr
    mov [rsp + 0x10], rsi              // op
    mov [rsp + 0x18], rdx              // args[0]
    mov [rsp + 0x20], r10              // args[1]
    mov [rsp + 0x28], r8               // args[2]
    mov [rsp + 0x30], r9               // args[3]
    mov [rsp + 0x38], rcx              // user rip  (for sysretq)
    mov [rsp + 0x40], r11              // user rflags (for sysretq)
    mov rax, [rip + {user_rsp}]        // reload the transiently-stashed user rsp
    mov [rsp + 0x48], rax              // save it IN THE FRAME (per-process; survives a yield to a peer)
    mov rdi, rsp                       // frame ptr -> handler arg
    call {handler}
    mov rax, [rsp + 0x00]              // result -> rax
    mov rcx, [rsp + 0x38]              // user rip
    mov r11, [rsp + 0x40]              // user rflags
    xor edx, edx                       // zero the scratch regs we do not restore (no kernel leak)
    xor esi, esi
    xor edi, edi
    xor r8d, r8d
    xor r9d, r9d
    xor r10d, r10d
    mov rsp, [rsp + 0x48]              // restore user rsp FROM THE FRAME (per-process; last frame read)
    sysretq                            // -> ring 3: rip=rcx, rflags=r11, cs/ss (RPL 3) from STAR
"#,
    user_rsp = sym USER_RSP,
    kernel_rsp = sym KERNEL_RSP,
    frame = const SYSCALL_FRAME_BYTES,
    handler = sym x86_syscall_handler,
);

/// Decode the syscall register frame the LSTAR stub spilled and dispatch through the generic
/// policy. Frame slots: 0=selector, 1=cptr, 2=op, 3..3+MSG_REGS=args (then user rip/rflags, which
/// the stub restores itself). Writes the result word into slot 0. Mirrors the aarch64 handler.
extern "C" fn x86_syscall_handler(frame: *mut u64) {
    // SAFETY: `frame` is the stub's spilled register frame; slots 0..3+MSG_REGS are in range.
    let sel = unsafe { frame.add(0).read() };
    if sel != sys::INVOKE {
        crate::user::fault("bad syscall selector", sel); // -> ! (kills the process)
    }
    let inv = unsafe {
        let mut args = [0u64; MSG_REGS];
        let mut i = 0;
        while i < MSG_REGS {
            args[i] = frame.add(3 + i).read();
            i += 1;
        }
        Invocation {
            cptr: frame.add(1).read() as u32,
            op: frame.add(2).read() as u16,
            args,
        }
    };
    let result = crate::user::syscall(&inv);
    // SAFETY: write the result into the selector/result slot; the stub loads it into rax.
    unsafe { frame.add(0).write(result) };
}

/// The P7a bring-up user program (native x86-64): capability-mediated `DEBUG_EMIT(0xBEEF)` then
/// `PROC_EXIT(0)`, each a `syscall` carrying an [`Invocation`] on the one Endpoint the process holds
/// (rax=selector, rdi=cptr, rsi=op, rdx=arg). Mirrors the aarch64 blob; replaced by `refproc` (P7b).
#[must_use]
pub fn el0_test_blob() -> &'static [u8] {
    let start = core::ptr::addr_of!(praesidium_x86_user_blob);
    let end = core::ptr::addr_of!(praesidium_x86_user_blob_end);
    // SAFETY: both symbols bound the same contiguous, immutable `.rodata` blob emitted below.
    unsafe { core::slice::from_raw_parts(start, end as usize - start as usize) }
}

/// The P7a fault bring-up program (native x86-64): a raw read of the supervisor-only `FAULT_PROBE_VA`
/// page from ring 3 — a #PF (U/S set) the kernel contains by killing the process, not itself.
#[must_use]
pub fn el0_fault_blob() -> &'static [u8] {
    let start = core::ptr::addr_of!(praesidium_x86_fault_blob);
    let end = core::ptr::addr_of!(praesidium_x86_fault_blob_end);
    // SAFETY: both symbols bound the same contiguous, immutable `.rodata` blob emitted below.
    unsafe { core::slice::from_raw_parts(start, end as usize - start as usize) }
}

global_asm!(
    r#"
.section .rodata
.balign 16
.global praesidium_x86_user_blob
.global praesidium_x86_user_blob_end
praesidium_x86_user_blob:
    mov  rax, {invoke}      // sys::INVOKE (the only syscall selector)
    mov  rdi, {ep}          // cptr = the one Endpoint capability the process holds
    mov  rsi, {debug}       // op = DEBUG_EMIT
    mov  edx, 0xBEEF        // args[0] = the value to emit
    syscall
    mov  rax, {invoke}      // sys::INVOKE
    mov  rdi, {ep}          // cptr = same Endpoint
    mov  rsi, {exit}        // op = PROC_EXIT
    xor  edx, edx           // args[0] = exit code 0
    syscall
    ud2                     // unreachable (PROC_EXIT does not return to ring 3)
praesidium_x86_user_blob_end:
.global praesidium_x86_fault_blob
.global praesidium_x86_fault_blob_end
praesidium_x86_fault_blob:
    mov  rdi, {probe}       // FAULT_PROBE_VA (a supervisor-only page)
    mov  al, [rdi]          // ring-3 read of a supervisor page -> #PF (U/S set)
    ud2                     // unreachable: the load traps; the kernel kills the process
praesidium_x86_fault_blob_end:
"#,
    invoke = const abi::invoke::sys::INVOKE,
    ep = const crate::user::EP_SLOT as u64,
    debug = const abi::invoke::op::DEBUG_EMIT as u64,
    exit = const abi::invoke::op::PROC_EXIT as u64,
    probe = const crate::user::FAULT_PROBE_VA,
);
