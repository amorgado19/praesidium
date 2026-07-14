//! The capability-invocation syscall ABI (ADR-0006 DEC-0006-1..4).
//!
//! Every syscall is an **invocation on a capability**: "invoke *this operation* on *this
//! capability* with *these arguments*." There is no `open("/path")` — a caller invokes an
//! operation on a `cptr` (an index into its own CSpace, SPEC-CAP §3), never a raw kernel address.
//! The kernel resolves the cptr against the caller's CSpace and checks the cap's rights before
//! acting, so the Root Invariant is enforced at the boundary. This is deliberately the same
//! machinery as the IPC fast path (ADR-0004): an IPC call is just an invocation on an `Endpoint`
//! cap, so the message registers are shared (DEC-0006-3).
//!
//! This module is the **stable wire contract**; the kernel-side *dispatch* (resolve cptr → check
//! rights → perform op) lives in `kernel::syscall` and takes an [`Invocation`] by value — shaped
//! as "what an SVC/`syscall` handler will call." P6 exercises that dispatch in-kernel; P7 wraps it
//! in the actual EL0/ring-3 trap (the transport), so the trap is a thin shell over a
//! proven-in-P6 dispatch rather than a rewrite.

/// Number of message registers carrying an invocation's arguments — shared with the IPC fast
/// path (ADR-0004). A register-only payload keeps the boundary hot and nothing to bulk-validate.
pub const MSG_REGS: usize = 4;

/// The syscall selector carried in the syscall-number register at the trap boundary (aarch64 `x8`
/// / x86-64 `rax`). There is exactly one — `INVOKE` — because every syscall is a capability
/// invocation (DEC-0006-3): `INVOKE` carries an [`Invocation`] in the remaining argument registers.
/// This is the arch-generic contract; the concrete register assignment is arch-specific (behind the
/// ADR-0007 seam).
pub mod sys {
    /// Perform a capability invocation (the argument registers carry the [`super::Invocation`]).
    /// The ONLY syscall selector: every syscall is an invocation on a capability, so there is no
    /// ambient "debug"/"exit" syscall — those are *operations* ([`super::op::DEBUG_EMIT`] /
    /// [`super::op::PROC_EXIT`]) invoked on a capability the process holds, enforcing the Root
    /// Invariant (no ambient authority) at the trap boundary.
    pub const INVOKE: u64 = 0;
}

/// Operation selectors. Stable wire values (they cross the user/kernel boundary). Kept tiny in
/// P6 — enough to prove cptr resolution + rights-checked dispatch + the IPC unification.
pub mod op {
    /// Return the resolved capability's `(type, rights)` — proves cptr resolution. Requires only
    /// that the slot is non-empty (a pure query).
    pub const CAP_IDENTIFY: u16 = 1;
    /// Probe a `Frame` capability — **requires the `READ` right**; returns the frame's object
    /// reference. A `Frame` cap lacking `READ` is refused (the rights-check demonstration).
    pub const FRAME_PROBE: u16 = 2;
    /// Send on an `Endpoint` capability — **requires the `SEND` right**; routes into the IPC
    /// machinery (ADR-0004). This is a syscall *and* an IPC call through one path (DEC-0006-3).
    pub const ENDPOINT_SEND: u16 = 3;

    /// **Bring-up only (P7a).** Emit the first argument register to the kernel's serial console — a
    /// capability-gated debug affordance modelled as a send to the process's bring-up-service
    /// Endpoint (DEC-0006-3): **requires the `SEND` right on an `Endpoint`**, so an EL0 process
    /// holding no such capability cannot reach the console (no ambient authority). Retired when a
    /// real console/log capability arrives (post-v1).
    pub const DEBUG_EMIT: u16 = 0x10;
    /// Terminate the calling process with the exit code in the first argument register — modelled
    /// as a send to the process's bring-up-service Endpoint (**requires `SEND` on an `Endpoint`**).
    /// The real self-termination authority (invoking the process's own `Task`/`Sched` cap) is P7b.
    pub const PROC_EXIT: u16 = 0x11;

    /// **Cross-process IPC (P7b, AC7.2).** CALL on an `Endpoint` (**requires `SEND`**): send the
    /// first argument register to whoever holds a RECV cap to the same Endpoint, block for a
    /// one-word reply, and return it. The synchronous call/reply of ADR-0004, over the shared
    /// Endpoint capability — cross-process, no address-space swap (SASOS).
    pub const ENDPOINT_CALL: u16 = 0x20;
    /// RECV on an `Endpoint` (**requires `RECV`**): block until a caller's [`ENDPOINT_CALL`] arrives,
    /// return its message word, and receive a single-use `Reply` capability (minted at a fixed slot
    /// the runtime knows) for the ensuing [`ENDPOINT_REPLY`].
    pub const ENDPOINT_RECV: u16 = 0x21;
    /// REPLY on the single-use `Reply` capability RECV minted (**consumes it**, CAP-REPLY-1): deliver
    /// the first argument register to the one blocked caller it names, unblocking it.
    pub const ENDPOINT_REPLY: u16 = 0x22;

    /// **Shared read-only transfer region (v1.1, ADR-0004).** Query a `SharedRo` capability
    /// (**requires `READ`**): return the read-only virtual address the kernel co-mapped the shared
    /// region at, so the holder reads the shared bulk data zero-copy *through the capability* (RI —
    /// it learns the window VA from the cap, not ambiently). No map operation is exposed; the
    /// co-mapping was done, privileged, at share-time — userspace never edits a page table.
    pub const SHARED_QUERY: u16 = 0x30;
}

/// A capability invocation: perform `op` on the capability at `cptr`, with `args`. The concrete
/// register encoding is arch-specific and lives behind the ADR-0007 seam (DEC-0006-4); this is
/// the arch-generic semantic form the dispatch consumes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Invocation {
    /// Index into the caller's CSpace of the capability being invoked (SPEC-CAP §3).
    pub cptr: u32,
    /// The operation selector (`op::*`).
    pub op: u16,
    /// The argument message registers.
    pub args: [u64; MSG_REGS],
}

impl Invocation {
    /// A no-argument invocation of `op` on `cptr`.
    #[must_use]
    pub fn new(cptr: u32, op: u16) -> Self {
        Self {
            cptr,
            op,
            args: [0; MSG_REGS],
        }
    }
}

/// Why an invocation was refused. Stable wire values — userspace observes these across the
/// boundary, so the discriminants are fixed. Every refusal is a capability check failing closed;
/// there is no "operation not permitted by ambient policy," only "you do not hold that authority."
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(i32)]
pub enum InvokeError {
    /// `cptr` is outside the caller's CSpace.
    BadCptr = 1,
    /// `cptr` names an empty slot — the caller holds no capability there.
    EmptySlot = 2,
    /// The operation is not defined for the capability's type.
    WrongType = 3,
    /// The capability does not carry the right the operation requires (RI: no authority).
    InsufficientRights = 4,
    /// Unknown operation selector.
    UnknownOp = 5,
}
