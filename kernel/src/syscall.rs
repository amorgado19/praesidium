//! The capability-invocation dispatch (ADR-0006 DEC-0006-1..3) — the kernel's entire external
//! operation surface as ONE small primitive: resolve the invoked `cptr` against the caller's
//! CSpace, check the operation's required right, then perform it. The Root Invariant is enforced
//! *here*, at the boundary: no cptr ⇒ no object; no right ⇒ refused (fail closed).
//!
//! [`invoke`] is deliberately shaped as **"what an SVC/`syscall` handler will call"**: it takes a
//! caller CSpace + a decoded [`Invocation`] and returns a result, with no dependency on how the
//! invocation arrived. P6 exercises it in-kernel (called directly); P7 adds the EL0/ring-3 trap
//! that decodes the arch registers into an [`Invocation`] and calls this — a thin transport over a
//! dispatch already proven here, not a rewrite (that boundary is *built* in P6, *fired* in P7).
//!
//! Syscall and IPC are the same machinery (DEC-0006-3): an invocation on an `Endpoint` cap is an
//! IPC send; an invocation on any other kernel-object cap is a "syscall". One path, one audit.

use abi::invoke::{op, Invocation, InvokeError};
use cap_core::{CSpace, CapError, CapType, Rights};

/// Dispatch a capability invocation against `cs` (the caller's CSpace). Returns the operation's
/// single result word, or a typed [`InvokeError`] — every refusal is a capability check failing
/// closed, never an ambient-policy denial.
pub fn invoke<const N: usize>(cs: &CSpace<N>, inv: &Invocation) -> Result<u64, InvokeError> {
    // Resolve the cptr against the caller's own CSpace (SPEC-CAP §3). Out-of-range vs empty are
    // distinguished so a caller can tell "no such slot" from "you hold nothing there".
    let cap = cs.resolve(inv.cptr as usize).map_err(|e| match e {
        CapError::OutOfBounds => InvokeError::BadCptr,
        _ => InvokeError::EmptySlot,
    })?;

    match inv.op {
        // Pure query: prove cptr resolution. Requires only a non-empty slot (already resolved).
        op::CAP_IDENTIFY => {
            Ok((u64::from(cap.cap_type as u8) << 32) | u64::from(cap.rights.bits()))
        }

        // Requires the READ right on a Frame — the rights-check demonstration (AC6.1).
        op::FRAME_PROBE => {
            if cap.cap_type != CapType::Frame {
                return Err(InvokeError::WrongType);
            }
            if !cap.rights.contains(Rights::READ) {
                return Err(InvokeError::InsufficientRights);
            }
            Ok(cap.objref)
        }

        // Requires the SEND right on an Endpoint — the syscall/IPC unification (DEC-0006-3). The
        // rendezvous itself is the P4 IPC path; here we prove the *same* invoke dispatch resolves
        // and rights-checks an Endpoint invocation. P7 wires this to real cross-process IPC.
        op::ENDPOINT_SEND => {
            if cap.cap_type != CapType::Endpoint {
                return Err(InvokeError::WrongType);
            }
            if !cap.rights.contains(Rights::SEND) {
                return Err(InvokeError::InsufficientRights);
            }
            Ok(cap.badge) // acknowledge the routed send (badge identifies the sender to the server)
        }

        // Bring-up (P7a): DEBUG_EMIT / PROC_EXIT are capability-gated process affordances modelled
        // as sends to the process's bring-up-service Endpoint (DEC-0006-3) — each REQUIRES `SEND`
        // on an `Endpoint`, so an EL0 process with no such capability reaches neither the console
        // nor a clean exit (RI: no ambient authority). The dispatch only resolves + rights-checks
        // and returns the argument word; the *effect* (log / terminate) is performed by the caller,
        // since a divergent exit cannot be expressed through this value-returning dispatch.
        op::DEBUG_EMIT | op::PROC_EXIT => {
            if cap.cap_type != CapType::Endpoint {
                return Err(InvokeError::WrongType);
            }
            if !cap.rights.contains(Rights::SEND) {
                return Err(InvokeError::InsufficientRights);
            }
            Ok(inv.args[0])
        }

        // Cross-process IPC rights-check (P7b AC7.2): CALL needs SEND, RECV needs RECV on an
        // Endpoint. The rendezvous EFFECT (block/deliver/reply) is the caller's (kernel::user),
        // just as DEBUG_EMIT's console write is — this dispatch only resolves + rights-checks (RI).
        op::ENDPOINT_CALL => {
            if cap.cap_type != CapType::Endpoint {
                return Err(InvokeError::WrongType);
            }
            if !cap.rights.contains(Rights::SEND) {
                return Err(InvokeError::InsufficientRights);
            }
            Ok(cap.badge)
        }
        op::ENDPOINT_RECV => {
            if cap.cap_type != CapType::Endpoint {
                return Err(InvokeError::WrongType);
            }
            if !cap.rights.contains(Rights::RECV) {
                return Err(InvokeError::InsufficientRights);
            }
            Ok(cap.badge)
        }

        // Shared read-only transfer region (v1.1): a `SharedRo` cap (**requires `READ`**) returns
        // the read-only VA the kernel co-mapped its region at (RI: the holder learns its window from
        // the cap, never ambiently). No map operation — the co-mapping was done privileged at
        // share-time. The `aux` field packs the VA as `va >> 12`.
        op::SHARED_QUERY => {
            if cap.cap_type != CapType::SharedRo {
                return Err(InvokeError::WrongType);
            }
            if !cap.rights.contains(Rights::READ) {
                return Err(InvokeError::InsufficientRights);
            }
            Ok(u64::from(cap.aux) << 12)
        }

        _ => Err(InvokeError::UnknownOp),
    }
}
