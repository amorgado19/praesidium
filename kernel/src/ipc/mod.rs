//! Synchronous capability IPC integration (P4).
//!
//! Wires the pure `ipc` crate (Endpoint rendezvous + message format) and `cap-core`'s
//! cross-CSpace GRANT + single-use Reply to the P3 async executor, and runs the P4 boot demo
//! asserting the gates:
//!  - **AC4.1** — a sync `call`/`reply` round-trips a register message between two async tasks.
//!  - **AC4.2** — the `Reply` is single-use: a second reply fails (CAP-REPLY-1).
//!  - **AC4.3** — the callee runs on the *caller's* `Sched` budget (passive server, verified by
//!    accounting: the caller pays for the server's work; the server has no budget of its own).
//!  - **AC4.4** — the fast path performs **no address-space swap** (CR3 / TTBR unchanged across
//!    the call) — the SASOS win, measured.
//!  - **AC4.5** — GRANT moves a capability into the receiver's CSpace with monotonic (narrowed)
//!    rights.
//!
//! IPC participants are **async futures** on the executor (architect ruling): `call().await`
//! registers a waker and yields when the receiver isn't ready (the tested P3a block pattern),
//! and the "continuation" is the executor polling the peer's future — no address-space switch,
//! so AC4.4 holds by construction. Cross-CSpace cap transfer + Reply consume run **preemption-
//! masked** (DEC-0003-7), so a concurrent REVOKE can never observe a half-transferred cap. No
//! capability is fabricated here — every mutation goes through `cap-core` (CAP-RUST-1).

use alloc::collections::BTreeMap;
use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll, Waker};

use cap_core::{grant, Budget, CSpace, CapError, CapType, GrantMode, Rights};
use ipc::{CapXfer, EndpointState, Message, PartyId, RecvOutcome, SendOutcome};
use mem::frame::{pfn_to_phys, phys_to_pfn};
use sched::{Executor, Task};

use crate::sync::SpinLock;
use crate::{arch, memory};

/// Slots per demo CSpace.
const SLOTS: usize = 16;
/// Party ids (the executor maps these to task wakers via the `wakers` table).
const CLIENT: PartyId = 1;
const SERVER: PartyId = 2;
/// Client CSpace layout.
const CLIENT_UNTYPED: usize = 0;
const CLIENT_FRAME: usize = 1;
const CLIENT_EP: usize = 2; // the client's Endpoint cap (SEND) — authority to call
/// Server CSpace layout: its Endpoint cap (RECV), granted caps at RECV_BASE.., the Reply at REPLY_SLOT.
const SERVER_EP: usize = 1; // the server's Endpoint cap (RECV) — authority to receive
const SERVER_RECV_BASE: usize = 5;
const SERVER_REPLY_SLOT: usize = 10;
/// Passive-server accounting.
const CLIENT_BUDGET: u32 = 100;
const CALL_DELEGATE: u32 = 50; // budget the caller lends the server for the call
const WORK_UNITS: u32 = 30; // CPU-time the server spends servicing the request
/// The demo request/response payload word + method label.
const REQ_WORD: u64 = 0x00C0_FFEE;
const LABEL: u32 = 0xABCD;

/// Zeroing hook for the demo CSpaces (frames zeroed through the HHDM — CAP-MEM-2/CAP-REVOKE-2).
fn zero_frames(frame: u64, frames: u32) {
    for i in 0..u64::from(frames) {
        memory::zero_frame(pfn_to_phys((frame + i) as u32));
    }
}

fn fatal(msg: &str) -> ! {
    kprintln!("[praesidium] FATAL: ipc: {msg}");
    crate::arch::halt();
}
fn fatal_err(op: &str, e: CapError) -> ! {
    kprintln!("[praesidium] FATAL: ipc: {op} failed: {e:?}");
    crate::arch::halt();
}

/// The kernel-side IPC state a demo endpoint's participants share. Held behind a `SpinLock`;
/// every future locks it only for a bounded, `.await`-free critical section (GC-07).
struct IpcWorld {
    endpoint: EndpointState,
    client_cs: CSpace<SLOTS>,
    server_cs: CSpace<SLOTS>,
    /// Caller's budget; the server runs on a delegated slice of it (passive server).
    client_budget: Budget,
    /// The server's own budget — deliberately zero: it runs only on delegated CPU-time.
    server_budget: Budget,
    /// A blocked party's executor waker, keyed by party id.
    wakers: BTreeMap<PartyId, Waker>,
    /// A request delivered to a blocked receiver: receiver → (sender, message).
    delivered: BTreeMap<PartyId, (PartyId, Message)>,
    /// A response delivered to a blocked caller: caller → response.
    responses: BTreeMap<PartyId, Message>,
    /// The caller's received response, captured for the AC4.1 assertion.
    client_result: Option<Message>,
}

impl IpcWorld {
    /// A receiver obtained a request: mint the single-use Reply naming `sender`, execute the
    /// message's capability transfers (GRANT sender→receiver, preemption-masked), and hand back
    /// the request plus the reply-cap slot. Returns `(request, reply_slot)`.
    fn on_request(&mut self, sender: PartyId, mut msg: Message) -> (Message, usize) {
        msg.badge = sender; // stamp the caller's identity onto the delivered message
        self.server_cs
            .mint_reply(SERVER_REPLY_SLOT, sender, msg.badge)
            .unwrap_or_else(|e| fatal_err("mint reply", e));
        for (i, x) in msg.transfers().iter().enumerate() {
            // Preemption-masked (DEC-0003-7): the cross-CSpace transfer and any concurrent REVOKE
            // are serialized on this single CPU, so no half-transferred cap is ever observable
            // (CAP-REVOKE-1 in-flight atomicity).
            let prev = arch::preempt_disable();
            let r = grant(
                &mut self.client_cs,
                x.src,
                &mut self.server_cs,
                SERVER_RECV_BASE + i,
                x.rights,
                msg.badge,
                x.mode,
            );
            arch::preempt_restore(prev);
            r.unwrap_or_else(|e| fatal_err("grant over ipc", e));
        }
        (msg, SERVER_REPLY_SLOT)
    }

    /// The server replies: consume the single-use Reply (CAP-REPLY-1), deliver `resp` to the
    /// caller it named, and wake the caller. Returns whether a reply was actually sent (a second
    /// reply on a consumed cap returns `false` — the slot is already empty).
    fn do_reply(&mut self, reply_slot: usize, resp: Message) -> bool {
        let prev = arch::preempt_disable();
        let consumed = self.server_cs.consume_reply(reply_slot);
        arch::preempt_restore(prev);
        match consumed {
            Ok(caller) => {
                self.responses.insert(caller, resp);
                if let Some(wk) = self.wakers.remove(&caller) {
                    wk.wake();
                }
                true
            }
            Err(_) => false,
        }
    }
}

/// The demo IPC world, `None` until [`run`]. A boot-time singleton (a single endpoint / one
/// client + one server); real per-process CSpaces + endpoints arrive with userspace (P7).
static WORLD: SpinLock<Option<IpcWorld>> = SpinLock::new(None);

/// Run `f` with the IPC world locked (bounded, `.await`-free — never held across a yield, GC-07).
fn with_world<R>(f: impl FnOnce(&mut IpcWorld) -> R) -> R {
    let mut g = WORLD.lock();
    f(g.as_mut()
        .unwrap_or_else(|| fatal("ipc world used before init")))
}

// ---- the sync-looking / async-underneath call & recv (DEC-0004-6) ----

/// A synchronous `call`: send `msg` on the endpoint and await the reply. If the receiver isn't
/// ready the future registers its waker and yields (`Pending`); the executor runs other work.
struct Call {
    party: PartyId,
    msg: Option<Message>,
    sent: bool,
}
fn call(party: PartyId, msg: Message) -> Call {
    Call {
        party,
        msg: Some(msg),
        sent: false,
    }
}
impl Future for Call {
    type Output = Message;
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Message> {
        with_world(|w| {
            if !self.sent {
                self.sent = true;
                let msg = self
                    .msg
                    .take()
                    .unwrap_or_else(|| fatal("call polled with no message"));
                if let SendOutcome::Delivered { receiver, msg } = w.endpoint.send(self.party, msg) {
                    // A receiver was already waiting: stash the request for it and wake it.
                    w.delivered.insert(receiver, (self.party, msg));
                    if let Some(wk) = w.wakers.remove(&receiver) {
                        wk.wake();
                    }
                }
                // Either way we now block awaiting the reply.
                w.wakers.insert(self.party, cx.waker().clone());
                return Poll::Pending;
            }
            match w.responses.remove(&self.party) {
                Some(resp) => Poll::Ready(resp),
                None => {
                    w.wakers.insert(self.party, cx.waker().clone());
                    Poll::Pending
                }
            }
        })
    }
}

/// A `recv`: wait for a request. On a match it returns the request plus the slot of the freshly-
/// minted single-use Reply cap (naming the caller) the server uses to answer.
struct Recv {
    party: PartyId,
    tried: bool,
}
fn recv(party: PartyId) -> Recv {
    Recv {
        party,
        tried: false,
    }
}
impl Future for Recv {
    type Output = (Message, usize);
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<(Message, usize)> {
        with_world(|w| {
            // A request may have been delivered to us while we were parked.
            if let Some((sender, msg)) = w.delivered.remove(&self.party) {
                return Poll::Ready(w.on_request(sender, msg));
            }
            if !self.tried {
                self.tried = true;
                if let RecvOutcome::Received { sender, msg } = w.endpoint.recv(self.party) {
                    return Poll::Ready(w.on_request(sender, msg));
                }
            }
            w.wakers.insert(self.party, cx.waker().clone());
            Poll::Pending
        })
    }
}

// ---- the demo participants ----

/// The no-ambient-authority gate (SPEC-CAP RI): verify the party holds an `Endpoint` capability
/// carrying `right` before it may send/recv. Fails the boot closed if the authority is missing.
fn authorize_endpoint(cs: &CSpace<SLOTS>, ep_slot: usize, right: Rights, who: &str) {
    let ep = cs.resolve(ep_slot).unwrap_or_else(|e| fatal_err(who, e));
    if ep.cap_type != CapType::Endpoint || !ep.rights.contains(right) {
        fatal("IPC refused — party holds no Endpoint capability with the required right (RI)");
    }
}

/// The client: lend the server budget, `call` with a data word + a GRANT'd Frame, keep the reply.
async fn client_body() {
    // No ambient authority: the client may call only because it holds an Endpoint cap with SEND.
    with_world(|w| {
        authorize_endpoint(
            &w.client_cs,
            CLIENT_EP,
            Rights::SEND,
            "client endpoint SEND",
        )
    });
    with_world(|w| {
        let _ = w
            .client_budget
            .delegate(&mut w.server_budget, CALL_DELEGATE);
    });
    let mut msg = Message::with_data(LABEL, &[REQ_WORD]);
    msg.info.n_caps = 1;
    msg.caps[0] = CapXfer {
        src: CLIENT_FRAME,
        rights: Rights::READ,
        mode: GrantMode::Move,
    };
    let resp = call(CLIENT, msg).await;
    with_world(|w| w.client_result = Some(resp));
}

/// The server: `recv`, spend `WORK_UNITS` of the delegated budget, `reply`, return the unused
/// budget to the caller.
async fn server_body() {
    // No ambient authority: the server may receive only because it holds an Endpoint cap with RECV
    // (the RECV-only cap the client GRANTed it — narrower than the client's own SEND/RECV cap).
    with_world(|w| {
        authorize_endpoint(
            &w.server_cs,
            SERVER_EP,
            Rights::RECV,
            "server endpoint RECV",
        )
    });
    let (req, reply_slot) = recv(SERVER).await;
    let resp = Message::with_data(req.info.label, &[req.data()[0].wrapping_add(1)]);
    with_world(|w| {
        w.server_budget.charge(WORK_UNITS); // charged to the caller's delegated budget
        w.do_reply(reply_slot, resp);
        let unused = w.server_budget.remaining();
        let _ = w.server_budget.delegate(&mut w.client_budget, unused); // hand the rest back
    });
}

/// Run P4: stand up a client + server, round-trip a capability-carrying call/reply on the
/// cooperative executor, and assert AC4.1–AC4.5. Prints `PRAESIDIUM-P4-OK` on success.
pub fn run() {
    // Build the client CSpace with an Untyped it retypes a GRANT-able Frame from.
    let phys = memory::alloc_frames(4).unwrap_or_else(|| fatal("no frames for the ipc untyped"));
    let base = u64::from(phys_to_pfn(phys));
    let mut client_cs = CSpace::<SLOTS>::new(zero_frames);
    client_cs.set_root_untyped(base, 16);
    client_cs
        .retype(CLIENT_UNTYPED, CapType::Frame, 1, 1, CLIENT_FRAME)
        .unwrap_or_else(|e| fatal_err("retype client frame", e));
    // Retype the Endpoint the two parties rendezvous on: the client owns it (SEND/RECV/GRANT), so
    // it can call — and it GRANTs the server a RECV-only cap to the SAME endpoint. Neither party
    // reaches the rendezvous except through a capability (SPEC-CAP RI — no ambient message bus).
    client_cs
        .retype(CLIENT_UNTYPED, CapType::Endpoint, 1, 1, CLIENT_EP)
        .unwrap_or_else(|e| fatal_err("retype endpoint", e));
    let frame_objref = client_cs
        .resolve(CLIENT_FRAME)
        .unwrap_or_else(|e| fatal_err("resolve client frame", e))
        .objref;
    let mut server_cs = CSpace::<SLOTS>::new(zero_frames);
    cap_core::grant(
        &mut client_cs,
        CLIENT_EP,
        &mut server_cs,
        SERVER_EP,
        Rights::RECV, // the server may only RECV on it (monotonic — narrower than the client's cap)
        0,
        GrantMode::Mint, // the client keeps its own SEND-capable endpoint cap
    )
    .unwrap_or_else(|e| fatal_err("grant endpoint RECV cap to server", e));

    *WORLD.lock() = Some(IpcWorld {
        endpoint: EndpointState::new(),
        client_cs,
        server_cs,
        client_budget: Budget::new(CLIENT_BUDGET, CLIENT_BUDGET),
        server_budget: Budget::new(0, CLIENT_BUDGET), // NO budget of its own
        wakers: BTreeMap::new(),
        delivered: BTreeMap::new(),
        responses: BTreeMap::new(),
        client_result: None,
    });

    kprintln!(
        "[praesidium] ipc: endpoint retyped; client holds Endpoint SEND, server GRANTed Endpoint RECV — rendezvous is cap-gated (no ambient authority, RI)"
    );

    // Snapshot the address-space root, run the call/reply on the executor, snapshot again (AC4.4).
    let root_before = arch::read_translation_root();
    let mut ex = Executor::new();
    ex.spawn(Task::new(client_body(), Budget::new(1000, 1000)));
    ex.spawn(Task::new(server_body(), Budget::new(1000, 1000)));
    ex.run_until_idle();
    let root_after = arch::read_translation_root();

    with_world(|w| {
        // AC4.1 — the call round-tripped: the reply carries `REQ_WORD + 1`.
        let resp = w
            .client_result
            .as_ref()
            .unwrap_or_else(|| fatal("call did not round-trip — no reply received"));
        if resp.data() != [REQ_WORD.wrapping_add(1)] {
            fatal("reply payload is wrong");
        }
        kprintln!(
            "[praesidium] ipc: call/reply round-trip ok — sent {:#x}, got {:#x} (AC4.1)",
            REQ_WORD,
            resp.data()[0]
        );

        // AC4.5 — the GRANT'd Frame moved into the server's CSpace, narrowed to READ, badged.
        let g = w
            .server_cs
            .resolve(SERVER_RECV_BASE)
            .unwrap_or_else(|e| fatal_err("granted frame missing from server", e));
        if g.cap_type != CapType::Frame || g.objref != frame_objref || g.rights != Rights::READ {
            fatal("GRANT did not move the frame with monotonic (narrowed) rights");
        }
        if w.client_cs.resolve(CLIENT_FRAME).is_ok() {
            fatal("MOVE-grant left the frame in the client's CSpace");
        }
        kprintln!(
            "[praesidium] ipc: GRANT moved Frame {:#x} into server (rights R, badge {:#x}); client no longer holds it (AC4.5)",
            g.objref,
            g.badge
        );

        // AC4.2 — the Reply was single-use: a second reply on the same cap fails cleanly.
        if w.server_cs.consume_reply(SERVER_REPLY_SLOT) != Err(CapError::EmptySlot) {
            fatal("Reply was not single-use — a second reply did not fail (CAP-REPLY-1)");
        }
        kprintln!(
            "[praesidium] ipc: second reply on the consumed Reply cap failed cleanly (AC4.2)"
        );

        // AC4.3 — passive server: the caller paid for the server's work; the server had no budget.
        if w.client_budget.remaining() != CLIENT_BUDGET - WORK_UNITS {
            fatal("passive-server accounting wrong — caller did not pay for the server's work");
        }
        kprintln!(
            "[praesidium] ipc: passive server ran on caller budget — caller paid {WORK_UNITS} units ({} left of {CLIENT_BUDGET}); server had none (AC4.3)",
            w.client_budget.remaining()
        );
    });

    // AC4.4 — the fast path performed no address-space swap (SASOS: one address space).
    if root_before != root_after {
        fatal("fast path swapped the address-space root — not a SASOS continuation (AC4.4)");
    }
    kprintln!(
        "[praesidium] ipc: address-space root unchanged across the call ({:#x}) — no page-table swap (AC4.4)",
        root_after[0]
    );

    kprintln!("[praesidium] PRAESIDIUM-P4-OK");
}
