//! SASOS isolation backstop (P5, ADR-0008 / CAP-MEM-3) — **the existential phase**.
//!
//! Single-address-space means capabilities govern *nameability*, but a raw pointer can still
//! touch any mapped byte; the isolation backstop is what makes isolation real rather than
//! theatre (CLAUDE.md: "red-team it hardest"). P5a stands up the **host-verifiable foundation**
//! of ADR-0008's three defensive layers and asserts each at boot:
//!
//!  - **Layer 1 (AC5.1) — compile-time nameability.** `Cap<T>` is unconstructable outside
//!    `cap-core` (its fields are private and its sole `unsafe fn fabricate` is `pub(crate)`), so
//!    a component with no capability to a region cannot even *name* it. This is enforced by the
//!    type system and regression-guarded off-target by `cap-core`'s `tests/ui` compile-fail
//!    fixture — there is nothing to run here, only to record.
//!  - **Layer 3 (AC5.3) — universal guard pages + W^X + zero.** [`arch::install_guard_page`]
//!    renders the frame below every task stack non-present (closing the P3b/P4 deferral); W^X
//!    stays in force on the live tables; freshly (re)allocated frames read back zero (CAP-MEM-2).
//!  - **Domain entry (DEC-0008-5).** Running *in* a protection domain is capability-gated via a
//!    thin `Domain` cap (ENTER only) — never ambient. Layer 2's actual hardware enforcement (MTE
//!    on aarch64 / fallback on x86) and the adversarial escape test land in P5b.
//!
//! No capability is fabricated here: every `Cap`/`CSpace` operation goes through `cap-core`
//! (CAP-RUST-1). Any violation fails the boot loudly (`FATAL`), never silently.

use cap_core::{CSpace, CapError};

use crate::{arch, memory};

/// 4 KiB page, as a `u64` for HHDM address arithmetic.
const PAGE: u64 = 4096;

/// Slots in the tiny demo CSpace used for the domain-entry gating check.
const SLOTS: usize = 8;
/// Slot 0 holds a (non-Domain) `Untyped` — broad authority that still must NOT authorize entry.
const NON_DOMAIN_SLOT: usize = 0;
/// Slot 1 holds the `Domain` cap (ENTER) that authorizes entering `DEMO_DOMAIN_ID`.
const DOMAIN_SLOT: usize = 1;
/// Slot 2 is deliberately empty — entry through it must be refused.
const EMPTY_SLOT: usize = 2;
/// An arbitrary opaque domain id for the demo (a real id names an MTE tag / key / page-table root).
const DEMO_DOMAIN_ID: u64 = 0xD0;

fn fatal(msg: &str) -> ! {
    kprintln!("[praesidium] FATAL: isolation: {msg}");
    arch::halt();
}

fn fatal_prot(msg: &str, got: Option<(bool, bool)>) -> ! {
    kprintln!("[praesidium] FATAL: isolation: {msg} (got {got:?})");
    arch::halt();
}

fn fatal_enter(msg: &str, got: Result<u64, CapError>) -> ! {
    kprintln!("[praesidium] FATAL: isolation: {msg} (got {got:?})");
    arch::halt();
}

/// Zeroing hook for the demo CSpace. It never RETYPEs or destroys an object, so the hook is
/// never invoked; a no-op keeps the demo self-contained.
fn no_op_zero(_frame: u64, _frames: u32) {}

/// Run the P5a isolation-foundation boot checks, emitting `PRAESIDIUM-P5A-OK` on success.
pub fn run() {
    kprintln!("[praesidium] isolation: P5a foundation — SASOS backstop (ADR-0008)");
    // Layer 1 is proven at compile time (cap-core `tests/ui`), not at runtime; record it.
    kprintln!(
        "[praesidium] isolation: Layer 1 — Cap<T> unforgeable outside cap-core (AC5.1, host-proven)"
    );

    verify_guard_page();
    verify_wx_and_zero();
    verify_domain_entry_gating();

    kprintln!("[praesidium] PRAESIDIUM-P5A-OK");
}

/// Layer 3 (AC5.3): exercise the guard-page primitive end to end. Allocate a 2-frame block,
/// unmap the lower frame's HHDM alias, and confirm it is gone while the upper frame stays mapped
/// RW — i.e. the huge-page split rendered exactly one page non-present without disturbing its
/// neighbour. This is the same primitive `scheduler::spawn` now installs below every task stack.
/// The block is intentionally never freed: a guard's unmapped alias must not return to the buddy.
fn verify_guard_page() {
    let block = memory::alloc_frames(1).unwrap_or_else(|| fatal("no frames for guard-page check"));
    let guard = memory::phys_to_virt(block);
    let neighbour = memory::phys_to_virt(block + PAGE);

    // SAFETY: `block` is the lower frame of our own freshly-allocated block; nothing else aliases
    // it, so unmapping its sole (HHDM) mapping is sound. See `arch::install_guard_page`.
    unsafe { arch::install_guard_page(guard) };

    if arch::translate(guard).is_some() {
        fatal("guard page is still mapped after install_guard_page");
    }
    if arch::translate(neighbour) != Some(block + PAGE) {
        fatal("the page above the guard lost its mapping (split corrupted a neighbour)");
    }
    if arch::page_prot(neighbour) != Some((true, false)) {
        fatal("the page above the guard is not RW-NX after the split");
    }
    kprintln!(
        "[praesidium] isolation: Layer 3 — guard page unmapped, neighbour intact RW-NX (AC5.3)"
    );
}

/// Layer 3 (AC5.3), continued: re-verify W^X is still in force on the live tables under P5 (a
/// `.text` address is R-X, the boot stack in `.bss` is RW-NX) and that a freshly (re)allocated
/// frame reads back zero (CAP-MEM-2).
fn verify_wx_and_zero() {
    // A live `.text` address: bind a kernel fn to a fn *pointer* first (a direct fn-item→integer
    // cast is lint-forbidden), then take its numeric value.
    let text_fn: fn() = run;
    let text = text_fn as usize as u64;
    match arch::page_prot(text) {
        Some((false, true)) => {}
        other => fatal_prot("kernel .text is not R-X", other),
    }
    let data = core::ptr::addr_of!(crate::BOOT_STACK) as u64;
    match arch::page_prot(data) {
        Some((true, false)) => {}
        other => fatal_prot("kernel .bss (boot stack) is not RW-NX", other),
    }

    let z = memory::alloc_zeroed_frame().unwrap_or_else(|| fatal("no frame for zero check"));
    let p = memory::phys_to_virt(z) as *const u64;
    // SAFETY: `p` is a freshly-allocated, HHDM-mapped, readable frame we own; reading two in-bounds
    // 64-bit words (first and last of the 4 KiB page) is sound.
    let (first, last) = unsafe { (p.read_volatile(), p.add(511).read_volatile()) };
    if first != 0 || last != 0 {
        fatal("a freshly zeroed frame was not zero (CAP-MEM-2 violated)");
    }
    memory::free_frames(z);
    kprintln!(
        "[praesidium] isolation: W^X re-verified (.text R-X / .bss RW-NX) + zero-on-alloc (AC5.3)"
    );
}

/// DEC-0008-5: domain entry is capability-gated, never ambient. Holding the `Domain` cap (ENTER)
/// authorizes entering exactly its domain; broad-but-wrong authority (an `Untyped`) and an empty
/// slot are both refused. Proves there is no ambient "switch to domain X".
fn verify_domain_entry_gating() {
    let mut cs = CSpace::<SLOTS>::new(no_op_zero);
    // Broad authority that is nonetheless the *wrong* authority for entry (installs at slot 0).
    cs.set_root_untyped(0, 1);
    // The one cap that authorizes entry — carries ENTER only, never VSpace MAP_TABLE.
    cs.set_root_domain(DOMAIN_SLOT, DEMO_DOMAIN_ID);

    match cs.enter_domain(DOMAIN_SLOT) {
        Ok(id) if id == DEMO_DOMAIN_ID => {}
        other => fatal_enter("holding a Domain cap did not authorize entry", other),
    }
    match cs.enter_domain(NON_DOMAIN_SLOT) {
        Err(CapError::WrongType) => {}
        other => fatal_enter(
            "a non-Domain (Untyped) cap was allowed to enter a domain",
            other,
        ),
    }
    match cs.enter_domain(EMPTY_SLOT) {
        Err(CapError::EmptySlot) => {}
        other => fatal_enter("an empty slot was allowed to enter a domain", other),
    }
    kprintln!(
        "[praesidium] isolation: domain entry is capability-gated, never ambient (DEC-0008-5)"
    );
}
