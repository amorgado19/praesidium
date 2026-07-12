//! The IPC message: a fixed **register payload** + optional capability-transfer descriptors.
//!
//! Payloads are registers-only (architect ruling): [`MSG_REGS`] inline data words and up to
//! [`MAX_CAPS`] capability-transfer slots — bulk data goes by GRANTing a `Frame`, never through
//! the kernel. [`MessageInfo`] is the one-word descriptor a caller sends; [`MessageInfo::decode`]
//! is the **hostile-input parser** (a `cargo fuzz` target): it bounds-checks the caller-supplied
//! word counts and refuses malformed values rather than trusting them to index anything.

use cap_core::{Cptr, GrantMode, Rights};

/// Inline data words in the fast payload (the ~4-word registers-only message, DEC-0004-4).
pub const MSG_REGS: usize = 4;
/// Maximum capability-transfer slots one message may carry.
pub const MAX_CAPS: usize = 2;

/// Why a [`MessageInfo`] failed to decode — a malformed (or hostile) descriptor word.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MsgError {
    /// The word claims more inline registers than [`MSG_REGS`].
    TooManyRegs,
    /// The word claims more capability transfers than [`MAX_CAPS`].
    TooManyCaps,
}

/// The one-word message descriptor a caller sends (seL4 `MessageInfo` style): a caller-defined
/// `label` (method selector) plus how many inline data words and capability transfers follow.
/// Packed into a single machine word: `n_regs` in bits 0..8, `n_caps` in bits 8..16, `label`
/// in bits 16..48.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MessageInfo {
    /// Caller-defined selector (method id, etc.). 32-bit; the kernel never interprets it.
    pub label: u32,
    /// Number of valid inline data words in the message (`0..=MSG_REGS`).
    pub n_regs: u8,
    /// Number of capability transfers in the message (`0..=MAX_CAPS`).
    pub n_caps: u8,
}

impl MessageInfo {
    /// Build a descriptor, validating the counts against the fixed limits.
    pub fn new(label: u32, n_regs: u8, n_caps: u8) -> Result<Self, MsgError> {
        if usize::from(n_regs) > MSG_REGS {
            return Err(MsgError::TooManyRegs);
        }
        if usize::from(n_caps) > MAX_CAPS {
            return Err(MsgError::TooManyCaps);
        }
        Ok(Self {
            label,
            n_regs,
            n_caps,
        })
    }

    /// Pack into a single machine word (what a caller places in the descriptor register).
    #[must_use]
    pub fn encode(&self) -> u64 {
        (u64::from(self.label) << 16) | (u64::from(self.n_caps) << 8) | u64::from(self.n_regs)
    }

    /// **Hostile-input parser** (the `cargo fuzz` target): decode a descriptor word from an
    /// untrusted caller. The count fields are **bounds-checked** — a word claiming `n_regs = 255`
    /// is rejected with `Err`, never trusted to index the register array. Total, panic-free, and
    /// pure: any `u64` in ⇒ a valid `MessageInfo` (counts within limits) or a clean `MsgError`.
    pub fn decode(word: u64) -> Result<Self, MsgError> {
        let n_regs = (word & 0xFF) as u8;
        let n_caps = ((word >> 8) & 0xFF) as u8;
        let label = ((word >> 16) & 0xFFFF_FFFF) as u32;
        Self::new(label, n_regs, n_caps)
    }
}

/// A capability-transfer descriptor within a message: which source slot to GRANT, narrowed to
/// which rights, moved or minted. The kernel resolves `src` in the **sender's** CSpace and calls
/// `cap_core::grant` into the receiver's CSpace (monotonic — rights only narrow, CAP-DERIVE-1).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CapXfer {
    /// Slot in the sender's CSpace holding the capability to transfer.
    pub src: Cptr,
    /// Rights the transferred capability should carry (must be ⊆ the source's rights).
    pub rights: Rights,
    /// Whether the sender hands the cap off (`Move`) or keeps it and grants a child (`Mint`).
    pub mode: GrantMode,
}

impl CapXfer {
    /// A placeholder slot (unused entries in a message's transfer array).
    pub const NONE: Self = Self {
        src: 0,
        rights: Rights::empty(),
        mode: GrantMode::Move,
    };
}

/// A fully-formed IPC message: the descriptor, the inline register words, the capability
/// transfers, and the badge the kernel stamps on delivery (identifying the caller/endpoint).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Message {
    pub info: MessageInfo,
    /// Inline data words; only `info.n_regs` are meaningful.
    pub regs: [u64; MSG_REGS],
    /// Capability transfers; only `info.n_caps` are meaningful.
    pub caps: [CapXfer; MAX_CAPS],
    /// Badge stamped by the kernel on delivery (0 until delivered).
    pub badge: u64,
}

impl Message {
    /// An empty message carrying `label` and no data/caps.
    #[must_use]
    pub fn empty(label: u32) -> Self {
        Self {
            // `new(_, 0, 0)` can't exceed the limits, so the unwrap is infallible.
            info: MessageInfo::new(label, 0, 0).unwrap_or(MessageInfo {
                label,
                n_regs: 0,
                n_caps: 0,
            }),
            regs: [0; MSG_REGS],
            caps: [CapXfer::NONE; MAX_CAPS],
            badge: 0,
        }
    }

    /// A message carrying `words` inline data words (truncated to [`MSG_REGS`]).
    #[must_use]
    pub fn with_data(label: u32, words: &[u64]) -> Self {
        let n = words.len().min(MSG_REGS);
        let mut regs = [0u64; MSG_REGS];
        regs[..n].copy_from_slice(&words[..n]);
        Self {
            info: MessageInfo {
                label,
                n_regs: n as u8,
                n_caps: 0,
            },
            regs,
            caps: [CapXfer::NONE; MAX_CAPS],
            badge: 0,
        }
    }

    /// The meaningful inline data words. The count is **clamped** to [`MSG_REGS`] so a `Message`
    /// whose `info.n_regs` was set out of range (a hand-built or hostile descriptor that bypassed
    /// [`MessageInfo::new`]/[`decode`](MessageInfo::decode)) can never index past the array — the
    /// count is advisory, the storage is the ground truth (defense in depth for the hot path).
    #[must_use]
    pub fn data(&self) -> &[u64] {
        &self.regs[..usize::from(self.info.n_regs).min(MSG_REGS)]
    }

    /// The meaningful capability transfers, count clamped to [`MAX_CAPS`] (see [`data`](Self::data)).
    #[must_use]
    pub fn transfers(&self) -> &[CapXfer] {
        &self.caps[..usize::from(self.info.n_caps).min(MAX_CAPS)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_roundtrips() {
        let mi = MessageInfo::new(0xABCD, 3, 1).unwrap();
        assert_eq!(MessageInfo::decode(mi.encode()), Ok(mi));
    }

    #[test]
    fn decode_rejects_oversized_counts() {
        // A hostile word claiming 255 regs / 255 caps must be refused, not trusted.
        assert_eq!(MessageInfo::decode(0x00FF), Err(MsgError::TooManyRegs));
        assert_eq!(MessageInfo::decode(0xFF00), Err(MsgError::TooManyCaps));
        // Exactly at the limit is fine; one over is not.
        assert!(MessageInfo::decode(u64::from(MSG_REGS as u8)).is_ok());
        assert_eq!(
            MessageInfo::decode(u64::from(MSG_REGS as u8) + 1),
            Err(MsgError::TooManyRegs)
        );
    }

    #[test]
    fn decode_is_total_and_panic_free() {
        // Sweep a range of hostile words: every one yields Ok(valid) or a clean Err — never panics
        // and never returns counts out of range (the fuzz target asserts this exhaustively).
        for w in 0u64..100_000 {
            match MessageInfo::decode(w.wrapping_mul(0x9E37_79B9_7F4A_7C15)) {
                Ok(mi) => {
                    assert!(usize::from(mi.n_regs) <= MSG_REGS);
                    assert!(usize::from(mi.n_caps) <= MAX_CAPS);
                }
                Err(_) => {}
            }
        }
    }

    #[test]
    fn with_data_truncates_and_reports_length() {
        let m = Message::with_data(1, &[10, 20, 30, 40, 50, 60]); // 6 > MSG_REGS
        assert_eq!(m.info.n_regs as usize, MSG_REGS);
        assert_eq!(m.data(), &[10, 20, 30, 40]);
        assert!(m.transfers().is_empty());
    }

    #[test]
    fn accessors_clamp_a_malformed_count() {
        // A Message whose info counts were set out of range (a hand-built / hostile descriptor
        // bypassing new()/decode) must not index past the fixed arrays — the accessors clamp.
        let mut m = Message::empty(0);
        m.info.n_regs = 250;
        m.info.n_caps = 99;
        assert_eq!(m.data().len(), MSG_REGS, "data clamps to the array size");
        assert_eq!(
            m.transfers().len(),
            MAX_CAPS,
            "transfers clamps to the array size"
        );
    }
}
