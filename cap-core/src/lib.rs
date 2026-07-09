//! # cap-core — the capability trust root (SPEC-CAP CAP-RUST-1)
//!
//! This is the *only* crate permitted to fabricate capability values via `unsafe`.
//! It is intentionally tiny so its entire unsafe surface can be audited
//! exhaustively; if it ever grows a large dependency tree or public API, that is a
//! red flag — the whole security model rests on this staying small and reviewable.
//!
//! **In P0 it is deliberately empty.** The kernel introduces no authority yet (it
//! validates the Warden handoff and halts), so no capability type or `unsafe`
//! fabrication exists. The capability primitives (`Cap<T, Rights>`, RETYPE/MINT/
//! COPY/MOVE, the MDB, REVOKE) land in **P2**.
#![no_std]
