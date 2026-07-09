//! Serial-first logging (GC-02 / T0.2).
//!
//! A tiny [`core::fmt::Write`] sink over the arch serial-byte seam, plus the
//! `kprint!` / `kprintln!` macros used everywhere in the kernel (including the
//! panic handler). Works headless over QEMU `-serial stdio` on both arches; `\n`
//! is expanded to `\r\n` so raw terminals render newlines correctly.

use core::fmt::{self, Write};

/// Zero-sized writer that emits bytes through the arch serial seam.
pub struct Serial;

impl Write for Serial {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for byte in s.bytes() {
            if byte == b'\n' {
                crate::arch::serial_write_byte(b'\r');
            }
            crate::arch::serial_write_byte(byte);
        }
        Ok(())
    }
}

/// Implementation detail of the `kprint!` macros; not called directly.
#[doc(hidden)]
pub fn _print(args: fmt::Arguments<'_>) {
    // Emitting to the serial port is infallible here; discard the formatter result.
    let _ = Serial.write_fmt(args);
}

/// Print a formatted line to the serial console (with a trailing newline).
/// The kernel's sole logging entry point in P0; `kprint!` (no newline) will be
/// added in the phase that first needs it.
macro_rules! kprintln {
    () => { $crate::serial::_print(format_args!("\n")) };
    ($($arg:tt)*) => { $crate::serial::_print(format_args!("{}\n", format_args!($($arg)*))) };
}
