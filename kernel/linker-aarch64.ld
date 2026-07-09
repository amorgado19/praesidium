/* Praesidium higher-half kernel image — aarch64 (P0 / ADR-0007).
 *
 * Warden's warden-rich loader maps the kernel via TTBR1 at the canonical
 * top-of-address-space base (0xffffffff80000000): 2 MiB-aligned, one contiguous
 * image, <= 2 MiB total, in an L0 slot (511) disjoint from the identity and HHDM
 * (256) slots. The layout is kept trivial so the loader stays trivial.
 *
 * aarch64 reaches this base with the default code model (adrp/add), so unlike
 * x86-64 it needs no `code-model` override — only `relocation-model=static`
 * (set in .cargo/config.toml). */
ENTRY(_start)

SECTIONS {
    . = 0xffffffff80000000;
    __kernel_start = .;

    .text   : ALIGN(0x1000) { *(.text .text.*) }
    .rodata : ALIGN(0x1000) { *(.rodata .rodata.*) }
    .data   : ALIGN(0x1000) { *(.data .data.*) }
    .bss    : ALIGN(0x1000) { *(.bss .bss.*) *(COMMON) }

    __kernel_end = .;

    /DISCARD/ : { *(.eh_frame) *(.eh_frame_hdr) *(.note .note.*) *(.comment) }
}
