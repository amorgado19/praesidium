/* Praesidium higher-half kernel image — x86-64 (P0 / ADR-0007).
 *
 * Warden's warden-rich loader maps the kernel at the canonical top-of-address-
 * space base (-2 GiB, 0xffffffff80000000): 2 MiB-aligned, one contiguous image,
 * <= 2 MiB total, in a PML4 slot (511) disjoint from the identity (0) and HHDM
 * (256) slots. The layout is kept trivial so the loader stays trivial.
 *
 * On x86-64 this base additionally requires `-C code-model=kernel` (set in
 * .cargo/config.toml) so RIP-relative references resolve within the negative
 * 2 GiB half. */
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
