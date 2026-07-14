/* refproc userspace image (P7b). Linked into the loader's reserved process VA window
 * [1 GiB, 2 GiB) (kernel loader::PROC_VA_BASE = 0x4000_0000). Two page-aligned PT_LOAD segments so
 * the packaged `.pex` has a clean R-X code segment + RW-NX data segment: the format forbids W+X and
 * requires 4 KiB-aligned vaddr AND mem_size, with the entry inside an executable segment.
 *
 * Arch-independent (pure section layout). xtask builds refproc for the bare-metal userspace target
 * with this script via RUSTFLAGS (overriding the kernel's high-half script in .cargo/config.toml).
 * The base is 2 MiB-aligned; P7b-i.2 links the second process at a distinct base to coexist in the
 * shared address space. */
ENTRY(_start)

PHDRS {
    text PT_LOAD FLAGS(5); /* PF_R | PF_X */
    data PT_LOAD FLAGS(6); /* PF_R | PF_W */
}

SECTIONS {
    . = 0x40100000;

    .text : ALIGN(0x1000) { *(.text .text.*) } :text
    .rodata : ALIGN(0x1000) { *(.rodata .rodata.*) } :text
    . = ALIGN(0x1000);

    .data : ALIGN(0x1000) { *(.data .data.*) } :data
    .bss : ALIGN(0x1000) { *(.bss .bss.*) *(COMMON) } :data
    . = ALIGN(0x1000);

    /DISCARD/ : {
        *(.eh_frame*) *(.eh_frame_hdr) *(.note*) *(.comment)
        *(.got) *(.got.plt) *(.plt) *(.plt.*)
        *(.dynamic) *(.dynsym) *(.dynstr) *(.hash) *(.gnu.hash) *(.interp)
    }
}
