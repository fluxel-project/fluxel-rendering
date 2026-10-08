# Workspace integration examples

This directory is reserved for examples that exercise the assembled renderer
workspace rather than one package in isolation.

Package-specific examples remain beside their crates:

- `crates/rendergraph/examples/` covers graph declaration, compilation, and
  the CPU-only `TestRhi` execution protocol;
- the sibling `fluxel-rhi/examples/` crate covers native device discovery and opening;
- `crates/renderer/examples/01_headless_frame.rs` exercises the public
  upload → poll → snapshot → draw → poll lifecycle without presentation or
  test-only readback. Run it with `--features gpu-upload` and either `dx12` or
  `vulkan` as its backend argument.

These are API examples, not hardware conformance claims. The only workspace
conformance entry point is `scripts/conformance.ps1` on a clean configured
Windows checkout; it runs ignored fixtures and preserves their evidence log.
