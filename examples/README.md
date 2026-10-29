# Workspace integration examples

This directory is reserved for examples that exercise the assembled renderer
workspace rather than one package in isolation.

Package-specific examples remain beside their crates:

- `crates/rendergraph/examples/` covers graph declaration, compilation, and
  the CPU-only `TestRhi` execution protocol;
- the sibling `fluxel-rhi/crates/rhi/examples/` directory covers the stable
  RHI workloads; its own examples guide defines supported invocation and evidence;
- `crates/renderer/examples/01_headless_frame.rs` exercises the public
  upload → poll → snapshot → draw → poll lifecycle without presentation or
  test-only readback. Run it with `--features gpu-upload` and either `dx12` or
  `vulkan` as its backend argument.

These are API examples, not hardware conformance claims. The only workspace
conformance entry point is `scripts/conformance.ps1` on a clean configured
Windows checkout; it runs ignored fixtures and preserves their evidence log.

The [implementation plan](../documents/version-plan.md) requires future examples
for renderer-internal Graph/stable RHI execution, 2–3 overlapping frames, and a
Material/Shader-driven MVP with built-in Mesh plus an independent Rust Feature
crate. Existing fixed-recipe examples do not satisfy those target gates.
