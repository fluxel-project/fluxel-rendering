# Portable workload catalogue

This directory is the portable half of Fluxel's hardware conformance suite.
One workload records only public RHI objects and commands, submits one
`SubmissionPlan`, observes its terminal completion, and makes a CPU-visible
assertion.  Backend fixtures may provide a private provider, code-form-specific
shader artifacts, a compatible submission lane, or a host presentation target.
They must not duplicate portable recording or replace a missing route with a
native API call.

`Unsupported` is a valid outcome only when the *exact* capability, format, or
route needed by the workload is absent from fresh `CapabilityFacts`.  Once that
fact is published, creation/recording/submission failure or an incorrect CPU
result is a conformance failure.

## Baseline matrix

| Workload | Current portable owner | Fixture shape / CPU assertion | Status |
| --- | --- | --- | --- |
| 01 triangle | `raster.rs` | Raster pipeline, optional vertex buffer; render `Rgba8Unorm`, then assert a selected texel. | Available. |
| 02 indexed triangle | — | Raster pipeline plus `INDEX` buffer and index binding; render the same deterministic triangle and assert centre texel. | Add as `indexed_raster.rs`; it must exercise `set_index_buffer` and `draw_indexed`, not merely reuse a vertex-index shader. |
| 03 uniform + texture + sampler | — | Raster pipeline and one bind group containing a uniform buffer, sampled `Rgba8Unorm` texture, and filtering sampler; upload distinct texels and assert the uniform-selected sampled colour. | Add as `sampled_raster.rs`.  The fixture owns only code form and pipeline/interface construction. |
| 04 storage-buffer compute | `compute.rs` | Compute pipeline and storage bind group; dispatch and compare every readback `u32`. | Available. |
| 05 texture upload / copy / readback | `transfer.rs` | Tightly packed upload, texture copy and readback; compare valid texels using returned row pitch.  The extended case also crosses texture-to-buffer and buffer-to-texture. | Available. |
| 06 depth + stencil | `depth_stencil.rs` | Fixture raster pipeline; clear depth/stencil and distinguish strict depth rejection from admission through colour readback. | Available; a clear/store form permits a backend that has no strict-depth shader fixture. |
| 07 MSAA + resolve | — | Gate `TextureSupportQuery` for a 2D 4x colour attachment and resolve route before allocating.  Render to 4x target, resolve to single-sample `Rgba8Unorm`, and assert an edge pixel that distinguishes resolve from a copy. | Deferred: DX12 and Vulkan deliberately publish only 1x while their resolve lowering is absent.  Add `msaa.rs` only together with the first backend that publishes this exact route. |
| 08 indirect | `../mod.rs` (`record_single_indirect_compute`) | Upload the 12-byte `[1, 1, 1]` argument buffer, dispatch indirectly, and compare storage output. | Available but should move into `indirect.rs` without semantic change when the common-module migration is made.  Raster indirect is a later independent workload. |
| 09 occlusion query | `../mod.rs` (`record_empty_occlusion_query`) | Empty raster query interval, resolve one slot, read `u64 == 0`. | Available but should move into `query.rs` with the same migration.  Empty geometry intentionally makes this portable across shader fixtures. |
| 10 multiview | — | Array colour attachment, contiguous mask such as `0b11`; shader writes a different layer-identifying colour, read both layers.  Sparse mask is a separate `SelectiveMultiview` case. | Deferred until a fixture can prove `Multiview`, array attachment creation, and per-layer readback.  Do not infer it from a pipeline descriptor alone. |
| 11 mapping | — | Create a buffer with the exact supported `MAP_WRITE|COPY_SRC` or `MAP_READ|COPY_DST` usage; await `map_buffer`, mutate/read bytes, explicitly `flush`/`invalidate`, drop the RAII lease, then verify through GPU copy/readback. | Add as `mapping.rs`.  It must cover the mapping lease, not confuse `ReadbackTicket::read()` with primary-buffer mapping. |
| 12 compressed texture | — | Choose one *published* family member (BC1, ETC2, or ASTC 4x4), upload one complete encoded 2D mip with exact block extent/byte count, then validate a supported copy or sampling route. | Deferred pending a backend-independent decoded-pixel oracle.  Raw compressed readback only proves byte transport; a sampling fixture with a known encoded block is preferred.  Capability selection must be per format, never a family-wide Boolean. |
| 13 presentation / resize / recreate | `presentation.rs` | Host target and freshly selected configuration; configure, acquire, clear/present, wait present, re-acquire. | Available as a host smoke case.  Resize/reconfigure belongs in a second host invocation because target ownership is platform-specific. |

## Rules for adding a missing case

1. Add one independent `*.rs` module here first, then register it from
   `common/mod.rs` in the same change.  A dormant source file is not a test.
2. The module must provide a capability-gated construction path, one-use
   `RecordedWork` ownership, and an async readback/mapping assertion.  It may
   name public shader, pipeline, resource, command, submission, and
   presentation types only.
3. A backend fixture supplies the smallest code-form-specific payload possible:
   DXIL/SPIR-V/MSL/WGSL/GLSL artifact construction, private provider creation,
   lane selection, and host-surface glue.  It never exposes a native handle to
   the portable case.
4. Add positive, unsupported, and boundary coverage.  Examples: exact block
   row size for compressed upload; `sample_count == 4` rather than a guessed
   MSAA level; mapping range alignment; contiguous versus sparse multiview
   masks; and a `Drop` check for a mapping lease.
5. Record result evidence as `Pass`, `Unsupported`, `Skipped`, or `Failure` as
   defined by `../../README.md` and ADR-0022.  A driver that does not publish a
   route is not a passing implementation of that route.

## Why triangle variants are separate

The existing offscreen raster workload intentionally permits a vertex-index
shader, because it is the smallest cross-code-form triangle proof.  That does
not test vertex/index input lowering.  `indexed_raster.rs` and
`sampled_raster.rs` therefore remain separate workloads: each adds one
observable contract (index-buffer ABI, then descriptor/sampler/texture ABI)
without weakening the baseline triangle's diagnostic value.
