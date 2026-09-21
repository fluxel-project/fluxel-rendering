# ADR-0006: Superseded — do not use fixed recipes as the RHI pipeline API

**Status:** Superseded by the normative RHI API v1 specification for 0.16.

**Supersession note.** The frozen RHI API now defines portable
`ShaderArtifact`, `BindGroupLayout`, `BindGroup`, `PipelineInterface`,
`RasterPipeline`, and capability-gated `ComputePipeline` objects with validated
descriptors and backend-private lowering. These are execution objects, not a
public material system, render-packet framework, or custom renderer policy.
This ADR remains historical evidence for the earlier fixed vertical slice and
must not constrain the 0.16+ public API.

## Context

The renderer has only evidence-backed fixed shader, layout, texture, sampler,
and binding combinations. A generic pipeline API would guess at unvalidated
ownership, reflection, layout, and portability requirements.

## Decision

This decision is superseded. New code follows the normative RHI API v1
specification; it must not add closed adapter recipes as a substitute for its
public shader, binding, pipeline, recorder, or submission vocabulary.

Represent each proven native combination as a closed RHI artifact and binding
recipe. Renderer-shaped Raster artifacts are exposed only under
`fluxel_rhi::adapter::fixed_artifacts`; keep shaders, descriptors,
samplers, and native objects opaque. Add a new closed recipe only for a
separately planned vertical slice.

## Alternatives

- Expose arbitrary WGSL, descriptors, layouts, and pipeline builders now.
- Encode every new combination as unchecked optional fields in one recipe.

## Consequences

The stable RHI surface does not imply that fixed renderer recipes are a lasting
general contract. General shader/material/pipeline policy waits for real
renderer requirements; closed adapter recipe changes may evolve with the current
vertical slice.

## Evidence

0.1.3 fixed compute and 0.1.4 fixed raster established closed artifacts.
The v0.7.0 workspace release contains discrete indexed, uniform, texture-load,
UV, sampler, sRGB, Lambert, and vertex-color recipes without broadening them
into a general API.

See [RHI architecture](../../crates/rhi/documents/design-rhi.md) and
[Renderer design](../design-renderer.md).
