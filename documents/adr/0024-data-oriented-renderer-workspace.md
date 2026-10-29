# ADR-0024: Integrate Graph into a data-oriented extensible Renderer

**Status:** Accepted target; implementation migration is pending.

## Context

The existing workspace has a standalone graph crate, a fixed renderer, and a
WASM capsule using an older RHI integration. The independent RHI is now stable.
The next ecosystem outcome is a complete renderer with reusable shader/material
semantics, batch scene input, and real external Feature integration.

## Decision

- The target has three principal crates: `fluxel-shader`, `fluxel-material`,
  and `fluxel-renderer`. Material consumes Shader; neither depends on Renderer.
  Shader remains usable by compute and other non-material consumers.
- RenderGraph and the replaceable `FramePipeline` SPI belong inside Renderer.
  Graph keeps its scene/material-independent semantics and directly consumes
  the portable RHI contract. Preserve useful implementation and tests during
  migration; do not maintain a second graph engine.
- `fluxel-assets` stays in Bases with its current logical identity, immutable
  generations, producer coordination, and CPU cache policy. Callers own loading,
  decoding, hot reload, and streaming policy; adapters supply platform I/O.
  Renderer owns per-device GPU realization, uploads, and retirement decisions
  based on RHI completion evidence.
- RenderScene uses typed dense storage, generational renderer identity, batched
  incremental updates, and a stable frame snapshot boundary. Adapters own ECS
  or scene-node mappings. No ECS or universal object layout is introduced.
- Features operate in batches. FramePipeline owns phases and rendering strategy;
  Features contribute render data, shader/material requirements, and declared
  graph work. Per-object virtual drawing and out-of-graph submissions are not
  extension contracts. Rust crate plugins precede any dynamic loading design.
- MVP acceptance includes built-in Mesh and a minimal Feature in a separate
  Rust crate. It proves batch updates, phase participation, and Graph resource
  declarations without a core object-type switch before APIs stabilize.

## Consequences

Consumers of the standalone Graph crate and older RHI interfaces must migrate
with exact dependency revisions, lock inputs, and real integration artifacts.
Changing documentation does not prove those migrations complete. Cached plans,
2–3 overlapping frames, safe retirement, and device/generation rejection are
early gates; optional optimization follows representative measurements.

Reuse Bases and establish Graph/RHI integration before the Shader/Material and
Renderer MVP. Broader material/pipeline/Feature work and measured resource
strategies complete the active scope. Other product directions are undecided,
with no retained design or sequencing commitment. The [implementation plan](../version-plan.md) and
organization [roadmap](https://github.com/fluxel-project/.github/blob/main/ROADMAP.md)
own sequencing; this ADR records boundaries, not a second schedule.

RHI maintenance follows demonstrated consumer requirements. Its open TODOs and
missing target evidence remain explicit, including the Metal runtime boundary.
