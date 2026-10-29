# ADR-0006: Historical fixed-recipe RHI boundary

**Status:** Superseded. It does not define the independent RHI API.

## Historical context

Early renderer slices used closed shader and raster recipes while the portable
pipeline contract was not yet established. That was evidence for a small
vertical slice, not a permanent public abstraction.

## Durable lesson

Renderer policy, material semantics, and Graph declarations must not be hidden
inside an RHI convenience recipe. Current portable shader, binding, pipeline,
recording, and submission types are defined only by the independent
[RHI Core design](https://github.com/fluxel-project/fluxel-rhi/blob/main/crates/core/DESIGN.md).

Historical recipes may remain private migration inputs until their consumers
migrate. New rendering work uses the selected RHI revision and the Shader,
Material, FramePipeline, and Graph boundaries described in the rendering design.
