# ADR-0017: Keep acquired presentation images separate from textures

**Status:** Accepted.

## Decision

An acquired presentation image is an exclusive, configuration-bound RHI surface
lease. In the current RHI it is represented by `SurfaceImage` and used through
`AttachmentView`; it is not a general `Texture`. It can be presented or
abandoned according to the surface contract exactly once.

Submission acceptance, GPU completion, and presentation outcome are separate
facts. A resize, browser callback, or successful present call does not prove
completion.

## Consequences

Graph models a presentation import without a second texture or presentation
resource system. Host/platform adapters own lifecycle events while RHI owns
surface configuration, acquisition, and native presentation behavior.
