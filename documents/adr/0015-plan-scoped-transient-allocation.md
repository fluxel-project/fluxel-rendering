# ADR-0015: Keep transient lifetime planning separate from physical allocation

**Status:** Accepted.

## Decision

RenderGraph derives logical transient lifetimes, compatible reuse, and alias
eligibility from declared access and dependencies. It uses the selected portable
RHI allocation and placement contract when a placement is required. Dedicated
allocation remains the correctness fallback.

RHI supplies allocation requirements and realizes placement; each backend owns
its native heap, alias, barrier, and synchronization strategy. Physical aliasing
is published only after measured need and complete lifetime, visibility,
completion, loss, and target evidence.

## Consequences

GL and WebGPU need no fabricated aliasing path. Graph does not expose native
heap offsets or native synchronization instructions.
