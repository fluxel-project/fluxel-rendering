# ADR-0015: Freeze plan-scoped transient lifetimes with a dedicated fallback

**Status:** Accepted

## Context

Transient allocation affects resource lifetime and synchronization, so waiting
for a later allocator design would leave an API hole. Native heap/aliasing
mechanisms differ substantially.

## Decision

`TransientLifetime` uses an acquire `PlanPoint` and release frontier;
`SubmissionPlanBuilder` reserves batches and exposes a plan-scoped allocator.
Every backend implements `Dedicated`. `Aliasing` is advertised only after
physical placement, non-overlap proof, native barriers/dependencies, and
completion-safe retirement are complete.

## Consequences

WebGPU and GL remain correct without aliasing. DX12, Vulkan, and Metal can add
heap strategies later without changing public lifetime vocabulary.
