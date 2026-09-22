# ADR-0012: Keep RHI limited to portable execution

**Status:** Accepted (clarified)

> Clarification (post-foundation target): this ADR keeps RHI free of
> RenderGraph-owned declarations and renderer policy. It does not require
> RenderGraph to be a pure IR with no RHI dependency. RenderGraph directly
> reuses the portable RHI contract; only backend-private APIs remain outside
> RenderGraph.

## Context

RenderGraph scheduling, declared coverage contracts, and renderer policy have
different consumers and lifetimes from portable GPU execution. Putting them in
RHI made backend implementations learn upper-layer concepts they cannot lower.

## Decision

RHI records only actual command-derived `command::ResourceUse`, `RecordedWork`,
`SubmissionPlan`, plan/completion points, and transient lifetimes. It has no
graph bridge, declared-work contract, or external declared-use validation.

`ResourceUse` includes synchronization-only objects when a command has a real
producer/consumer relation but no portable byte-addressable backing. Query-set
slot writes and query resolves are the current case: they retain set identity,
slot range, stage scope, and query access in the work record. They are neither
invented buffers nor native barrier commands.

## Consequences

RenderGraph may keep its own declarations and scheduling checks. Backends see a
small execution vocabulary and derive native hazards from actual uses. New
upper-layer policy must not expand the RHI object model.

A plan can therefore derive an ordering edge between a query producer and a
resolve on another lane before any backend chooses a query-pool, fence, or
memory-dependency mechanism. Query-only uses do not imply a buffer/image state
transition by themselves.
