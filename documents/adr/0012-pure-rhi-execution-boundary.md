# ADR-0012: Keep RHI limited to portable execution

**Status:** Accepted

## Context

RenderGraph scheduling, declared coverage contracts, and renderer policy have
different consumers and lifetimes from portable GPU execution. Putting them in
RHI made backend implementations learn upper-layer concepts they cannot lower.

## Decision

RHI records only actual command-derived `command::ResourceUse`, `RecordedWork`,
`SubmissionPlan`, plan/completion points, and transient lifetimes. It has no
graph bridge, declared-work contract, or external declared-use validation.

## Consequences

RenderGraph may keep its own declarations and scheduling checks. Backends see a
small execution vocabulary and derive native hazards from actual uses. New
upper-layer policy must not expand the RHI object model.
