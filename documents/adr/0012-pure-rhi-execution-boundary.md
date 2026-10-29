# ADR-0012: Keep RHI limited to portable execution

**Status:** Accepted.

## Context

Graph declarations and renderer policy have different owners from portable GPU
execution. Backends cannot lower scene policy, inferred hazards, or a second
command model.

## Decision

RenderGraph directly reuses portable RHI descriptors, resources, command scopes,
explicit transitions, submission plans, completion facts, and surface leases.
It owns logical resource versions, declared access coverage, hazards,
dependencies, scheduling, and lifetime analysis. RHI validates portable object
provenance and execution shape; it does not infer Graph dependencies, hazards,
transitions, or retirement policy.

Backends own native command buffers, barriers, synchronization, heaps, and
native handles. Graph never supplies native instructions or accesses native
objects.

## Consequences

There is no Graph-to-RHI bridge vocabulary or duplicate command stream. A
portable wait orders work but does not create missing visibility; Graph's
explicit state and dependency plan remains authoritative.
