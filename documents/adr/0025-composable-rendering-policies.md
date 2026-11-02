# ADR-0025: Compose rendering policies over batch data contracts

**Status:** Accepted target direction; public APIs require consumer evidence.

Extension dispatch is further constrained by
[ADR-0026](0026-static-extension-composition.md): compile-time generic composition,
with no trait-object dispatch or runtime type erasure.

## Context

ADR-0024 established three crates, an internal Graph and batch-oriented Features.
The Renderer design still illustrated a global CPU culling/sorting service and
a visibility list as the common preparation path. Those examples cannot define
the extension boundary for heterogeneous geometry, independent spatial indexes,
GPU-produced work or external animation/deformation providers.

## Decision

Renderer fixes typed exchange, phase composition, resource authority and lifetime
rules. Applications select algorithms through Rust crate implementations.
`RenderFeature` and `FramePipeline` are the principal renderer extension seams;
Feature/Pipeline-local providers make visibility, geometry access, ordering,
batching and deformation independently replaceable. Shared policy traits grow
only where cross-Feature cooperation requires a common protocol. Algorithm
traits otherwise remain with Features or external crates, even with multiple
consumers; they do not become one global service or a trait for every object.

Features own typed data and process batches. Scene adapters push incremental
changes and preserve external identity mappings. SoA, AoS and AoSoA are storage
choices based on access patterns, not a universal public layout. This follows
ECS-style identity/data/batch separation without requiring an ECS or its scheduler.

Geometry is a stream/interface description, not a fixed vertex struct. Shader
requirements and geometry layouts are matched in preparation. Animation
simulation and pose evaluation stay external; deformation Features consume
evaluated inputs and contribute CPU preparation, vertex-stage inputs or GPU
Graph work as supported by the selected consumer.

Multiple local algorithms can coexist. CPU candidates and declared GPU products
need different consumption protocols; neither a final CPU DrawList nor GPU
readback is mandatory. A phase owns ordering across contributions, including
transparent composition. Independently sorted Feature lists do not establish a
correct global order.

All frame GPU accesses enter the internal Graph. Plugins do not submit, invent a
second scheduler or compile a new portable command language. Rust plugins are
statically assembled crate implementations, not a stable dynamic-library ABI.

## Consequences

- The reference mesh and Forward path must exercise the same public facade as
  external consumers. Replacing a local provider must not require core edits.
- Exact trait signatures, result protocols and compatibility rules stabilize
  only with executable consumers. Unproved GPU and cross-pass combinations stay
  provisional or explicitly unsupported.
- Keep S0/S1 foundation work and the organization stage order. S4-S6 establish
  batch/extension consumers; S7 proves materially different extensions; S8
  measures optimizations. No public trait scaffolds or RHI changes are authorized
  by this design revision.
- Feature removal or provider replacement cannot release in-flight owners;
  Graph/RHI completion remains the authority for reuse and retirement.

The detailed contract is in [Renderer design](../design-renderer.md); acceptance
is in [the implementation plan](../version-plan.md). This ADR clarifies ADR-0024
without changing repository ownership or the delivery sequence.
