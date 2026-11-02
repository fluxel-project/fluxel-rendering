# ADR-0026: Use static dispatch for Rendering extensibility

**Status:** Accepted target constraint; public APIs remain consumer-driven.

## Decision

Fluxel Rendering's extensibility architecture uses Rust generics, traits and
monomorphization exclusively. RenderFeature, FramePipeline and all geometry,
visibility, sorting, batching and deformation providers are concrete types
assembled by the application at compile time. A trait defines a contract; it
does not require a trait object.

Do not use `dyn Trait`, `Box<dyn Trait>`, `Vec<Box<dyn Trait>>`, trait-object
registries or runtime type erasure for extension composition, including setup.
There is no per-stage dynamic-dispatch exception. External crates remain free
to implement the contracts with their own concrete types.

Typed tuples or ordinary aggregates can implement a Feature-set traversal
contract. Runtime enable/disable uses state. Required runtime selection uses
explicit enum branches among compiled concrete implementations. This knowingly
excludes loading previously unknown Feature types without rebuilding.

## Boundaries and consequences

- Keep Feature/Pipeline processing at batch granularity. Static dispatch alone
  does not make an object-by-object design data-oriented.
- Algorithms and their traits normally belong to Features or external crates.
  Core fixes only required cross-Feature exchange and lifetime protocols.
- CPU batches and Graph-declared GPU products are both valid inputs to supported
  phases. Static composition must not force a CPU DrawList or GPU readback.
- Runtime scene data, geometry descriptors, material parameters and Graph
  topology remain dynamic. They are not type parameters for each object or asset.
- Start with ordinary traits, structs, tuples and necessary associated types.
  Do not add a generic framework, type-level scheduler or macro-generated public
  API to satisfy this rule.
- No trait-object dispatch is an architectural property, not a zero-cost or
  automatic-inlining performance claim. Measure code size, compilation time and
  workload behavior; reduce duplication without type-erasing extensions.

Example method names or `Renderer<P, F>` syntax do not freeze API signatures,
error results, backend parameters or borrowing. S4-S6 consumers establish those
details. This refines [ADR-0025](0025-composable-rendering-policies.md) without
changing S0/S1 scope, crate ownership, RHI contracts or the stage order.
