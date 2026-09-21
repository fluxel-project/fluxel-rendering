# ADR-0003: Keep execution lowering serial until evidence justifies more

**Status:** Accepted

## Context

A RenderGraph execution plan describes portable dependencies, not a promise of
multiple queues, parallel recording jobs, command caching, or aliasing.

## Decision

The current native lowering uses one logical queue, serial recording, and one
submission per frame. A shared queue-operation lock covers submit and
error-path idle waits. Same-state write dependencies remain semantic facts even
when a backend needs no state transition.

## Alternatives

- Treat graph parallelism as native queue parallelism.
- Add multi-queue, parallel recording, aliasing, or caches before a measured
  profile requires them.

## Consequences

Correctness is specified independently of performance strategy. Later lowering
optimizations must preserve plan ordering, visibility, completion, and lease
semantics and require their own evidence.

## Evidence

C03 in 0.1.2 exercised same-state overlapping writes; 0.1.3 added the shared
queue lock; 0.1.4 proved Raster→Compute→Copy in one submission. No release has
claimed multi-queue or parallel-recording support.

See [RenderGraph architecture](../design-rendergraph.md) and
[RHI design](../../crates/rhi/documents/design-rhi.md) for the current execution contract.
