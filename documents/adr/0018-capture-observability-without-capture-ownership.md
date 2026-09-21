# ADR-0018: Preserve RHI observability without owning capture runtime

**Status:** Accepted

## Context

Tools need reconstructable command and object semantics, but artifact storage,
snapshot policy, dependency closure, and replay runtime are product concerns.

## Decision

RHI exposes stable identities, canonical recorded work, semantic events, and
tooling descriptions. It does not own capture files, snapshot policy, artifact
dependency closure, or ReplayRuntime.

## Consequences

Capture systems can consume a stable semantic seam without making every RHI
backend implement a persistence format or a replay engine.
