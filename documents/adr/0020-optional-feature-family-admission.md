# ADR-0020: Admit optional feature families as complete portable contracts

**Status:** Accepted

## Context

Avoiding a feature solely because one baseline backend lacks it makes the API a
least-common-denominator wrapper. Publishing a feature from an enum alone is
equally unsafe.

## Decision

Each optional family has a portable vocabulary, precise capability facts or
queries, requirements negotiation, descriptor/recording validation, lowering,
retention, loss behaviour, and conformance evidence. Families include queries,
indirect commands, compressed formats, multiview, storage access, and advanced
shader/pipeline work. Advanced but incomplete lowerings retain their vocabulary
while their facts stay disabled.

## Consequences

`Unsupported` is a correct result, not an API omission. Current TODO families
include physical aliasing, mesh/task and ray paths, cooperative matrices,
external interop, multiplanar routes, native debug capture, HDR/timing, and
advanced descriptor indexing. A reachable path must never use a placeholder
success or panic.
