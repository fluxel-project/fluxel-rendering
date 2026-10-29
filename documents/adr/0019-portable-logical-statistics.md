# ADR-0019: Keep RHI statistics logical, not profiling

**Status:** Accepted.

## Decision

RHI statistics are inexpensive logical request and handoff observations. They
do not add GPU work, imply residency/presentation status, or claim driver timing
or hardware counters. Backend profiling and performance measurement have their
own target-specific evidence.

## Consequences

Renderer diagnostics may use these counters without treating them as a
performance oracle. The exact current fields belong to the independent RHI.
