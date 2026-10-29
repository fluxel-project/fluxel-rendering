# ADR-0003: Add execution parallelism only when evidence justifies it

**Status:** Accepted.

## Decision

Graph correctness is independent of multi-queue scheduling, parallel recording,
command caching, or physical aliasing. The initial renderer integration uses the
selected RHI execution model directly and adds those optimizations only after a
representative workload identifies a need.

Any optimization preserves declared dependencies, transitions, visibility,
completion retention, and failure recovery. The current RHI defines the actual
lanes and recording protocol; this ADR introduces no separate scheduler.
