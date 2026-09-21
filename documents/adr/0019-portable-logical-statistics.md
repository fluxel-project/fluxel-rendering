# ADR-0019: Define statistics as logical observation, not profiling

**Status:** Accepted

## Context

Native counters have different availability, cost, and meaning. Treating them
as portable statistics would silently add work or report incomparable values.

## Decision

RHI statistics report logical object, command, submission, and lifecycle
observations. They never insert GPU commands or claim native profiling facts.
Native profiling remains an optional backend/tooling concern with its own
capability and conversion closure.

## Consequences

Statistics are predictable across backends and safe to query in diagnostics;
their values do not imply timing, pipeline counters, or driver metrics.
