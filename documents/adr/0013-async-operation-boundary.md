# ADR-0013: Make only future-event operations asynchronous

**Status:** Accepted

## Context

Thread safety and a future event are different properties. Turning ordinary
logical creation and recording into futures obscures ownership without adding
an asynchronous boundary.

## Decision

Adapter/device request, shader/pipeline compilation, submission, completion,
readback, presentation lifecycle, and idle waiting are async. Capability
queries, logical buffer/texture/view/sampler/binding creation, validation,
recording, statistics, and diagnostics remain synchronous.

## Consequences

Backends may use immediate-ready implementations for an async operation, while
concurrent synchronous calls remain ordinary calls. A new async API requires a
real externally completed event.
