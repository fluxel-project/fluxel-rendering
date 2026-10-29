# ADR-0023: Keep host lifecycle separate from RHI surface integration

**Status:** Accepted.

## Decision

Host and browser adapters own the lifetime of windows, canvases, resize,
foreground/background, and destruction events. A runtime composition adapter
may borrow standard platform handles to establish an opaque RHI surface route;
it keeps the host object alive for the route's lifetime. `fluxel-rhi` owns
configuration, acquired surface images, presentation, GPU synchronization, and
backend-native surface state.

Neither RHI nor Renderer imports Host, browser-session, or raw platform object
types into its public resource model. The adapter is lifecycle composition, not
a second GPU API, resource manager, or frame scheduler.

## Consequences

Host destruction or lifecycle changes retire the affected surface route through
the selected RHI surface contract. Platform routes require their own lifecycle
and real-target evidence; paths and test harness arrangements are implementation
details, not this ADR's contract.
