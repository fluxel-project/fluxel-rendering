# ADR-0022: Separate portable tests from hardware conformance evidence

**Status:** Accepted

## Context

Portable positive, negative, and boundary tests establish the RHI contract,
but cannot prove that a backend's native lowering, synchronization, format
route, or device-loss handling works on a real adapter. Conversely, a window
that appears correct is weak evidence: it is difficult to automate and often
misses readback, ordering, and lifetime errors.

## Decision

Keep evidence in three deliberately separate tiers:

1. **Portable contract tests** are deterministic CPU-facing unit tests. They
   validate public API rules and must not require a native adapter.
2. **Headless hardware conformance tests** record one focused RHI workload on
   a selected backend, read its observable output back, and assert it on the
   CPU. They are the required native evidence for a capability claim. Cases
   should isolate a route where practical: buffer copy, compute, texture
   upload/readback, raster output, depth/stencil, resolve, indirect, query,
   compressed format, and loss/completion behaviour.
3. **Presentation smoke tests and examples** exercise acquire, present,
   resize, retirement, and shutdown with a real surface. They may be visually
   inspected, but do not replace tier 2. Larger examples adapt representative
   workloads rather than importing another framework's object model.

Shared tier-2 cases run against every backend that publishes the relevant
fact. Backend-private fixtures may create native windows, contexts, or
adapters, but those details never become public RHI types.

Every run records the crate revision, target, backend, adapter/driver and
enabled feature set. Results are classified precisely:

- **Hardware pass:** the asserted workload ran on the named real adapter.
- **Unsupported:** the capability fact correctly rejected the case before
  native work; this is not a pass for that capability.
- **Skipped:** required platform, adapter, browser, surface, or permission was
  unavailable; this supplies no conformance evidence.
- **Failure:** setup, lowering, validation, completion, readback, or assertion
  failed. A failure is never reclassified as unsupported after work begins.

Mock, static, compile-only, and pure encoder tests remain valuable regression
tests, but must be labelled as such and never reported as hardware evidence.

## Consequences

Capability closure requires at least one focused headless hardware assertion
on each platform/backend whose fact is published. Surface support additionally
requires an appropriate presentation smoke result. Windows, Android, browser,
and macOS evidence are separate: success on one cannot stand in for another.

The suite grows from small mechanical cases first; examples are a second-layer
combination workload, not a replacement for conformance cases or a promise of
wgpu API compatibility.
