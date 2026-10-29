# ADR-0005: Require real-target evidence for native correctness claims

**Status:** Accepted.

## Decision

CPU tests, mocks, compilation, and validation establish protocol behavior; they
do not establish that a native backend, driver, shader route, synchronization,
or presentation path works. Every supported native claim requires a retained
real-target run for that named target, with an independent observable oracle.

The evidence records exact participating revisions and locks, command, OS,
backend, adapter/driver or browser/host, inputs, expected and actual output,
diagnostics, and unsupported or untested routes. Visual work also retains an
inspected output artifact. The ecosystem evidence policy defines the current
retention format and stage-close matrix.

## Consequences

One backend, browser, emulator, or platform does not certify another. Hardware
unavailability leaves the relevant claim open rather than converting it into a
portable pass.
