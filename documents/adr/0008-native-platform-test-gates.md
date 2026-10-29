# ADR-0008: Execute platform-specific paths on their native target

**Status:** Accepted.

## Decision

Platform-conditioned code is checked on the platform that executes it. A cross
compile, another operating system, or another browser does not certify a native
fallback, host lifecycle path, or backend route. The active evidence matrix
names the target triple or host, backend, and required real-target oracle.

## Consequences

CI may detect compile and contract drift, while support claims retain native
target evidence. An unavailable platform is recorded as untested rather than
silently treated as success.
