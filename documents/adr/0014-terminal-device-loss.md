# ADR-0014: Treat device loss as a terminal device identity state

**Status:** Accepted

## Context

Native APIs cannot revive a removed device safely, and browser contexts have
the same ownership problem. A mutable generation counter would permit stale
objects to appear valid after recovery.

## Decision

Loss is terminal for one `DeviceIdentity`. It wakes pending completion,
readback, acquire, present, and idle futures; later operations return structured
`DeviceLost`; `status()` and `loss_info()` remain synchronously queryable. A
new request creates a distinct identity. There is no required `wait_lost()` API.

## Consequences

Callers recover by rebuilding device-scoped objects. Backends must route every
native loss observation through the same terminal authority and never publish
readback success after loss.
