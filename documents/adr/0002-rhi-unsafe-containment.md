# ADR-0002: Contain native handles and unsafe in RHI backends

**Status:** Accepted.

## Decision

Native API objects, unsafe lowering, backend synchronization, and backend
lifetime rules remain inside the independent RHI implementation. Public RHI
resources are typed, device-affine wrappers with structured errors. Renderer and
Graph use portable contracts and do not unwrap native handles or record native
commands.

## Consequences

RHI validates portable provenance and operation shape before native calls.
Backend-native invariants stay with the backend that can actually enforce them.
See the current [RHI Core design](https://github.com/fluxel-project/fluxel-rhi/blob/main/crates/core/DESIGN.md).
