# ADR-0013: Make asynchronous boundaries describe real future events

**Status:** Accepted principle; current signatures belong to RHI.

## Decision

Use an asynchronous API only when the operation needs a future external event,
such as device acquisition, backend compilation, surface acquisition or
presentation, completion observation, mapping, or a wait. Validation, resource
description, and command recording remain ordinary operations unless the
selected RHI contract says otherwise.

Thread safety alone does not justify a future. The independent
[RHI Core design](https://github.com/fluxel-project/fluxel-rhi/blob/main/crates/core/DESIGN.md)
is authoritative for current method signatures and execution models.

## Consequences

Renderer does not wrap synchronous RHI work in a speculative async layer, and
does not infer completion from submission or task completion.
