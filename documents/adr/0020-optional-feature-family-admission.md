# ADR-0020: Admit optional features only as complete portable contracts

**Status:** Accepted.

## Decision

An optional feature is published only when it has one defined portable meaning,
opened-device capability facts, validation, lowering, lifetime and loss behavior,
structured refusal, and target evidence. A backend implementation gap is not a
capability fact.

## Consequences

Capability support may differ by backend, format, stage, resource shape, or
route. Current and future RHI work is selected by a demonstrated consumer rather
than by a frozen TODO inventory.
