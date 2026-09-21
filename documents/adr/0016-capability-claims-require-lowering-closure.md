# ADR-0016: Publish capabilities only with lowering closure

**Status:** Accepted

## Context

Native API names, extension strings, traits, and partial lowering do not prove
that a portable operation is executable. Over-reporting moves predictable
errors from validation into submission or drivers.

## Decision

Capability facts are instance data. A `Supported` answer requires exact
validation, native lowering, resource retention, completion/loss behaviour,
and at least one conformance case. Incomplete paths stay absent and fail
structured before native work.

## Consequences

Facts may be asymmetric across formats, dimensions, stages, routes, and
platforms. Backend-private optimizations do not become public facts merely
because a native mechanism exists.
