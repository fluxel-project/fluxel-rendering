# ADR-0009: Historical resource-floor closure

**Status:** Historical. It does not define current RHI semantics.

The former fixed resource floor and its `DeviceIdentity`, generation, frame
attachment, and accepted-unknown vocabulary were evidence for the retained
renderer baseline. They are migration inputs, not public target types.

The durable lessons are that a logical Graph resource is not a physical asset,
that reuse requires compatible declared state and completion-safe retention, and
that unsupported capability routes fail structurally. Current resource
provenance, submission outcomes, surfaces, and allocation behavior belong to
the independent [RHI Core design](https://github.com/fluxel-project/fluxel-rhi/blob/main/crates/core/DESIGN.md).
