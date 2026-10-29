# Architecture Decision Records

These ADRs preserve durable Fluxel boundaries. The organization roadmap owns
delivery order; repository designs own current API details. The independent
[RHI Core design](https://github.com/fluxel-project/fluxel-rhi/blob/main/crates/core/DESIGN.md)
is the sole authority for portable RHI types, signatures, execution outcomes,
surface leases, and backend behavior.

An ADR records a decision that remains useful after its original release. A
historical or superseded ADR may preserve a safety lesson, but cannot authorize
an old API, implementation path, capability claim, or completion result.

## Current boundary records

- [ADR-0024: Integrate Graph into a data-oriented extensible Renderer](0024-data-oriented-renderer-workspace.md)
- [ADR-0023: Keep host lifecycle separate from RHI surface integration](0023-host-rhi-integration-boundary.md)
- [ADR-0020: Admit optional features only as complete portable contracts](0020-optional-feature-family-admission.md)
- [ADR-0019: Keep RHI statistics logical, not profiling](0019-portable-logical-statistics.md)
- [ADR-0017: Keep acquired presentation images separate from textures](0017-presentation-is-not-a-texture-view.md)
- [ADR-0016: Publish capabilities only with lowering closure](0016-capability-claims-require-lowering-closure.md)
- [ADR-0015: Keep transient lifetime planning separate from physical allocation](0015-plan-scoped-transient-allocation.md)
- [ADR-0014: Treat a lost device domain as terminal](0014-terminal-device-loss.md)
- [ADR-0013: Make asynchronous boundaries describe real future events](0013-async-operation-boundary.md)
- [ADR-0012: Keep RHI limited to portable execution](0012-pure-rhi-execution-boundary.md)
- [ADR-0011: Build the GL family through three private layers](0011-gl-family-three-layer-boundary.md)
- [ADR-0010: Keep renderer residency private](0010-renderer-private-fixed-asset-residency.md)
- [ADR-0008: Execute platform-specific test paths natively](0008-native-platform-test-gates.md)
- [ADR-0005: Require real-target evidence for native correctness claims](0005-gpu-conformance-evidence.md)
- [ADR-0003: Keep execution lowering serial until evidence justifies more](0003-serial-execution-lowering.md)
- [ADR-0002: Contain native handles and unsafe in the RHI implementation](0002-rhi-unsafe-containment.md)
- [ADR-0001: Keep assets and renderer policy outside RenderGraph](0001-assets-outside-rendergraph.md)

## Historical or withdrawn records

- [ADR-0021: Keep executable shader requirements authoritative](0021-shader-owned-immediate-abi.md)
- [ADR-0018: Withdrawn capture/replay proposal](0018-capture-observability-without-capture-ownership.md)
- [ADR-0009: Historical resource-floor closure](0009-resource-floor-and-reuse-safety.md)
- [ADR-0007: Historical fixed renderer recipes](0007-closed-fixed-renderer-recipes.md)
- [ADR-0006: Historical fixed-recipe RHI boundary](0006-no-general-pipeline-yet.md)
- [ADR-0004: Historical submission-uncertainty quarantine](0004-accepted-unknown-quarantine.md)
