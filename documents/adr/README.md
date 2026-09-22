# Architecture Decision Records

This directory preserves the small set of decisions that constrain future
work. Plans and reviews record a release's investigation and evidence; an ADR
records only a decision that remains relevant after that release.

The compact [RHI architecture guide](../../crates/rhi/documents/design-rhi.md)
describes the current model and points to code. Each ADR records one durable
boundary, why it was chosen, and its cost; it does not duplicate descriptor
fields or backend implementation detail. An ADR may later be marked
`Superseded by: ADR-NNNN`.

## Index

- [ADR-0011: Build the GL family through three private layers](0011-gl-family-three-layer-boundary.md)
- [ADR-0020: Admit optional feature families as complete portable contracts](0020-optional-feature-family-admission.md)
- [ADR-0019: Define statistics as logical observation, not profiling](0019-portable-logical-statistics.md)
- [ADR-0018: Preserve RHI observability without owning capture runtime](0018-capture-observability-without-capture-ownership.md)
- [ADR-0017: Keep presentation frames separate from textures and completion](0017-presentation-is-not-a-texture-view.md)
- [ADR-0016: Publish capabilities only with lowering closure](0016-capability-claims-require-lowering-closure.md)
- [ADR-0015: Freeze plan-scoped transient lifetimes with dedicated fallback](0015-plan-scoped-transient-allocation.md)
- [ADR-0014: Treat device loss as a terminal identity state](0014-terminal-device-loss.md)
- [ADR-0013: Make only future-event operations asynchronous](0013-async-operation-boundary.md)
- [ADR-0012: Keep RHI limited to portable execution — clarified: RenderGraph directly uses the portable RHI contract](0012-pure-rhi-execution-boundary.md)
- [ADR-0010: Keep fixed-asset residency renderer-private — RHI semantics superseded by API v1](0010-renderer-private-fixed-asset-residency.md)
- [ADR-0009: Keep the resource floor closed and reuse stateful — historical 0.12–0.15](0009-resource-floor-and-reuse-safety.md)
- [ADR-0008: Execute platform-specific test paths natively](0008-native-platform-test-gates.md)
- [ADR-0007: Keep fixed renderer recipes closed — renderer-private only](0007-closed-fixed-renderer-recipes.md)
- [ADR-0006: Do not expose a general pipeline abstraction yet — superseded by RHI API v1](0006-no-general-pipeline-yet.md)
- [ADR-0005: Require GPU conformance evidence for native correctness claims](0005-gpu-conformance-evidence.md)
- [ADR-0004: Quarantine accepted-unknown GPU work — superseded by RHI API v1 acceptance contract](0004-accepted-unknown-quarantine.md)
- [ADR-0003: Keep execution lowering serial until evidence justifies more](0003-serial-execution-lowering.md)
- [ADR-0002: Contain native handles and unsafe in the RHI implementation](0002-rhi-unsafe-containment.md)
- [ADR-0001: Keep assets and renderer policy outside RenderGraph](0001-assets-outside-rendergraph.md)

## Release mapping

Newest first. “Confirmed” means a release retained or strengthened an existing
constraint; it does not create a new ADR.

ADR-0004 and ADR-0006 are historical and superseded as 0.16 RHI API decisions.
ADR-0007 remains applicable only to renderer-private recipes; none of these
three ADRs define or constrain the normative RHI API v1 surface.

| Release | ADR decision recorded or confirmed |
| --- | --- |
| 0.2.8 | ADR-0007 confirmed; ADR-0008 added. |
| 0.2.7 | ADR-0007 confirmed. |
| 0.2.6 | ADR-0006 confirmed. |
| 0.2.5 | ADR-0003 and ADR-0006 confirmed. |
| 0.2.4 | ADR-0006 confirmed. |
| 0.2.3 | ADR-0001 and ADR-0006 confirmed. |
| 0.2.2 | ADR-0004 and ADR-0006 confirmed. |
| 0.2.1 | ADR-0004 and ADR-0006 confirmed. |
| 0.2.0 | ADR-0001 and ADR-0004 confirmed. |
| 0.1.4 | ADR-0002, ADR-0003, and ADR-0005 confirmed. |
| 0.1.3 | ADR-0003, ADR-0004, and ADR-0005 confirmed. |
| 0.1.2 | ADR-0003, ADR-0004, and ADR-0005 confirmed. |
| 0.1.1 | ADR-0002 confirmed; no submission decision existed yet. |
| 0.1.0 | ADR-0001 and ADR-0002 added. |
