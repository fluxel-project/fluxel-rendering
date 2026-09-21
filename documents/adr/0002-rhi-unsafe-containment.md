# ADR-0002: Contain native handles and unsafe in the RHI implementation

**Status:** Accepted

## Context

DX12 and Vulkan lifetime, state, alignment, and queue rules cannot safely be
represented by leaked HAL handles in application-facing APIs.

## Decision

Keep HAL/native types and all `unsafe` within the private RHI implementation
boundary (`imp` and its private submodules). Public APIs expose opaque,
device-affine resources, pipelines, bindings, leases, and structured errors.

## Alternatives

- Expose raw handles for renderer or graph code to record commands.
- Spread small `unsafe` wrappers across graph and renderer modules.

## Consequences

The RHI must validate device identity, usage, state, ranges, alignment,
thread serialization, and lifetime before native calls. Renderer and
RenderGraph can state portable contracts without inheriting HAL safety rules.

## Evidence

0.1.0 established the boundary; 0.1.1 resource leasing and 0.1.2–0.1.4 native
execution validated it. 0.2.8 preserves it while splitting the private module.

See [RHI design](../../crates/rhi/documents/design-rhi.md) for the current native boundary.
