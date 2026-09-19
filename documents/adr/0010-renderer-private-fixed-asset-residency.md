# ADR-0010: Keep fixed-asset residency renderer-private

**Status:** Accepted for renderer-private residency. The `0.14` browser-token
implementation detail is historical. All RHI identity, submission, completion,
presentation, retirement, and loss semantics are defined solely by
[Fluxel RHI API v1](../design-rhi.md).

## Context

The fixed renderer needs to reuse uploaded geometry and RGBA8 images across
frames without making persistent assets part of a RenderGraph declaration or
turning the renderer into a general asset/cache API.  A CPU asset may be
replaced with new contents, a device may be recreated, and an old GPU upload
may still be referenced by accepted work.  Reusing an identifier alone would
therefore permit stale content or resources from another device to be bound.

The v1 RHI `SubmissionReceipt` and its `CompletionPoint`s establish when native
work has terminally completed. Extending RHI, RenderGraph, or the
closed shader/recipe boundary merely to express residency would duplicate that
responsibility and broaden unrelated APIs.

## Decision

The following 0.14 decision remains renderer-private and compatible with v1;
it is not an alternative RHI resource contract. It adds a residency layer for
only these fixed CPU asset domains:

- `MeshAsset` backed by `Geometry`; and
- `ImageAsset` backed by linear `Rgba8Image`.

The renderer's `gpu-residency` feature consumes `fluxel-assets` from the exact
`fluxel-bases` `v0.13.4` revision
`22c4eb0e199575aa71b59f3abc6ec3f72d934b9a`. It resolves a resident GPU
representation by the complete key
`(AssetId, ContentGeneration, DeviceIdentity)`. `AssetId` selects the logical
asset, `ContentGeneration` selects its exact immutable contents, and
`DeviceIdentity` prevents a resource created for one device from being used on
another. This key is renderer bookkeeping, not a public RHI handle or a
RenderGraph resource identity.

Each entry has the deliberately small lifecycle
`PendingUpload -> Committed -> RetireCandidate`. A pending upload cannot be
selected by a draw. A committed entry is the only selectable representation
for its exact key. Superseded, evicted, or old-device entries become retirement
candidates; their native resources remain retained through the relevant
`SubmissionReceipt` and last-referencing `CompletionPoint`, until terminal
completion or terminal device loss. The renderer never infers completion from
a frame boundary or a cache replacement.

Asset resolution occurs during renderer preparation. Preparation obtains an
immutable CPU `AssetSnapshot`, resolves or starts the fixed upload, and passes
only the resulting renderer-owned GPU snapshot/lease into graph declaration. A
failed historical attempt is never silently republished as a committed entry.
For v1 submission, the renderer associates retirement with the returned receipt
and terminal completion rather than maintaining an accepted-unknown state.
A raster pass receives no `AssetStore` and does not discover, load, or choose
assets. RenderGraph continues to see explicit imports and access states only.

On terminal device loss, the residency layer retains CPU `AssetSnapshot`
values, requests a new device, creates entries under the new terminal
`DeviceIdentity` / `DeviceGeneration` domain, and reuploads their fixed
contents. It does not transplant old native objects. Old-domain entries remain
retirement candidates until their receipt points have terminal completion or
the old domain has terminally lost.

This decision does not expand the general/native RHI, RenderGraph, or general
shader contract. It adds no public cache API, generic resource residency
protocol, descriptor model, material system, or configurable shader/pipeline
path. The historical adapter used a closed experimental browser-residency seam.
The v1-compatible replacement uses renderer-private asset/content generations
and the terminal `DeviceIdentity` / `DeviceGeneration` domain. No browser
session/token is a public or backend resource architecture.

## Alternatives

- Key residency by `AssetId` alone, or by asset identity without content or
  device generation.
- Resolve assets while recording a render pass and hand `AssetStore` into that
  pass.
- Destroy superseded or old-device GPU objects immediately on replacement or
  device recreation.
- Put residency entries, loading, or eviction into RenderGraph or RHI.
- Generalize the RHI resource API or shader/material contracts before proving
  this fixed mesh-and-image path.

## Consequences

The renderer gains deterministic reuse for the two fixed asset domains while
preserving immutable per-frame graph inputs. It must retain CPU snapshots and
track a small amount of per-key upload, lease, and retirement state. A changed
asset or recreated device can temporarily require another upload even when a
visually similar older resource exists.

The exact key and receipt-governed retirement prevent stale-content and
cross-device binding. Asset selection remains renderer preparation policy, not
a graph or pass concern.

This decision reinforces [ADR-0001](0001-assets-outside-rendergraph.md),
[ADR-0004](0004-accepted-unknown-quarantine.md), and
[ADR-0009](0009-resource-floor-and-reuse-safety.md).

## Evidence

The historical 0.14 implementation covered key separation by asset, content
generation, and device; pending-not-selectable behavior; committed reuse; and
lease-governed retirement. A v1 implementation must instead verify receipt and
terminal-completion retirement after replacement and terminal device loss, CPU
snapshot reupload, and proof that preparation, rather than a raster pass,
resolves assets. It must also demonstrate that no general/native RHI,
RenderGraph, or general shader API expansion is required.

See [Renderer design](../design-renderer.md) for the architectural flow.
