# ADR-0010: Keep renderer residency private

**Status:** Accepted boundary; historical fixed implementation superseded.

## Context

Logical assets, persistent GPU realizations, and frame work have different
lifetimes. Making residency a Graph resource system or an RHI asset cache would
mix loading policy with execution.

## Decision

`fluxel-assets` retains logical identity and immutable content generations.
Renderer preparation owns per-device persistent realization, upload, selection,
and completion-safe retirement. Its private cache key includes asset identity,
content generation, and the selected RHI device provenance. A renderer-private
generation may be used only where the selected RHI exposes the necessary fact.

Persistent residency uploads cover immutable or streaming assets such as meshes,
textures, and tiles. Per-frame transforms, bone palettes, and comparable inputs
use a separate frame-data lifetime, although both may share transfer machinery.
A pending producer may be consumed only through an enforceable RHI dependency
with matching state/visibility requirements. GPU skinning, simulation, and
culling remain Graph work.

## Consequences

Passes receive resolved imports and never query an asset store or start uploads.
Replacement, loss, partial acceptance, and indeterminate outcomes retain all
potentially affected resources until current RHI evidence permits retirement or
the recovery path discards their device domain. No browser session/token or
general public residency API is introduced.

See [Renderer design](../design-renderer.md) for the current boundary.
