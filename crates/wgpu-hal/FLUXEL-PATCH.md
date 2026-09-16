# Fluxel patch for wgpu-hal 30.0.1

`crates/wgpu-hal` is the crates.io `wgpu-hal` 30.0.1 source under its original
MIT/Apache-2.0 licenses. Fluxel carries one DX12 fix from upstream commit
`7d1314d537216a2c84b6a4bbadd4df7b87fc0a6b` / gfx-rs/wgpu#10221:

- lower `SamplerDescriptor::compare == None` to
  `D3D12_COMPARISON_FUNC_NONE`, not the default `ALWAYS` value
  (`src/dx12/device.rs`, sampler conversion).

The crates.io implementation triggers D3D12 validation error #1361 when a
standard filtering sampler is created.

## Why a path dependency and not `[patch.crates-io]`

The workspace previously carried this source in `vendor/wgpu-hal-30.0.1` and
redirected it with a root `[patch.crates-io]` entry. Cargo reads `[patch]` only
from the *root* manifest of the build, so the redirect never reached anyone who
depended on `fluxel-rhi` through the documented Git dependency: an external
consumer resolved the unpatched crates.io 30.0.1 and ran different DX12 sampler
code than the repository's own CI. An independent consumer's `cargo metadata`
confirmed the registry source.

`fluxel-rhi` therefore depends on this directory by path:

```toml
wgpu-hal = { package = "fluxel-wgpu-hal", path = "../wgpu-hal", version = "30.0.1", default-features = false }
```

Path dependencies resolve inside the checked-out Git source, so consumers get
this source with no `[patch]` of their own. The package is renamed to
`fluxel-wgpu-hal` so it cannot collide in the resolver with the registry
package it derives from; `[lib] name = "wgpu_hal"` keeps every `wgpu_hal::…`
path in this crate and in `fluxel-rhi` identical to upstream.

## Workspace status

The root manifest lists `crates/wgpu-hal` in `exclude`. It is a path dependency,
not a workspace member, so:

- `--workspace` commands neither build it with `--all-features` nor scan it;
- its upstream dev surface never resolves. Upstream's `[[example]]` targets,
  the `examples/` directory and the dev-dependencies (`winit`, `glutin`,
  `glutin-winit`, `glam`, `env_logger`) are removed from this fork, because
  Fluxel's ecosystem rules exclude winit/glutin from every repository. They are
  never built for a path dependency, and dropping them keeps that guarantee
  structural rather than incidental.

## Updating or removing

- **Upstream release containing the fix, still Rust-1.87-compatible:** delete
  this directory, restore `wgpu-hal = { version = "=30.0.1", default-features = false }`
  in `crates/rhi/Cargo.toml`, drop `exclude = ["crates/wgpu-hal"]`, and remove
  the paragraph about this patch from `crates/rhi/README.md`. The workspace must
  re-run the independent-consumer resolution check (0.15 release gate), since
  the delivery path changes again.
- **Rebasing onto a newer upstream source:** re-apply the single sampler fix
  above, then re-apply the renames and trims listed here. Both are mechanical,
  and the listed `exclude` entry and manifest keys are the complete set.
