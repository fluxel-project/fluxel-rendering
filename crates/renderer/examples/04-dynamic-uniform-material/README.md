# Renderer 04 dynamic-uniform material

This is the Renderer Material consumer adapted from RHI
`04_dynamic_uniform_buffer`. It retains the source workload's 5 × 5 × 5 cube
lattice, 125 independent draws, deterministic per-cube transforms, right-handed
camera conventions, `Depth32Float` clear-to-one, `LessEqual` depth writes, and
`CullMode::None`. It does not claim hardware instancing: the source uses one
dynamic uniform offset per indexed cube draw.

The Renderer input expands the source's 36 indices to a non-indexed 36-vertex
stream because current `GeometryData` has no index-buffer contract. Position,
zero UV, and source vertex colour remain explicit stream attributes; scene
objects still share one immutable geometry and Material asset.

The consumer requires `ForwardDepthMode::Depth32LessEqualWrite` and must use a
single Renderer Graph submission. It saves the cube stream, model matrices,
camera, record count, depth contract, independent projected/depth oracle, and
raw readback before evaluating results. The initial acceptance set contains the
125-cube grid at two deterministic times plus a two-colour overlap draw-order
reversal. Both orders must retain the nearer result, proving raster depth rather
than CPU sorting.

RHI source remains at `fluxel-rhi` revision
`ebdcb614140296197fcf9abe2924ab0faf5cc490`; this directory contains no direct
RHI command recording. Native offscreen and visible paths use Renderer and the public Host/RHI
interfaces. Offscreen execution retains four grid cases (two fixed times, each
with both draw orders) and two constant-colour overlap controls. Independent
CPU raster inputs, expected images and separated stable probes are written
before any case is submitted. Raw readbacks precede numerical verification;
probe tolerance is two UNORM8 steps, while order-reversal images must match
exactly. The time control must change the output. This checks depth and dynamic
uniform data without claiming PBR, GPU instancing or Deferred multi-object support.

```powershell
cargo +1.87.0 run --locked --manifest-path crates/renderer/examples/04-dynamic-uniform-material/Cargo.toml --features dx12 -- --offscreen --output target/renderer-04-depth
```

Select `native-gl-wgl` for WGL. `--visible --exit-after-capture` presents one
125-cube case into a real acquired Surface lease and captures its window using
Windows Graphics Capture. The same immutable scene also produces an independent
RHI offscreen companion. Its RGB must match the window outside the fixed bottom
16-by-16 DWM corners. The contract and mask are saved before rendering; window
alpha is ignored for the opaque lease. `--time` and `--reverse` select the initial
case. Without `--exit-after-capture`, close the demo window to exit.

Diagnostic activation is checked by the source-bound attempt runner's
`--validate` option. A compile or an unverified diagnostic record does not
constitute target acceptance.
