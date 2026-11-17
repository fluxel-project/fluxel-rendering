# Mipmap tunnel Material consumer

This package adapts RHI example 09's actual tunnel geometry, base-only metalplate
image, perspective camera, repeated UVs and derivative-driven sampling through
Material, Shader, Renderer and RenderGraph. It retains the RHI example and does
not copy its direct encoder. It does not reproduce its Phong lighting or animation.

Supply the original `tunnel_cylinder.gltf` and `metalplate_nomips_rgba.ktx` through
`--model` and `--asset`. The strict shared glTF decoder expands the indexed
geometry and applies the actual nonuniform node TRS; the example then matches
RHI's position/normal Y flip. Only base pixels become `ColorTextureData`; Renderer
Graph blits generate the complete linear RGBA8 chain.

The authored Code Node calls `textureSampleBias` with UV multiplied by `[2, 1]`
and baseline bias zero. Its named sampler requests MirrorRepeat and linear
min/mag/mip filtering with anisotropy one. Source alpha contains specular data;
this opaque sampling consumer outputs sampled RGB and alpha one. Forward uses
direct color; the legacy Deferred colored-albedo consumer applies its existing
diagnostic normal/depth lighting. The oracle models these outputs separately.

The independent CPU oracle clips the real triangles, resolves depth, computes
perspective UVs and same-primitive 2x2 derivatives, and rebuilds the box mip chain
from retained base bytes. It screens stable probes using CPU-only coarse/fine
and LOD precision controls before GPU execution. The probe set must reject
LOD0-only, nearest-mip and clamp-address alternatives at the same RGB tolerance
four. This is a sampling consumer, not a general software renderer or a
physical lighting reference. Inputs, transforms, the actual lock and independent
expectations precede rendering; raw readbacks precede comparison.

Visible mode presents both pipelines in a real background Host window and
compares independent WGC captures with separate same-input offscreen companions.
The fixed 512-pixel DWM corner mask and exact remaining RGB comparison are shared
with the other acceptance consumers. Capture excludes the cursor without moving
it, changing focus, raising the window or changing its topmost state.

`--anisotropy16` requests RHI's original value and must be admitted by the actual
device. It never silently falls back. This control remains explicitly unverified
for numerical quality: a scalar trilinear CPU value is not an aniso-16 oracle.
Bias/distance/filter controls, animated reference lighting, and sRGB mip
production remain additional work; this baseline alone does not close all of 09.

```text
cargo run --locked --features dx12 -- --offscreen --model <tunnel_cylinder.gltf> \
  --asset <metalplate_nomips_rgba.ktx> --output <empty-attempt-directory>
cargo run --locked --features native-gl-wgl -- --visible --exit-after-capture \
  --model <tunnel_cylinder.gltf> --asset <metalplate_nomips_rgba.ktx> \
  --output <empty-attempt-directory>
```
