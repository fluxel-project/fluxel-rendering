# Four named texture PBR acceptance

Two real scene records use one authored UV0 metallic/roughness material through
Forward and Deferred. Four named images provide base color, tangent-space
normal, packed ORM, and emissive. Base/emissive are sRGB; normal/ORM are linear.
The shader explicitly unpacks normal RGB. A shared ordinary sampler uses linear
filtering, clamp addressing, and LOD zero.

Bindings are deliberately reordered: normal 0, sampler 1, ORM 2, base 3,
emissive 4, dynamic tint 5. Source references arrive in a different order.
The two records share normal/emissive assets and use distinct base/ORM assets.
An alias control binds one physical image to two logical names in the second
record. Separate source controls change only base, normal, AO, roughness,
metallic, or emissive. All images are nonconstant 2x2 payloads.

```text
cargo run --locked --features dx12 -- --offscreen --output target/four-textures-dx12
cargo run --locked --features native-gl-wgl -- --visible --exit-after-capture --output target/four-textures-wgl
```

Before each case uses GPU resources, the consumer saves geometry, model/tint
data, every image, the role/source mapping, material definition and independent
CPU predictions. The reference projects the actual diagonal plane models,
derives UV at pixel centers, decodes source sRGB texels before bilinear filtering,
and calculates GGX direct light, diffuse irradiance/AO and emissive. Deferred
base/metal/normal/roughness UNORM and binary16 effects are modeled separately.
Twelve interior probes and four background probes use a predeclared RGB tolerance
of three. Raw readbacks
are saved before the oracle runs. This is a numerical source/binding consumer;
it does not claim curved geometry, perspective, mip generation, specular IBL,
transparency, HDR output or full material closure.

Probe selection uses only CPU source cases before GPU execution. Candidate
coordinates must remain within two RGB codes when filter coordinates take either
neighbor at eight fractional bits and normal/ORM samples take neighboring
UNORM8 values, for every control and both pipelines. Probes remain at least
16 pixels apart. The
middle-column probes still interpolate both texture axes. All controls use the
same selected coordinates. This protects the point oracle from amplifying a
legal filter-coordinate rounding into a packed-normal GGX highlight change;
it does not widen the runtime tolerance. See the
[Direct3D sampling precision contract](https://microsoft.github.io/DirectX-Specs/d3d/archive/D3D11_3_FunctionalSpec.htm).

Visible mode presents both pipelines through real acquired Surface leases and
captures their actual windows. Every RGB pixel must match a same-input companion
except the fixed 512-pixel DWM bottom-corner mask. Windows begin hidden and are
shown behind the working window without focus changes.

The additional `--mixed-capture` mode uses a two-record PBR producer and a
full-screen PBR receiver in one Forward Graph submission. Base, normal, and ORM
come from receiver-owned assets; emissive reads the independently named GPU
capture product. Producer and receiver controls have distinct asset identities.
The shared eight-case matrix changes producer base/normal or receiver
base/normal/AO/roughness/metallic independently.

```text
cargo run --locked --features dx12 -- --offscreen --mixed-capture --output target/mixed-capture-dx12
cargo run --locked --features native-gl-wgl -- --visible --mixed-capture --exit-after-capture --output target/mixed-capture-wgl
```

This mode saves complete source inputs and a CPU-only producer reference image
before GPU work. The CPU image is exclusively an oracle artifact; the renderer
never uploads it. Four fixed receiver probes independently evaluate asset
sampling, capture sampling and PBR shading at RGB tolerance three. Surface
capture additionally compares every unmasked pixel with a same-input offscreen
companion. This mode does not establish Deferred capture, multiple producers,
reflection blur, refraction, or a complete material milestone.

`--mip-generation` is a separate acceptance slice through both pipelines. It
supplies only a nonconstant linear RGBA8 4x4 base image and requests three mip
levels. Graph uploads mip zero and generates the next levels with actual GPU
blits. Explicit LOD 0, 1, 1.5 and 2 plus a source-only control are checked against
an independent quantized box/bilinear/trilinear oracle at five fixed probes and
RGB tolerance two. Inputs and predictions precede GPU work; raw bytes precede
verification. CPU-generated reference levels never become upload payloads.

```text
cargo run --locked --features dx12 -- --offscreen --mip-generation --output target/material-mips-dx12
cargo run --locked --features native-gl-wgl -- --visible --mip-generation --exit-after-capture --output target/material-mips-wgl
```

Visible mode retains the same window/companion comparison contract. WGC
excludes the desktop cursor through a verified session setting; the fixed DWM
corner mask and exact RGB comparison remain unchanged. This slice
does not close the full 09 moving tunnel, derivative-based minification, sRGB mip
generation, anisotropic filtering, or per-slot sampler policies.
