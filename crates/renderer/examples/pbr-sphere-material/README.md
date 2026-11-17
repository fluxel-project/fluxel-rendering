# PBR ORM/IBL sphere material acceptance

This standalone consumer renders one combined six-sphere triangle mesh through
the existing Forward and Deferred paths.  Its `6x1` RGBA8 texture is ORM:
R=AO, G=perceptual roughness, B=metallic.  Each sphere's UV0 is the centre of
its own texel.  The material samples that same image for properties and
occlusion; the base colour is a linear tint and the tangent-space normal is
neutral.

The scene is a perspective right-handed clip-from-world camera with the six
sphere centres baked into geometry. A `--multi-records` mode uses six
independent local geometry and texture assets, different model matrices, and
two compatible materials. Its radiance mip chain
and height-correlated LUT are synthetic control payloads, not a physical
prefilter/LUT generation pipeline.

Run a native backend on Windows:

```text
cargo run --locked --features dx12 -- --offscreen --output target/pbr-spheres-dx12
cargo run --locked --features dx12 -- --visible --exit-after-capture --output target/pbr-spheres-visible-dx12
cargo run --locked --features dx12 -- --visible --multi-records --exit-after-capture --output target/pbr-multi-visible-dx12
```

Before each case allocates resources, uploads, or records, the program persists
raw geometry, ORM, radiance, LUT, VP/camera, material and CPU raster oracle input.
The CPU oracle projects
the original triangles, uses perspective-correct interpolation, samples
binary16 IBL, and probes interiors of all six spheres.  It rejects AO loss,
camera/world-position loss, channel interchange, and missing IBL at the same
tolerance used for companion readback.

The independent predictions account for Deferred UNORM G-buffer channels and
binary16 effects storage separately. Each route checks twelve non-saturated
interior probes covering all six spheres with a predeclared RGB tolerance of
five. Raw readbacks are saved before verification. Visible mode additionally
captures the actual window and compares every RGB pixel with a same-input
offscreen companion, except the fixed 512-pixel DWM bottom-corner mask.

Multi-record mode saves each actual local stream, texture, material, asset
identity/generation, and model mapping separately. Its CPU reference reads
those real sources, independently applies each translation, and derives the
same world-space raster data used for verification. The fourth sphere has no
surface-effects extension, exercising neutral third-MRT output when other
materials require effects. This does not support arbitrary mixed Deferred
lighting families or multiple texture bindings within one material.
