# Renderer examples

This directory contains standalone consumers of public Renderer APIs:

- `02-texture-material/` runs the same textured material through Forward and
  Deferred, with numerical readback and real-window comparison. Its
  `--srgb-filter` control checks hardware RGB decoding before interpolation.
- `04-dynamic-uniform-material/` adapts the original cube grid with 125 draws,
  dynamic uniforms, vertex color, and actual Renderer-owned Forward depth.
- `07-offscreen-material/` captures a producer and samples its GPU image in a
  material receiver through one Graph, including bounded Surface presentation.
- `09-mipmap-tunnel-material/` exercises perspective tunnel sampling across
  generated mip levels, with filtering probes and independent readback checks.
- `pbr-environment-material/` isolates selected split-sum IBL radiance mips,
  BRDF LUT, and orthographic camera controls.
- `pbr-four-texture-material/` binds four named PBR images to two real objects,
  testing sRGB/data roles, normal unpacking, ORM channels, shared sources and
  aliases with independent numeric and window comparisons.
- `pbr-sphere-material/` combines six sphere surfaces in one mesh and exercises
  perspective shading, packed ORM, tangent-space normals, diffuse irradiance,
  and selected specular IBL through both pipelines and native backends.
- `external-renderer/` checks RenderScene, CPU Feature, controlled Forward
  composition, and private-boundary compile failures without creating a device.
- `rendering-integration/` owns the Windows DX12 and desktop OpenGL/WGL fixture
  executables and their independent locks.

Platform event loops and Surface lifecycle composition stay in these consumers,
not in the Renderer library. They must use the public facade and cannot access
private Graph, allocator, encoder, or submission state.

The sphere consumer supports both one combined mesh and `--multi-records`
with six independent geometry/texture owners and model matrices. Deferred
multiple records currently share the UV0-normal metallic/roughness packing and
lighting family; arbitrary mixed surface families remain outside this scope.
Synthetic environment and LUT controls do not
establish a physical prefiltering or BRDF-integration generator.

The four-texture consumer also provides `--mixed-capture`: a Forward-only PBR
receiver combines base/normal/ORM assets with a same-frame GPU capture as
emissive. Its independent producer/receiver controls and numerical oracle are
separate from the ordinary Forward/Deferred asset matrix.

The four-texture consumer's independent `--mip-generation` mode supplies only a
linear RGBA8 base level and samples GPU-generated levels at explicit LOD through
Forward and Deferred. This precedes the complete 09 tunnel/filtering consumer.
