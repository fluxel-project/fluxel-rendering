# Renderer 02 texture material

This independent example adapts the RHI `02_texture` scene as a Renderer
consumer.  It deliberately does **not** reuse its direct RHI pipeline,
bindings, recording, or submission: `MaterialNodeGraph` and `MaterialCodeNode`
are compiled by the public Material → Shader → Renderer → Graph route.

The quad is expanded from the source indexed four-vertex geometry because the
current public `GeometryData` path is non-indexed.  Its positions and UVs keep
the original RHI scene's coordinate/UV convention.  The sample uses the exact
KTX input only after verifying its SHA-256 before it creates an asset snapshot
or opens a GPU device.

## Asset provenance

`metalplate01_rgba.ktx` is intentionally not copied here.  RHI records it as
an unchanged byte copy from `SaschaWillems/Vulkan-Assets`
`a27c0e584434d59b7c7a714e9180eefca6f0ec4b`; that pack does not establish a
redistributable license for this file.  Use a locally fetched input from the
RHI checkout, revision `ebdcb614140296197fcf9abe2924ab0faf5cc490`, with SHA-256
`38495B089915D38E2AE6CE14CDF872FF7D2DF6F79F3A1E5A7A7851CBE65882BD`.

The earlier `target/e0-texture-material-dx12-06` run was a development
diagnostic and is not acceptance evidence: later source changed the Deferred
path and the retained directory does not freeze this example's complete source
and lock input. A new attempt must use the controlled runner below: it freezes
the complete workspace source and the actual example lock before execution,
then retains raw outputs and verifies post-run source stability. The executable
itself writes GPU-precondition inputs/expectations before submission and raw
outputs only after completion. Do not amend a previous attempt.

```powershell
python tools/integration/m0_sampled_attempt.py --backend dx12 --workload e0-texture-material-forward-deferred --asset ../fluxel-rhi/crates/rhi/examples/assets/textures/metalplate01_rgba.ktx --attempt-dir target/e0-texture-material-source-bound-01
```

The runner is the required evidence entrypoint for this E0 workload. It retains
and runs a copied Cargo executable, checking its hash before and after execution.
Diagnostic activation and window inspection require their own evidence procedures.

## Synthetic sRGB filter acceptance

`--srgb-filter` selects a built-in non-constant 2×2 RGBA8 input and therefore
does not accept or require `--asset`.  This is a separate consumer of the same
`ColorUv0` `textureSample` Material route, including Forward, Deferred, native
window presentation, and their same-input companions.  Its texture is declared
`TextureEncoding::Srgb`; the CPU oracle applies the IEC sRGB piecewise decode
to each colour texel before clamp-linear filtering (alpha remains linear).

The synthetic quad offsets U only in this mode so readback pixel `(319, 0)` has
exactly `U=0.5`; its `V=0.5/360` clamps to row zero, producing the midpoint
between the black and white texels.  Its
expected red result is linear `0.5` (128 in the linear UNORM readback).  A
second non-binary channel probe rejects both an ignored decode (byte 128) and
filter-then-decode (about byte 55 at the black/white midpoint).  The executable
writes texture bytes, encoding, shifted geometry, and predeclared expectations
before device work; readbacks are saved before their fixed one-code oracle.

Use the dedicated runner workload, which passes `--srgb-filter --noasset`:

```powershell
python tools/integration/m0_sampled_attempt.py --backend dx12 --workload m1-srgb-filter-forward-deferred --attempt-dir target/m1-srgb-filter-source-bound-01
```

## Permuted Material bindings

`--permuted-bindings` keeps the Material names and `ColorUv0` shader exactly
the same, but constructs the actual group-zero resources as dense bindings
`material_texture=0`, `material_sampler=1`, and `material_tint=2`.  The
default remains `tint=0`, `texture=1`, `sampler=2`.  It may be combined with
`--srgb-filter`; both Forward and Deferred, including their native presentation
and same-input companions, consume the selected definition.  `source.json`
records the selected layout and the pre-submission `material.txt` contains the
actual resource locations.

For a bounded native window acceptance run, invoke the copied executable with
`--visible --exit-after-capture`; the example accepts either `--features dx12`
or `--features native-gl-wgl`. WGL maps the drawable in the background before
the first present for its driver-specific first-frame rule, while offscreen WGL
keeps its Host window hidden.

`--offscreen` writes source identity, decoded pixels, material/geometry inputs,
and independent CPU expected-probe files before GPU submission. Each public
Forward and Deferred Renderer readback is saved before its oracle runs.
The oracle samples the saved KTX bytes using the
documented UV interpolation and the public Forward sampler's clamp/linear/LOD0
contract, then compares edge and non-constant horizontal/vertical regions with
a one-code Forward tolerance. Deferred uses the same ColorUv0 root through its
UV0 G-buffer, records its existing flat-plane lighting equation separately, and
allows one code for the fixed flat-color RGBA8 path. A one-to-one composite
samples pixel centers without an additional half-texel offset; G-buffer storage
must preserve this scene's already quantized color. This bound does not apply to
general PBR lighting. The example intentionally does not claim the RHI source shader's
camera transform or lighting semantics: Renderer uses the same source quad/UV
scene through its `ColorUv0` material output. `--visible` opens a real DX12 or
WGL surface, submits that same scene through `submit_forward_surface` then
`submit_deferred_surface`, and shows the final Deferred presentation with the
host's non-activating background display operation. `--exit-after-capture`
creates a bounded acceptance run: each presented route is captured with Windows
Graphics Capture and compared at every unmasked RGB pixel with an independently
submitted same-input offscreen readback. The fixed DWM mask excludes only the
bottom 16x16 client corners; opaque Surface alpha is not compared. It is a
visual consumer; its companion does not replace its window evidence.
