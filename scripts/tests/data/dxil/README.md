# DXIL fixtures

Checked-in DirectX Intermediate Language blobs. **No test ever invokes `dxc`** — a
test that generated its input could not tell "the driver accepted this program"
apart from "the toolchain on this machine produced something", and the fixture
would change under the test without anyone deciding to change it.

## `fill_cs.dxil`

A storage-buffer write kernel. It is the compute closure's payload: dispatch it,
read the buffer back, and compare against what the CPU expects — which is the
`version-plan.md` section 4 compute requirement met with a read-back result rather
than with an absence of errors.

| | |
|---|---|
| Profile | `cs_6_0` |
| Entry point | `main` |
| Thread group | `(8, 8, 1)`, declared by `[numthreads(8, 8, 1)]` |
| Result | `output.Store(id.x * 4, id.x)` over `RWByteAddressBuffer` at `u0` |
| Size | 2752 bytes |
| SHA-256 | `0c569075a49592f8fa162c49420b674eca5eee89a61aceb38210c08502ff32a4` |
| `dxc` | `dxcompiler.dll 1.9(5399-a107ba61)` / `1.9.0.5399`, from Vulkan SDK `1.4.357.0` |

Regenerate:

```sh
dxc -T cs_6_0 -E main -Fo fill_cs.dxil fill_cs.hlsl
```

`fill_cs.hlsl` is the source, kept beside the blob so that what the bytecode *is*
can be read without a disassembler. `dxc -dumpbin fill_cs.dxil` reports the
profile, entry point and thread group that the table above records.

## Why the source is not the test input

A `ShaderArtifact` carrying HLSL source would be a different artifact: the RHI's
portable vocabulary has no HLSL form (section 19.2), and DXIL is what a DX12 device
consumes. Keeping both files makes the provenance legible while leaving exactly one
of them as test input.

## `triangle_vs.dxil`, `solid_ps.dxil`, and `sampled_ps.dxil`

These are the vertex and fragment halves of the real DX12 offscreen-raster
evidence. `triangle_vs.dxil` consumes `float2 LOCATION0`; `solid_ps.dxil` writes
the fixed RGBA value `(0.25, 0.5, 0.75, 1.0)`; and `sampled_ps.dxil` samples
`Texture2D t0, SamplerState s1` at `(0.5, 0.5)`. The test uploads a 2×2 image and
reads the raster target back, so the three fixtures jointly exercise vertex input,
graphics PSO, root signature, descriptor tables, sampler heap, draw, RTV and copy
readback.

Generated with Vulkan SDK `1.4.357.0` `dxc.exe` (`dxcompiler.dll 1.9.0.5399`):

```sh
dxc -T vs_6_0 -E vs_main    -Fo triangle_vs.dxil triangle.hlsl
dxc -T ps_6_0 -E ps_solid   -Fo solid_ps.dxil    triangle.hlsl
dxc -T ps_6_0 -E ps_sampled -Fo sampled_ps.dxil  triangle.hlsl
```

| Fixture | SHA-256 |
| --- | --- |
| `triangle_vs.dxil` | `803cc49ee00ef619337d695ac1d5e1033e19006ddfe70456c6804f498cb5f925` |
| `solid_ps.dxil` | `ca064da4429dd9aa000cdb9854b24ba5296732041bac96b69d33211fb0e56bca` |
| `sampled_ps.dxil` | `77a957db2680eb8e98a54eafcd3bda3f2c48dfb95824a88c7b79d0442b31df33` |

Tests only `include_bytes!` the compiled blobs and never invoke `dxc`.
