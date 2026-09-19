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
| Result | `output[id.x] = id.x` over `RWStructuredBuffer<uint>` at `u0` |
| Size | 2876 bytes |
| SHA-256 | `291f5b8a77d86610a0f12da31617f53eb2fdeeed9b5f379283095c85514ad0e9` |
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
