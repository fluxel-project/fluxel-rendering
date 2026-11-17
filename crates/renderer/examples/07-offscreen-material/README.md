# Renderer 07 offscreen material

This is the Renderer acceptance consumer adapted from RHI `07_offscreen`: a
dragon capture pass must produce a frame-local color product which a Material
authored mirror receiver samples later in the same Renderer Graph. It is not
proof of refraction, PBR reflection, blur, or a direct RHI workload.

## Current executable scope

The executable validates exact local model bytes and strictly decodes the
selected embedded glTF subset into Renderer geometry: data-URI buffers,
indexed triangle primitives, FLOAT position/normal/UV attributes, and the
two source node TRS transforms. The dragon's first primitive has no UVs, so
only that absent attribute receives explicit `(0, 0)` UVs. Sparse accessors,
matrix transforms, child-node hierarchies, non-uniform scale, and malformed or
out-of-range accessors reject before any GPU allocation. This CPU admission is
not Material, Shader, Renderer, Graph, GPU, image, or window evidence. The
capture consumer builds an immutable dragon producer scene, imported plane
receiver, Material inputs, and asset snapshots from those accepted inputs. The
only CPU texture is the dragon producer's constant albedo; the plane receiver
declares only the `captured_dragon` capture slot. It must not replace capture
sampling with a CPU readback/reupload or a second Renderer submission.

The acceptance framing is intentionally not the reference example's camera:
the dragon uses local XY fitted to clip XY at a forced clip depth of `0.5`; the
reference plane is XZ, so its receiver transform maps XZ to clip XY and also
forces depth `0.5`. The CPU oracle uses those same documented geometric
conventions independently and verifies the actual decoded vertices remain at
the valid depth.

```powershell
cargo +1.87.0 run --manifest-path crates/renderer/examples/07-offscreen-material/Cargo.toml --features dx12 -- `
  --offscreen `
  --dragon ../fluxel-rhi/crates/rhi/examples/assets/models/chinesedragon.gltf `
  --plane ../fluxel-rhi/crates/rhi/examples/assets/models/plane.gltf `
  --output target/renderer-07-offscreen-material
```

This is the real two-attempt offscreen Renderer entry. On Windows with either `dx12` or `native-gl-wgl`,
replace `--offscreen` with `--visible` to create a hidden Host window, submit
the primary and producer-color control capture receivers once each to acquired
Surface leases, complete those same Graphs, and show the window in the
background without activation. DX12 shows it after its first completed frame;
WGL maps its drawable in the background before the first presentation.
The offscreen WGL host remains hidden throughout. Add `--exit-after-capture` for a bounded
automation run; otherwise the visible demo remains open until its window is
closed. The visible path
uses the integration-owned Windows Graphics Capture helper for an own-window
snapshot when the service is available; otherwise it records an explicit
unverified result. It never emulates presentation via CPU upload, a second
submission, or an RHI copy.

Before GPU work the example writes a fixed dense comparison contract and mask.
Each captured BGRA8 window BMP must equal the RGB channels of its same-input
RGBA8 companion readback at every pixel except the documented bottom-left and
bottom-right 16×16 DWM client-corner masks. Surface alpha is intentionally
ignored because the lease is opaque. The BMP remains Windows Graphics Capture
evidence; companion raw is never converted into a screenshot.

Before creating WinRT capture, the Host window, or native GPU objects, visible mode
opts only this fixture process into per-monitor-aware-v2 DPI coordinates and
records that fact in `visible-dpi.json`. This keeps the pixel-sized Surface
lease and Graphics Capture client crop in the same coordinate space without
changing system DPI, focus, or z-order.

## Source and asset provenance

Reference source is `fluxel-rhi` revision `ebdcb614140296197fcf9abe2924ab0faf5cc490`,
particularly `crates/rhi/examples/07_offscreen/{main,resources,workload}.rs`.
The RHI example remains unmodified. Its resource recording is intentionally
not copied: this consumer expresses the scene through Material → Shader →
Renderer → Graph.

The model bytes are not redistributed. RHI records them as unchanged resources
from `SaschaWillems/Vulkan-Assets` revision
`a27c0e584434d59b7c7a714e9180eefca6f0ec4b`; that pack leaves the individual
model licenses unresolved. Locally supplied inputs must hash to:

| Input | SHA-256 |
| --- | --- |
| `chinesedragon.gltf` | `8E69E1A2337BABF6A508BFA44D7AED41F92D85B04C9C3C2EEB5BA4059509A2E5` |
| `plane.gltf` | `F77C1B7D558A26610CDE672EFC53EAC8975CA68AAB631EFE8AAC6D34EE98E10E` |

## Required capture-product contract

The Renderer-owned prototype must create a filterable, renderable capture
target and declare its write before the mirror material's sampled read in the
same Graph. The target/view/sampler, exact snapshots and resolved inputs remain
frame owners until the completion covering the mirror pass. Neither native
handles, a Graph resource, nor a Surface lease can escape to this example.
Cached plans still declare and revalidate the real capture owner every frame.

Reject before the protected allocation/recording operation: zero or mismatched
capture/receiver extent; missing receiver UV or declared texture slot; a
format lacking render+filter support; self-feedback; stale scene/view; and
foreign or mutated actual owner/usage. Resize, source replacement, device loss,
partial recording/submission and failed completion each retain or clear owners
according to Renderer completion evidence—never as a silent visual fallback.

## Acceptance evidence prepared before submission

Deterministic mode first saves the decoded dragon and plane streams, fixed
capture/receiver fit, Material slot contract, primary/control albedos and an
independent CPU oracle, then retains both raw final readbacks. The control
attempt changes only the producer's one-pixel albedo from orange to cyan. The
oracle rasterizes decoded positions and receiver UVs itself, selects four
interior capture footprints and one transparent-clear footprint, and requires
the exact corresponding albedo for every interior receiver probe in each
attempt. A clear-only path, ignored capture, stale source, or wrong texture
binding therefore cannot pass merely by producing different nonempty images.

The renderer must additionally expose evidence that the producer write precedes
the receiver sampled read and that their target owners are distinct. Those
Graph facts are checked by the Renderer capture entry, not recreated by this
example.

Visible mode creates a hidden Host window, submits that same Graph, shows it by
the Host non-activating background operation, and retains an independently
captured native-window image for inspection. That visual evidence complements,
but is not replaced by, RHI readback.
