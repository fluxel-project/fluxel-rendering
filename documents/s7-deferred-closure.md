# S7 Deferred Acceptance Record

**Decision: accepted for the named Windows DX12 and desktop OpenGL/WGL S7
subset.** This is a scoped Deferred-pipeline closure, not a general rendering,
backend, or release claim. It does not close S8, final full regression,
six-repository documentation reconciliation, commit, or push.

## Demonstrated scope

The selected consumer uses one mesh and shared scene/material inputs through a
real Graph route: upload -> optional GPU deformation -> G-buffer -> fixed-light
lighting -> readback, or the corresponding Surface presentation route. An
independently built static external Feature supplies the optional deformation;
it has no Graph, allocator, encoder, GPU-owner, or submission authority.

The Material/Shader portion demonstrates validated `CodeNode` composition and a
selected opaque metallic/roughness PBR consumer. The light and view are fixed.
This acceptance does not claim IBL, normal maps, emissive inputs, arbitrary SRP
chains, Forward+, shadows, PostFX, general deformation, or a general PBR
authoring contract.

The retained workload set also exercises cache reuse and replacement, Surface
presentation and resize, device isolation and generation replacement, pending
owner/completion handling, and structured failure/refusal boundaries. The
focused negative cases protect allocation, mapping, recording, or submission as
applicable; they are not substituted for native GPU attempts.

## Evidence and source identity

The authoritative artifact is
[`native-evidence.zip`](evidence/s7-deferred-candidate/native-evidence.zip):
39,603,597 bytes, 789 members, SHA-256
`d56a35e791f0733d0407fc0b4f8cd3b4dae803f06261fd5902e2402300927b81`.
Its external archive descriptor is
[`archive.json`](evidence/s7-deferred-candidate/archive.json), and its final
review result is
[`archive-review.json`](evidence/s7-deferred-candidate/archive-review.json).

The six first-attempt native captures are one Deferred offscreen, one Deferred
Surface/resize, and one Deferred device-isolation workload for each of DX12 and
WGL. Each records 503 identical frozen source inputs. Capture occurred in a
dirty worktree whose HEAD locator was
`c163db2362154fbb75289a6a5263312b69350d1e`; that commit is only a locator, not
a claim that it contains the captured S7 source. The exact bytes used are the
per-attempt archived source inputs. The pinned RHI
`b5d569feace548a6ed498901deaf31a5efcaca61`, Bases
`ee640d57a8d7c32d251252d0e8b32950636a770c`, and Host
`33a6bbb3fa15500e928b4192ca8f280b735034d9` checkouts were verified unchanged.

The first focused run passed 27 native, 62 Renderer, 10 external-consumer, and
18 integration-tool checks. After native capture, four reviewer-only files and
five focused reviewer checks were retained separately. Those corrections align
the reviewer with the actual deformation pending schema and fix relocated raw
artifact hash binding; no native Renderer or workload input changed.

## Independent review and visual evidence

The archive contains the original reviewer report plus the final reviewer tools
under `reviewer-v2/`. The latter were extracted from the archive and run without
worktree tools while the original runtime directory was temporarily unavailable.
All 14 checks passed, including the closure ledger, all six provenance/command/
diagnostic checks, same-source matrix, and all six named
input/readback/lifecycle checks. The earlier relocation failure remains retained
as history; it exposed an absolute-path matching error in the reviewer rather
than a native-runtime change.

The DX12 and WGL six-frame Surface BMP montages were opened. In addition to that
visual inspection, the independent retained-input raw/BMP oracle passed under
the documented exact-background and alpha rules.

## Limits and follow-up

This acceptance covers only the six retained Windows-native attempts and their
named inputs. It makes no claim for Vulkan, browser WebGPU/WebGL, Metal,
browser hosts, Unity/Babylon parity, or other backends. Subsequent measured S8
strategies and final verification are retained separately in the
[S8 closure record](s8-measured-closure.md); they do not rewrite this archive's
source, historical capture method, or accepted scope.
