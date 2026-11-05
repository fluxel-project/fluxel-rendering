# S0–S8 Review Follow-up

This review starts from Rendering `ec05ae4` and Bases `6c80772`. Stage names
retain the bounded meanings in [the implementation plan](version-plan.md).
Historical native and performance archives retain their original source and
dependency identities; they do not certify this follow-up's changed sources.

## Correctness and engineering gates

The reported texture access defect is valid. The mip partitioner included
endpoints from declarations outside the requested interval. A pass declaring
`0..1` and `3..4` could therefore reject a valid request for `0..1`. Only
nonempty intersections may partition the requested mip interval. Layer and
aspect coverage must still hold independently in every resulting partition;
separate declarations cannot fill a missing mip or layer by implication.
Regression tests exercise the real declared-access validator, including legal
disjoint declarations and illegal gaps, rather than only its interval helper.

The new [CI workflow](../.github/workflows/ci.yml) runs on pushes, pull requests
and manual dispatch. Windows jobs run workspace behavior and documentation
tests, the independent external consumer (including diagnostic-checked private
API compile failures), and DX12/WGL integration CPU contracts on Stable and
Rust 1.87. Stable additionally checks formatting and Clippy. Python discovery
checks the integration and benchmark tooling. These jobs create no GPU device
or visible demo and are not hardware acceptance. Branch-protection policy and
native hardware certification remain separate from installing the workflow.

Enabling these checks also exposed preexisting formatting differences and two
`cloned_ref_to_slice_refs` test warnings. The affected files are formatted and
the two test inputs now borrow their original descriptor without cloning.
The independent consumer's complex scene return type also has a local alias;
its behavior and public API are unchanged.

The Bases warning is real on GitHub's Stable 1.99.1, although local Stable
1.98 does not reproduce it. Run `42009617744` failed on `fetch_update` at
`6c80772`; the earlier S2/S3 review's toolchain-specific result was not a
current green-CI claim. Bases `124e825f82e5cd0eb485c95fe1281b11c8dbc844`
uses a relaxed compare-exchange loop instead, retaining unique claimed IDs,
contention retry and exhaustion rejection without requiring a post-MSRV API.
Its [CI run](https://github.com/fluxel-project/fluxel-bases/actions/runs/42018516642)
passes both quality and Rust 1.87 jobs.

Rendering selects that exact Bases revision in all four consumer manifests and
all five Cargo lockfiles, updated with targeted Cargo resolution. No unrelated
package versions change. RHI remains
`b5d569feace548a6ed498901deaf31a5efcaca61`; the native executable still selects
Host `8d101624c603a6c1e0badf729768c6b26e70fba8`. The runnable cross-repository
artifact is `examples/rendering-integration` (owned by Rendering); its DX12/WGL
CPU contracts and the independent external consumer validate the changed pin.
This review changes no backend recording, presentation or shader behavior and
does not claim refreshed GPU/performance certification for the new pin.

The Python aggregate failure was also valid: discovery imports `smoke_tests` as
a top-level package, where its relative parent import cannot resolve. The
support module now selects its import mode explicitly, preserving package
execution and making the aggregate command itself usable. Historical S8 logs
remain unchanged, including their originally recorded aggregate failure.

## Current maintenance validation

Local Rust 1.87 validation on the updated pins passes 310 workspace behavior
tests (one diagnostic remains intentionally ignored), 45 DX12 integration CPU
tests and 48 WGL integration CPU tests. Stable Clippy passes for all workspace
targets and the independent consumer; workspace formatting passes. Stable's
independent consumer passes ten behavior tests, including both private-API
compile failures. Rust 1.87 workspace doctests also exit successfully but
discover zero cases, so they add no behavioral coverage. Benchmark-tooling
discovery passes 25 tests. CI repeats the defined matrix on the pushed source;
its status must be read from the commit's workflow rather than inferred from
these local results.

The repaired aggregate integration command discovers 117 tests: 116 pass and
one Windows symlink-privilege case is skipped. Its successful aggregate run
replaces the need to combine separate package logs for this maintenance check.

## Review judgments and future work

| Finding | Judgment and next evidence |
| --- | --- |
| Fixed Feature/pass structure | Accurate limitation, already outside the bounded S6/S7 closure. Before expanding the facade, choose a real external producer and downstream pass consumer; prove declarations, current-frame cache-hit validation, failure cleanup and completion ownership through the same Graph. No new registry, scheduler, public Graph or speculative generic framework is justified by this review. |
| Large evidence archives | Git-tree bytes are dominated by retained evidence. Keep existing archives immutable. A future storage migration must first prove durable retrieval, source/lock identity and SHA-256 verification from a clean checkout. No archive or Git history is removed here. |
| Shader composition cost | Repeated Naga capability validation and frame-local composers are visible in source, but do not establish a performance regression. Measure representative variants, cold/warm composition and composer hits before choosing Renderer-level caching. No new performance claim or cache is introduced. |
| Forward/Deferred duplication | Similar preparation/cache paths alone do not establish a correctness problem or justify merging ownership protocols. Future reduction needs a concrete duplicated invariant and tests preserving admission, actual-owner facts, failure cleanup and retirement. |
| S8 measurement | Retain the selected CPU/API validation-index optimization and its workload-specific historical result. No GPU-duration, VRAM or whole-frame speedup is inferred. |

The original S0–S8 archives remain evidence for their named subsets, including
the fixed-light, one-mesh Deferred and selected position-deformation paths.
General pass chains, general GPU products, complete PBR, LOD, streaming and GPU
culling remain unimplemented or unaccepted as described in the plan. This
maintenance review does not authorize those as new implementation scope.
