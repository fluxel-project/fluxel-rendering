# Releasing Fluxel Rendering

A rendering release is identified by one workspace package version and an
annotated tag, such as `v0.16.0`. The independent `fluxel-rhi` has its own releases;
its version is not coupled to the rendering workspace version. Source consumers
must pin tags or revisions.

## Candidate inputs

Before tagging, record:

- the exact Rendering candidate and participating Bases/RHI/Host/JSBridge commits;
- package versions, selected Cargo features, toolchains, and relevant lock hashes;
- the runnable integration artifact and independent public-API consumer;
- supported and affected targets, backend/device/runtime identities, validation
  settings, unsupported routes, and required missing evidence; and
- commands, fixed workloads, expected outputs, and the evidence retention paths.

Refresh every affected manifest and lockfile, including independent integration
consumers. Build the actual producer artifact with those consumers. A dependency
change must not be certified from separate green library tests.

The integration artifact owner maintains a machine-readable closure manifest.
The integration runner writes workload-scoped records under
`target/integration/<stage>/<rendering-sha>/manifest.json`; the selected workload makes the path and
commands executable. Evidence records must remain retrievable and attributable
to the tested commits after a release. Large generated artifacts need not enter
Git, but deleting their only retained copy invalidates reproducibility claims.

## Verification gate

Follow the ecosystem [evidence policy](https://github.com/fluxel-project/.github/blob/main/EVIDENCE_POLICY.md).
During development, group coherent changes and run only necessary affected
checks after the batch is complete. Full regression is reserved for a completed
major refactor or complete goal implementation and closure. A release review
runs the complete affected target matrix after required implementation is
finished; routine batches and individual edits do not justify full runs.

The release candidate must pass formatting, relevant Clippy/compile checks,
focused contract/regression tests, and all required real-target runs. Check that
the intended cases actually ran; zero matched tests do not establish a pass.
Preserve diagnostics and partial artifacts when execution fails.

For graphics behavior, inspect actual output. For continuous frames, resize,
loss/recreation, or multi-frame reuse, retain sufficiently dense frame/readback
or recording evidence to detect stale contents and intermittent failures. A clean
validation log, surviving process, or successful present call is insufficient.

Exercise the selected RHI's acceptance/completion outcomes, safe resource
retirement, and the affected public extension consumer. Once extension APIs are
implemented, that consumer must use the current static generic Feature/Pipeline
facade with no trait-object registry; foundation fixtures do not prove this
later contract. Browser evidence requires
the actual WASM artifact and matching JSBridge revision. A named browser does not
certify another browser or a mini-game host; emulator evidence does not replace a
required physical-device run. Missing required evidence leaves its support claim
open. Unsupported cases must return the documented structured result.

Performance claims additionally retain reproducible baseline/candidate workloads,
raw metrics, measurement procedure, and declared thresholds. Do not substitute a
counter of successful API calls for GPU timing or memory behavior.

## Publication

After all required checks pass, ensure the candidate source and dependency inputs
are committed. Bind evidence to those exact commits, create an annotated tag, and
push the branch and tag. Verify the remote branch/tag identities. Do not create a
release tag for a failing or unverified required matrix.

The annotated tag is the release identity; a GitHub Release object is not required.
Ordinary documentation commits and local planning updates do not require a release
tag and do not imply that the described implementation gates have closed.
