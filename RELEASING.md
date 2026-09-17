# Releasing Fluxel Renderer

Each workspace release uses one package version and one annotated tag. For
example, package version `0.7.0` is released as `v0.7.0`; the tag points to the
same commit consumed by all three Git dependencies.

Before creating that tag, commit the release candidate and make sure the
checkout is clean. On a Windows `x86_64-pc-windows-msvc` machine with both
DX12 and Vulkan validation available, run:

```powershell
./scripts/conformance.ps1
```

The script is deliberately fail-closed: it rejects a dirty checkout, a GNU or
other non-Microsoft host/target, an invalid `HEAD`, and a conflicting
`CARGO_BUILD_TARGET`. It runs all workspace ignored fixtures, stores the exact
SHA and test command in `target/conformance/<sha>/manifest.json`, and keeps
the combined Cargo log there even if a fixture fails. Inspect the manifest and
log before tagging. They stay out of the source commit so that the tested SHA
does not change, but both files must be uploaded as assets of the GitHub Release
for the matching tag. A release is incomplete until those durable asset URLs
exist and identify the tagged commit.

After the conformance gate and the required platform checks pass, create an
annotated tag, push the branch and tag, then verify that `origin/main` and the
remote tag resolve to the same release commit. Create the GitHub Release and
attach that commit's `manifest.json` and `cargo.log`. Do not create or push a
release tag when the hardware gate has failed or could not run.

When a release changes the browser executor, also build the release WASM for
the exact candidate commit and run the named-browser evidence harness against
the matching `fluxel-jsbridge` candidate. Archive its manifest, representative
screenshots, and browser diagnostics with an explicit SHA-256 checksum; record
both repository SHAs. Browser evidence is additional to, never a replacement
for, the native DX12/Vulkan gate above.

When a release changes the GL family, the same candidate also needs the
GL-specific gates, in this order:

```powershell
python scripts/check_gl_architecture.py
python scripts/check_desktop_gl4_conformance.py --report scripts/tests/data/desktop_gl4_radeon_780m.json
python scripts/check_gl4_raster_readback.py
```

The first is the three-layer import boundary and runs anywhere. The second
re-adjudicates a durable conformance report on a machine with no GPU, so CI can
hold a driver claim to its numbers; a report that no longer adjudicates green
must be re-measured, not re-labelled. The third drives a real desktop GL context
and requires the driver and GPU to be recorded with it. A frame that comes from
a GLES implementation reached through EGL, or from a browser, is evidence about
that implementation and must say so rather than being filed as desktop GL
evidence. Do not create or push a release tag when any of these has failed or
could not run.
