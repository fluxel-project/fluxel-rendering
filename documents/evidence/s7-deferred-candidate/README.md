# S7 Deferred Closure Evidence

`native-evidence.zip` is the immutable S7 native evidence archive. It contains
six first-attempt Windows-native captures: Deferred offscreen, Deferred
Surface/resize, and Deferred device-isolation for each of DX12 and desktop
OpenGL through WGL. Every attempt captured the same 503 dirty-worktree source
inputs at HEAD locator `c163db2362154fbb75289a6a5263312b69350d1e`; the locator
is not a claim that committed HEAD contains the exact source. The archived
inputs are authoritative.

| Property | Value |
| --- | --- |
| Bytes | 39,603,597 |
| Members | 789 |
| SHA-256 | `d56a35e791f0733d0407fc0b4f8cd3b4dae803f06261fd5902e2402300927b81` |
| Review result | 14/14 checks passed |

[`archive.json`](archive.json) records the archive identity; [`archive-review.json`](archive-review.json)
records the final self-contained review. The archived `reviewer-v2` tools were
extracted and executed independently of the worktree while the original runtime
directory was unavailable. Four post-run reviewer-only files and five focused
reviewer checks are retained separately: they correct actual deformation pending
schema handling and relocated artifact hash binding. Native Renderer/workload
inputs did not change.

The archive also retains both backend Surface BMP montages, opened during review,
and raw/BMP oracle evidence with the applicable alpha rules. This evidence is a
bounded S7 acceptance record; it does not certify other backends, browser hosts,
Unity/Babylon parity, S8, final regression, repository reconciliation, commit,
or push. See the [S7 closure record](../../s7-deferred-closure.md).
