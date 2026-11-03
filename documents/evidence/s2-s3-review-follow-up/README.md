# Native S2/S3 review-follow-up evidence

`native-evidence.zip` retains the complete selected DX12 and desktop WGL
attempts, the original integration manifest, and their external review records.
`index.json` lists every ZIP member, its original path, byte length and SHA-256,
as well as the archive SHA-256. All 423 members were checked against their
original bytes after compression. Original JSON and absolute paths are retained
unchanged as historical facts; extraction does not rewrite them into new runs.

The original manifest also lists earlier attempts that are not selected by the
review; only the two attempt IDs in the index are retained in this archive.
The selected attempts contain their exact source archives, independent locks,
executables, native diagnostics, workloads, raw inputs/expected/readbacks,
window captures and focused-test logs. The review identifies the limited
Windows CPU/GPU and device coverage. This archive does not certify other
backends or a later source revision.

See `../../s2-s3-closure.md` for the assessment and boundaries. The archive is a
retained copy of the original run, not a claim that its absolute paths exist on
another checkout.
