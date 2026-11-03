# Attempt Status

`attempt-001` was captured before the runner verified target Git blobs, fixed
the executable working directory, copied raw object/ECS evidence, and checked
test discovery. It is retained for audit only and is not a complete S4 result.
Use a later attempt produced by the current Python runner for completion review.

`attempt-002` is a retained preflight refusal: an early runner revision included
the untested example README in its target-input set and correctly observed it
differed from the target commit. It did not execute any workload.

`attempt-003` ran the workload and copied raw evidence, but an early recursive
hashing implementation attempted to hash the raw-evidence directory itself.
It is incomplete and not a completion result.

`attempt-004` verifies target blobs, cwd, test discovery, and copied raw
evidence, but predates explicit stale-raw isolation. It is retained for audit
only and is not the final completion result.

`attempt-005` is the first complete retained result. It verifies all executed
tracked Cargo/Rust inputs against target commit blobs, fixes the executable cwd
to this example, records two discovered tests, isolates old raw evidence before
running, and copies newly generated object/ECS raw evidence.

`attempt-006` is an intentional runner failure validation. Its invalid package
selection returns nonzero while retaining error.json, command logs, raw evidence,
and hashes.

`attempt-007` is an intentional unhandled-exception validation. It retains
error.json and hashes after the harness raises.

`attempt-008` predates retaining a copy of the executing harness source and is
historical only.

`attempt-009` is the current complete result. It includes the exact executing
`harness.py` text and hash, target-blob checks, stale isolation, raw evidence,
nonzero test discovery, command logs, and recursive hashes.
