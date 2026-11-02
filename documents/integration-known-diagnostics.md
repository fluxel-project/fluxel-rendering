# Historical foundation integration diagnostics

The Renderer and its integration consumer have been reset. This ledger records
diagnostics from a removed workload only; it is not a runnable procedure, a
current validation waiver, or evidence for the current checkout. A rebuilt
consumer must establish and review its own diagnostics before any support claim.

## DX12 offscreen clear optimization warning

The same-format offscreen image capture may emit D3D12 message ID 820 at warning
severity: `ClearRenderTargetView` reports that the requested clear value differs
from the optimized clear value supplied at resource creation. The selected
DX12 backend creates ordinary color render attachments with optimized clear
`[0,0,0,0]` (`crates/dx12/src/backend/texture/mod.rs` in RHI); this frozen workload
requests `[0.02,0.03,0.12,1]`. The diagnostic itself states that clearing still
produces the desired value and may be slower.

For this correctness baseline, retain and classify this exact warning, together
with the independent pixel oracle. Do not suppress the callback, change the
frozen expected image to hide it, or call the run diagnostic-free. It is not a
memory-lifetime, state-transition or device-loss diagnostic. Other messages
remain unreviewed failures until separately investigated. No RHI interface
change is required for this warning; any future RHI interface proposal requires
prior user consultation.

Performance measurements include the current behavior. This baseline does not
claim clear optimization or a performance comparison.

## Window capture service availability

The Windows Computer Use native pipe was unavailable during the first image
oracle development run. Both the prescribed reconnect sequence and the alternate
computer-use entry point reported a missing native pipe. The window's actual
presentation could not be inspected through that service. Inspected offscreen
readback PNGs establish only their captured image; they leave actual-window
screenshots, resize presentation and visible flicker evidence open.

## WGL debug-output observation

The executable uses the existing RHI device-native interop lease to install an
observer before normal Renderer work, and a second short lease to restore the
previous callback, user pointer, and debug-output flags afterward. It never
keeps that lease through Renderer operations or changes the RHI interface.
The selected AMD OpenGL 4.6 core context reports no debug-context flag; this is
recorded separately from successful core debug-output callback activation.

Activation requires observing the exact application marker `Fluxel WGL observer
activation`, with source APPLICATION, type MARKER, id `0x46580001`, and severity
NOTIFICATION. Installation alone is not activation evidence. The bounded
observer retains source/type/id/severity and raw message bytes in
`wgl-debug-messages.jsonl`. Lost, contended or truncated messages and failed
restoration invalidate the run's validation result.

RHI's fixed pass labels also produce application PUSH_GROUP and POP_GROUP
notifications: id zero, severity NOTIFICATION, and exact label `renderer fixed
compute` or `renderer fixed triangle raster`. Classify only this exact tuple as
application instrumentation, and require a balanced, matching label stack. These
notifications are not driver warnings. Preserve every raw notification; a
different source, type, severity, id, or label remains an unreviewed failure.
No driver diagnostic is waived by this classification. An unavailable observer
route leaves validation unverified, even if pixel and completion oracles pass.
