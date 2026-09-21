# ADR-0008: Execute platform-specific test paths natively

**Status:** Accepted

## Context

Conditional compilation can leave platform fallbacks and feature combinations
untested. The Linux all-features Clippy failure in the non-Windows RHI stub
showed that Windows cross-compilation does not exercise the Linux path.

## Decision

Use Windows MSVC as the Windows development and native GPU target. Run Linux
workspace tests and `clippy -D warnings` on a native Linux environment such as
WSL2/Ubuntu, not merely as a Windows cross-target. Test reachable default,
all-feature, no-default-feature, and test-support combinations, including
structured platform-stub errors. Android support, when introduced, requires an
explicit target/NDK check and executable emulator or device tests for platform
code.

## Alternatives

- Treat a Windows build as coverage for `cfg(not(windows))` code.
- Require only that platform stubs compile.

## Consequences

CI gains a Windows workspace gate in addition to backend feature gates, while
Linux remains a native gate. Platform tests distinguish `BackendDisabled` from
`PlatformUnsupported`; GPU correctness still requires target hardware.

## Evidence

The 0.2.8 implementation removed the four unused non-Windows test-support stub
warnings under Linux all-features Clippy and added Windows workspace and
no-backend gates. The no-backend gate verifies the structured
`BackendDisabled` result for both native backend requests.

See [RHI design](../../crates/rhi/documents/design-rhi.md) for the current platform policy.
