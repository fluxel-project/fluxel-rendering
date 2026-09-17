#!/usr/bin/env bash
# The browser WebGL2 suite, on one of two paths.
#
# Default: the CI recipe -- headless Chrome, the workspace's SwiftShader
# webdriver config -- so a local run and a CI run are the same run.  Pass
# NO_HEADLESS=1 to drive a real headed GPU instead.
#
# The headed path does not run *here*: NO_HEADLESS is interactive mode, where the
# runner serves its harness and waits for a browser, and this script would appear
# to hang.  `wasm_test_headed.py` is that path -- it starts this same cargo line
# with NO_HEADLESS set and drives cdp_shot.py against the harness (EXP-031).
set -u
cd "$(dirname "$0")/../.."

# A download rather than source, so it stays under the regenerable tree; point
# elsewhere with FLUXEL_CHROMEDRIVER_DIR.
DRIVER_DIR="${FLUXEL_CHROMEDRIVER_DIR:-$PWD/target/tools/chromedriver-win64}"
export PATH="$DRIVER_DIR:$PATH"
export CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner
export WASM_BINDGEN_TEST_TIMEOUT=120
export WASM_BINDGEN_TEST_DRIVER_TIMEOUT=90
export WASM_BINDGEN_TEST_WEBDRIVER_JSON="$PWD/scripts/webgl2_webdriver.json"
[ "${NO_HEADLESS:-0}" = "1" ] && export NO_HEADLESS=1 && unset WASM_BINDGEN_TEST_WEBDRIVER_JSON
cargo test -p fluxel-rhi --target wasm32-unknown-unknown --no-default-features --features webgl2 --lib --locked 2>&1
