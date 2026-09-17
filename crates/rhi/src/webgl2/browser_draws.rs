//! Driving a live browser WebGL2 context through the compatibility adapter.
//!
//! # What this is the other half of
//!
//! `Checkpoint G` measures the same workload on two surfaces.  The native one is
//! [`crate::webgl2::compat`]'s desktop entry, which opens a WGL context over the
//! caller's drawable and drives the frame inside the crate.  This is the browser
//! equivalent: a canvas-backed `webgl2` context, the browser provider over it,
//! and the *same* workload -- `compat::drive_workload`, one implementation, so
//! that the cached-versus-uncached differential compares two contexts rather
//! than two frames.
//!
//! # Why it is a module here and not a test inside a layer
//!
//! The two layers it joins cannot name each other, and `scripts/check_gl_architecture.py`
//! enforces both directions: `compat` may not name `web_sys`, `js_sys` or
//! `wasm_bindgen`, and `api/browser` may not name `compat`.  So the test sits
//! above both, where naming either is allowed -- the same place
//! [`crate::webgl2::conformance`] sits for the native side, and for the same
//! reason: a module that consumes two layers may not be inside one of them.
//!
//! # Why it is only a test, and why nothing here is an API
//!
//! The production browser cutover is F6(c), which is 0.16's decision, so nothing
//! outside this crate consumes a browser GL-family device yet and this module
//! declares no entry beyond its own fixture.  It drives the adapter *directly*,
//! which the checkpoint's own plan note says is possible and is why the browser
//! half is not gated on that cutover.
//!
//! # What it does not produce, and why no screenshot is owed
//!
//! It renders a four-by-four off-screen texture and reads counters, so there is
//! no window, no presented frame and no image for a reviewer to open.  CLAUDE.md
//! §5 asks for visual evidence from tests that produce a picture; the evidence
//! this surface owes is the counter differential, and the picture-producing
//! browser path -- the retained 0.14 residency suite and the JSBridge smoke --
//! is where frames are actually looked at.

#![cfg(all(target_arch = "wasm32", feature = "webgl2"))]

use wasm_bindgen::JsCast as _;
use wasm_bindgen_test::*;

use crate::webgl2::api::{ContextEpoch, ContextStamp, DeviceIdentity, WebGl2BrowserDiscovery};
use crate::webgl2::compat::{DrawCost, drive_workload};
use crate::webgl2::state::ExecutionMode;

wasm_bindgen_test_configure!(run_in_browser);

/// A live browser provider on a fresh canvas of its own.
///
/// One canvas per test, so a test's retained objects and context state cannot
/// reach another's, and a fabricated stamp, because discovery evidence is bound
/// to whatever stamp its owner supplies rather than to one the driver reports.
fn open() -> WebGl2BrowserDiscovery {
    let document = web_sys::window()
        .expect("browser window")
        .document()
        .expect("browser document");
    let canvas = document
        .create_element("canvas")
        .expect("create canvas")
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .expect("canvas element");
    canvas.set_width(4);
    canvas.set_height(4);
    let stamp = ContextStamp::new(
        DeviceIdentity::new(1).expect("nonzero device identity"),
        ContextEpoch::INITIAL,
    );
    WebGl2BrowserDiscovery::open(stamp, canvas).expect("open WebGL2 provider")
}

/// One run of the shared workload over a fresh browser context.
fn drive(mode: ExecutionMode, draws: u32) -> DrawCost {
    drive_workload(open(), mode, draws, [4, 4]).unwrap_or_else(|error| {
        panic!("the workload runs over the browser context in {mode:?}: {error}")
    })
}

/// The uncached path completes a frame, over a real browser context.
///
/// This is the regression guard for the defect the native surface found first:
/// a raster draw resolved the vertex array a *pipeline install* named rather than
/// the one the driver holds, and under the uncached execution mode the geometry
/// domain replaces that array on every request, so the draw validated a fact
/// about the past and refused a live array.  The browser provider carried the
/// same shape and was fixed with the same field; this test is what says so on a
/// real driver rather than over the mock that could not see it.
///
/// The draw counts are the ones the native mock differential uses, including the
/// single-draw case, because that is the case the defect refused outright.
#[wasm_bindgen_test]
fn the_uncached_path_completes_a_frame_with_more_than_one_draw() {
    for draws in [1, 2, 8] {
        let cost = drive(ExecutionMode::Oracle, draws);
        assert_eq!(cost.passes, 1, "the workload is one pass");
        assert_eq!(cost.pass_loads, 1, "with one attachment loaded");
        assert_eq!(cost.pass_stores, 1, "and stored at the end");
        assert_eq!(cost.draws_requested, draws);
    }
}

/// The two modes are asked the same question and answer differently.
///
/// A differential whose halves did not differ would be evidence of nothing, so
/// the assertion is two-sided: the uncached run must skip no work at all, and the
/// cached run must skip some.  Which domains carry the difference is left open --
/// that is the layer's business, not this test's -- but the domains that do carry
/// it are required to be ones the frame actually named, so a skip recorded
/// against a domain the workload never touched would fail rather than pass.
#[wasm_bindgen_test]
fn the_two_modes_are_asked_the_same_thing_and_answer_differently() {
    let optimized = drive(ExecutionMode::Optimized, 8);
    let oracle = drive(ExecutionMode::Oracle, 8);

    assert_eq!(optimized.passes, oracle.passes);
    assert_eq!(optimized.draws_requested, oracle.draws_requested);
    assert_eq!(
        optimized.domains.len(),
        oracle.domains.len(),
        "the report names every domain, not only the ones this frame touched"
    );

    let skipped = |cost: &DrawCost| cost.domains.iter().map(|row| row.skipped).sum::<u64>();
    assert_eq!(
        skipped(&oracle),
        0,
        "the uncached path is not entitled to skip anything"
    );
    assert!(
        skipped(&optimized) > 0,
        "the cached path skipped nothing, which would make this differential vacuous"
    );
    for row in &optimized.domains {
        if row.skipped > 0 {
            assert!(
                row.requests > 0,
                "domain `{}` reports skips without having been asked for anything",
                row.domain
            );
        }
    }
}
