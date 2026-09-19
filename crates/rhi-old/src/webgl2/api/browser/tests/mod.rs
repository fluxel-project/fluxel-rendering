//! Browser Layer 1 evidence, split by the contract each slice proves.
//!
//! Everything here is `#[wasm_bindgen_test]` because this whole module tree is
//! `#![cfg(target_arch = "wasm32")]` and the wasm runner silently skips a plain
//! `#[test]`: an ignored assertion is worse than no assertion, because it reads
//! as proof. These tests therefore need a real WebGL2 context and really run
//! against one (headless Chrome under the wasm runner).
//!
//! The provider under test is `WebGl2BrowserDiscovery`, which nothing outside
//! this module tree consumes yet, so these tests are its first consumer.
//!
//! What cannot be provoked on a live driver — a disrupted timer interval, a
//! driver that refuses a reported extension object, a context whose probes all
//! fail — is covered by the pure decisions the driver readings feed
//! (`classify_measurement`, `admit`, `record_markers`) rather than by a
//! simulation of the driver itself.
//!
//! `version` gates the family string, `renderbuffer` the renderbuffer storage
//! facts, `timer` the counter-width oracle, and `identity` the driver markers.
//! Only the live provider, the error classifiers and the imports every slice
//! shares live here; each slice keeps its own fixture.

use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_test::*;

use super::super::{
    ContextEpoch, ContextStamp, DeviceIdentity, ExtensionProvenance, GlCapability, GlContextFlags,
    GlElapsedQueryApi as _, GlError, GlFamilyApi as _, GlFormat, GlFormatCapabilities,
    GlFormatEvidence, GlFormatResourceKind, GlFormatTable, GlKnownExtension,
    GlQueryObjectsApi as _, GlQueryResult, GlRenderBufferDesc, GlResourceApi as _,
    GlTimestampQueryApi as _,
};
use super::discovery::WebGl2BrowserDiscovery;
use super::renderbuffer_facts::{self, RenderbufferRejection};
use super::{discovery, driver_identity, exec_timer, format_map};

mod identity;
mod renderbuffer;
mod timer;
mod version;

wasm_bindgen_test_configure!(run_in_browser);

/// Opens a live browser provider on a fresh canvas.
///
/// The stamp is fabricated because discovery evidence is bound to whatever
/// stamp its owner supplies; a fresh canvas gives each test its own context, so
/// one test's probes and retained extension objects cannot reach another's.
fn provider() -> WebGl2BrowserDiscovery {
    let document = web_sys::window()
        .expect("browser window")
        .document()
        .expect("browser document");
    let canvas = document
        .create_element("canvas")
        .expect("create canvas")
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .expect("canvas element");
    canvas.set_width(1);
    canvas.set_height(1);
    let stamp = ContextStamp::new(
        DeviceIdentity::new(1).expect("nonzero device identity"),
        ContextEpoch::INITIAL,
    );
    WebGl2BrowserDiscovery::open(stamp, canvas).expect("open WebGL2 provider")
}

fn is_unsupported(error: &GlError, operation: &str) -> bool {
    matches!(error, GlError::Unsupported { operation: op, .. } if *op == operation)
}

fn is_validation(error: &GlError, operation: &str) -> bool {
    matches!(error, GlError::Validation { operation: op, .. } if *op == operation)
}
