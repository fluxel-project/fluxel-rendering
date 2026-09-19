//! Driver-identity marker evidence.
//!
//! The rule under test is "never invent an identity": a route that answers
//! nothing, or answers only half, records the unavailable marker rather than a
//! name that was never reported. The masked pair the browser chooses to publish
//! stays the context's own answer and is never replaced by the unmasked one.

use wasm_bindgen_test::*;

use super::*;

#[wasm_bindgen_test]
fn identity_markers_record_exactly_one_shape() {
    let mut flags = GlContextFlags::default();
    driver_identity::record_markers(&mut flags, None);
    assert_eq!(flags.other.len(), 1);
    assert!(flags.other.contains("webgl.unmasked-identity=unavailable"));

    let mut flags = GlContextFlags::default();
    driver_identity::record_markers(&mut flags, Some(("Vendor X".into(), "Renderer Y".into())));
    assert_eq!(flags.other.len(), 2);
    assert!(flags.other.contains("webgl.unmasked-vendor=Vendor X"));
    assert!(flags.other.contains("webgl.unmasked-renderer=Renderer Y"));
    assert!(!flags.other.contains("webgl.unmasked-identity=unavailable"));

    // A route is not half an identity and a blank string is not a name: each of
    // these must collapse to the unavailable marker rather than be recorded as
    // a driver identity that was never answered.
    assert_eq!(driver_identity::identity_from_strings(None, None), None);
    assert_eq!(
        driver_identity::identity_from_strings(Some("Vendor X".into()), None),
        None
    );
    assert_eq!(
        driver_identity::identity_from_strings(None, Some("Renderer Y".into())),
        None
    );
    assert_eq!(
        driver_identity::identity_from_strings(Some("  ".into()), Some("Renderer Y".into())),
        None
    );
    assert_eq!(
        driver_identity::identity_from_strings(Some("Vendor X".into()), Some(String::new())),
        None
    );
    assert_eq!(
        driver_identity::identity_from_strings(Some("Vendor X".into()), Some("Renderer Y".into())),
        Some(("Vendor X".to_owned(), "Renderer Y".to_owned()))
    );
}

#[wasm_bindgen_test]
fn identity_markers_match_the_live_context_and_never_replace_its_own_answers() {
    let provider = provider();
    let flags = provider.snapshot().context().flags();
    let live = driver_identity::unmasked_identity(&provider.raw);
    let expected = if live.is_some() { 2 } else { 1 };
    // Whether this context exposes the unmasked route is the browser's choice
    // and is not asserted as a fixed answer, so the branch that ran is recorded
    // instead: the unavailable branch is honest only if a reader can tell it is
    // the branch this context took rather than the only branch ever written.
    web_sys::console::log_1(
        &format!("browser driver identity: unmasked={}", live.is_some()).into(),
    );
    match &live {
        Some((vendor, renderer)) => {
            assert!(
                flags
                    .other
                    .contains(&format!("webgl.unmasked-vendor={vendor}"))
            );
            assert!(
                flags
                    .other
                    .contains(&format!("webgl.unmasked-renderer={renderer}"))
            );
            assert!(!flags.other.contains("webgl.unmasked-identity=unavailable"));
        }
        None => {
            assert!(flags.other.contains("webgl.unmasked-identity=unavailable"));
            assert!(
                !flags
                    .other
                    .iter()
                    .any(|marker| marker.starts_with("webgl.unmasked-vendor="))
            );
        }
    }
    // Exactly one identity shape is recorded, never the pair beside a
    // contradicting unavailable marker.
    let shapes = flags
        .other
        .iter()
        .filter(|marker| marker.starts_with("webgl.unmasked-"))
        .count();
    assert_eq!(shapes, expected, "identity markers: {:?}", flags.other);
    // The masked pair the browser chose to report stays the context's own
    // answer, so both facts remain visible and distinguishable.
    let vendor = provider
        .raw
        .get_parameter(web_sys::WebGl2RenderingContext::VENDOR)
        .ok()
        .and_then(|value| value.as_string())
        .expect("VENDOR");
    assert_eq!(provider.snapshot().context().vendor(), vendor);
}
