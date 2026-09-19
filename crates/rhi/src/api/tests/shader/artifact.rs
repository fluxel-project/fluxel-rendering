//! Sections 19.9-19.10: the artifact as a value, and the module handle.
//!
//! The default provenance and the created module's identity. `use super::*`
//! brings in the fixtures.

use super::*;

// ---------------------------------------------------------------------------
// Section 19.9 / 19.10: the artifact as a value, and the module handle.
// ---------------------------------------------------------------------------

#[test]
fn an_artifact_defaults_to_the_most_restrictive_provenance() {
    // Section 19.9's constructor list has no provenance parameter and supplies only
    // a builder, so the constructor must choose something. It fails closed: an
    // artifact nobody described as replayable is not replayable.
    let artifact = artifact(ShaderStage::Vertex, vertex_interface());
    match artifact.provenance {
        ShaderProvenance::ExecutableOnly { replay_acceptance } => assert_eq!(
            replay_acceptance,
            ExecutableReplayAcceptanceScope::Denied,
            "an artifact with no stated provenance must not be replayable"
        ),
        ShaderProvenance::PortableSource { .. } => {
            panic!("the default provenance must not claim portable source")
        }
    }
}

#[test]
fn a_module_reports_the_artifact_and_stage_it_was_created_from() {
    let artifact = artifact(ShaderStage::Vertex, vertex_interface());
    let module = ShaderModule::new(
        object(3),
        device(),
        artifact.clone(),
        crate::base::mock::module_backend_for_test(&artifact),
    );

    assert_eq!(module.id(), object(3));
    assert_eq!(module.device_identity(), device());
    assert_eq!(module.stage(), ShaderStage::Vertex);
    assert_eq!(module.artifact().entry_point, "main");
    assert_eq!(module.artifact().content_hash, ArtifactHash([7; 32]));
    assert!(module.artifact().interface.writes_position());
    assert!(module.artifact().requirements.compute_workgroup().is_none());
}

#[test]
fn a_module_debug_prints_portable_identity_only() {
    // Defect D6 of the 0.16 series: the specification declares `#[derive(Clone)]`
    // and no `Debug` on the handle, while descriptors that contain one do derive
    // `Debug`. The handle implements `Debug` by hand, printing identity rather than
    // contents, so that the native field the backend port will add never has to be
    // printable.
    let artifact = artifact(ShaderStage::Vertex, vertex_interface());
    let module = ShaderModule::new(
        object(4),
        device(),
        artifact.clone(),
        crate::base::mock::module_backend_for_test(&artifact),
    );
    let text = format!("{module:?}");
    assert!(text.contains("ShaderModule"), "{text}");
    assert!(text.contains("id"), "{text}");
    assert!(
        !text.contains("main"),
        "contents must not be printed: {text}"
    );
}

// ---------------------------------------------------------------------------
// Section 19.10: the creation verb, against the mock.
// ---------------------------------------------------------------------------

/// The whole portable chain of section 19.10, on the conformance backend: the
/// verdict is read from the device's recorded form, the backend is reached only
/// after it, and the handle that comes back carries the backend's own object.
///
/// The device is stated rather than defaulted because the accepted code forms are a
/// recorded device fact — the default mock records none and refuses every artifact,
/// which the next test relies on.
#[test]
fn create_shader_reads_the_device_verdict_first_and_keeps_the_backend_object() {
    use crate::api::shader::vocabulary::AcceptedCodeForm;

    let (device, native) = crate::base::mock::shaders_for_test(device(), &[AcceptedCodeForm::Wgsl]);
    let artifact = artifact(ShaderStage::Vertex, vertex_interface());

    let module = device
        .create_shader(&artifact)
        .expect("a device that records the artifact's form and states no limits accepts it");

    assert_eq!(
        native.shader_modules(),
        1,
        "the backend was reached exactly once"
    );
    assert_eq!(module.stage(), ShaderStage::Vertex);
    assert_eq!(module.device_identity(), device.identity());

    // The seam: the portable handle hands back the backend's own type, which is
    // what the pipeline lowering will downcast to. Reading it here is also what
    // keeps `ShaderModule::native` and `ShaderModuleBackend::as_any` from being
    // dead code in a test build.
    let held = module
        .native()
        .as_any()
        .downcast_ref::<crate::base::mock::MockShaderModule>()
        .expect("the mock device's module is the type its own backend put there");
    assert_eq!(held.artifact().entry_point, artifact.entry_point);
    assert_eq!(held.artifact().content_hash, artifact.content_hash);
}

/// The verdict precedes the port, and this is the test that can tell.
///
/// A refusal that had already reached the backend would return the same `Err`, so
/// the error alone cannot distinguish "the portable layer refused before lowering"
/// from "the backend was asked and refused too". `MockDevice::shader_modules` is
/// what separates them, and the distinction is the whole of discipline 1.
#[test]
fn a_device_that_records_no_code_form_refuses_before_the_backend_is_reached() {
    let (device, native) = crate::base::mock::shaders_for_test(device(), &[]);
    let artifact = artifact(ShaderStage::Vertex, vertex_interface());

    let error = device
        .create_shader(&artifact)
        .expect_err("a device that consumes no code form cannot accept any artifact");

    assert_eq!(error.kind(), RhiErrorKind::Unsupported);
    assert_eq!(error.operation(), Some("Device::create_shader"));
    assert_eq!(
        native.shader_modules(),
        0,
        "the refusal must happen before the backend is reached"
    );
}

/// A module's clone is the same logical object, not a second one.
///
/// Section 18.6's last-owner rule is why the backend object sits behind an `Arc`:
/// two clones each owning a native entry point would be two modules wearing one
/// identity, and the point at which the last one drops would have no single moment.
#[test]
fn a_cloned_module_shares_one_backend_object() {
    use crate::api::shader::vocabulary::AcceptedCodeForm;
    use std::sync::Arc;

    let (device, _) = crate::base::mock::shaders_for_test(device(), &[AcceptedCodeForm::Wgsl]);
    let module = device
        .create_shader(&artifact(ShaderStage::Vertex, vertex_interface()))
        .expect("the device records this artifact's form");

    let clone = module.clone();
    assert_eq!(clone.id(), module.id());
    assert!(Arc::ptr_eq(module.native(), clone.native()));
}
