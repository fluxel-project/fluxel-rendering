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
    let module = ShaderModule::new(object(3), device(), artifact);

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
    let module = ShaderModule::new(
        object(4),
        device(),
        artifact(ShaderStage::Vertex, vertex_interface()),
    );
    let text = format!("{module:?}");
    assert!(text.contains("ShaderModule"), "{text}");
    assert!(text.contains("id"), "{text}");
    assert!(
        !text.contains("main"),
        "contents must not be printed: {text}"
    );
}
