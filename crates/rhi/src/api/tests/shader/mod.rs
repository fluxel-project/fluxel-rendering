//! Shader artifact contract tests (specification section 19).
//!
//! Section 19's rules are the ones a producer most often gets wrong, because
//! every one of them is a *canonicality* rule: a duplicate `(group, slot)`, a
//! non-ascending location list, a workgroup whose total is not the product of its
//! dimensions, an unsorted compiler-option list. Section 19.6 is explicit that the
//! RHI must not repair any of them silently, so each test below drives the
//! refusal path as well as the accepting one and asserts the exact
//! [`RhiErrorKind`].
//!
//! Nothing here needs a device: [`validate_shader_artifact`] takes the one
//! capability answer it depends on as a parameter, exactly so that these rules are
//! decidable — and testable — before a backend exists. Every test that must be
//! accepted is fed a [`BindingSupport::Supported`] answer, so only the rule named
//! in the test can be what refuses.
//!
//! # Files
//!
//! One part of the chapter per file, mirroring the `api/shader` split:
//!
//! ```text
//! mod.rs          the fixtures every section drives, and nothing else
//! vocabulary.rs   section 19.1, the stage set
//! requirements.rs sections 19.8, the canonical requirement collections
//! validation.rs   sections 19.6-19.7, what an acceptable artifact is
//! artifact.rs     sections 19.9-19.10, the artifact as a value and the handle
//! ```
//!
//! Keeping the fixtures in `mod.rs` is what lets each section file say only what
//! the section says: `use super::*` brings them in, exactly as `tests/pipeline`
//! does.

use std::sync::Arc;

use crate::api::binding::{
    BindGroupIndex, BindingCount, BindingKind, BindingSlotId, BindingSupport, BindingSupportQuery,
    BufferBindingAccess,
};
use crate::api::error::{RhiErrorKind, RhiResult};
use crate::api::identity::{DeviceGeneration, DeviceIdentity, DeviceInstanceId, ObjectId};
use crate::api::platform::requirements::{LimitKey, LimitRequirement, OptionalFeature};
use crate::api::shader::validation::validate_shader_artifact;
use crate::api::shader::vocabulary::stage_mask;
use crate::api::shader::{
    ArtifactHash, ArtifactProducerId, ArtifactProducerVersion, ComputeWorkgroupRequirements,
    ExecutableReplayAcceptanceScope, InterpolationMode, InterpolationSampling,
    PortableShaderLanguage, ShaderAbiVersion, ShaderArtifact, ShaderCode, ShaderInterface,
    ShaderInterpolation, ShaderLocation, ShaderLocationInterface, ShaderModule, ShaderNumericType,
    ShaderProvenance, ShaderResourceRequirement, ShaderStage, ShaderStages,
};

// ---------------------------------------------------------------------------
// Fixtures.
// ---------------------------------------------------------------------------

fn identity(instance: u64, generation: u64) -> DeviceIdentity {
    DeviceIdentity::new(
        DeviceInstanceId::new(instance),
        DeviceGeneration::new(generation),
    )
}

fn device() -> DeviceIdentity {
    identity(1, 1)
}

fn object(value: u64) -> ObjectId {
    ObjectId::new(value)
}

fn assert_kind(result: RhiResult<()>, expected: RhiErrorKind) {
    match result {
        Ok(()) => panic!("expected {expected}, but the artifact was accepted"),
        Err(error) => assert_eq!(error.kind(), expected, "{}", error.message()),
    }
}

/// The device answer that refuses nothing, so that only the rule under test can
/// be the reason a call fails.
fn permissive(_: &BindingSupportQuery) -> BindingSupport {
    BindingSupport::Supported
}

/// The device answer that refuses every binding.
fn refuses_bindings(_: &BindingSupportQuery) -> BindingSupport {
    BindingSupport::Unsupported
}

fn interpolation(mode: InterpolationMode) -> ShaderInterpolation {
    ShaderInterpolation {
        mode,
        sampling: InterpolationSampling::Center,
    }
}

fn location(
    value: u32,
    numeric_type: ShaderNumericType,
    components: u8,
) -> ShaderLocationInterface {
    ShaderLocationInterface {
        location: ShaderLocation::new(value),
        numeric_type,
        components,
        interpolation: None,
    }
}

fn float32(value: u32, components: u8) -> ShaderLocationInterface {
    location(value, ShaderNumericType::Float32, components)
}

fn resource(group: u32, slot: u32) -> ShaderResourceRequirement {
    ShaderResourceRequirement {
        group: BindGroupIndex::new(group),
        slot: BindingSlotId::new(slot),
        kind: BindingKind::UniformBuffer { min_size: 64 },
        count: BindingCount::One,
    }
}

/// The interface a legal vertex entry point has: the position built-in, nothing
/// else.
fn vertex_interface() -> ShaderInterface {
    ShaderInterface::new().with_writes_position(true)
}

fn artifact(stage: ShaderStage, interface: ShaderInterface) -> ShaderArtifact {
    artifact_with(stage, interface, ShaderRequirementsBuilder::plain(stage))
}

/// A small stand-in for [`crate::api::shader::ShaderRequirements`]'s builder, so
/// that the common case of "this stage's requirements and nothing else" is one
/// call rather than a conditional at every call site.
struct ShaderRequirementsBuilder;

impl ShaderRequirementsBuilder {
    fn plain(stage: ShaderStage) -> crate::api::shader::ShaderRequirements {
        match stage {
            ShaderStage::Compute => crate::api::shader::ShaderRequirements::new()
                .with_compute_workgroup(ComputeWorkgroupRequirements::new(8, 8, 1, 64, 0)),
            ShaderStage::Vertex | ShaderStage::Fragment => {
                crate::api::shader::ShaderRequirements::new()
            }
        }
    }
}

fn artifact_with(
    stage: ShaderStage,
    interface: ShaderInterface,
    requirements: crate::api::shader::ShaderRequirements,
) -> ShaderArtifact {
    ShaderArtifact::new(
        stage,
        "main",
        ShaderCode::Wgsl(Arc::from("@vertex fn main() {}")),
        ShaderAbiVersion { major: 1, minor: 0 },
        interface,
        requirements,
        ArtifactHash([7; 32]),
        ArtifactProducerId("fluxel-shaderc".to_string()),
        ArtifactProducerVersion {
            major: 0,
            minor: 16,
        },
    )
}

mod artifact;
mod requirements;
mod validation;
mod vocabulary;
