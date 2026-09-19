//! Section 28: `ComputePipeline` and its validator.
//!
//! The capability gate, the interface and merged-requirement checks shared with
//! the raster path, and the workgroup limits. A compute pipeline has no target
//! signature and no fixed state, and that absence is the whole of the difference
//! between the two descriptors.
//!
//! Not owned here: the raster-only rules (section 27, `raster.rs`) and the
//! workgroup shape rules themselves, which belong to the shader artifact's own
//! validator (section 19.7) and are asked through it rather than copied.

use core::fmt;

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::identity::{DeviceIdentity, Label, ObjectId};
use crate::api::platform::requirements::{LimitKey, OptionalFeature};
use crate::api::shader::{
    ArtifactAcceptance, ComputeWorkgroupRequirements, ShaderModule, ShaderStage,
};

use crate::api::shader::validation::validate_compute_workgroup;

use super::PipelineDeviceFacts;
use super::interface::{PipelineInterface, validate_pipeline_interface_descriptor};
use super::resources::{merge_shader_resources, validate_shader_resource_requirements};

// ---------------------------------------------------------------------------
// Section 28 - ComputePipeline
// ---------------------------------------------------------------------------

/// Everything a caller states about a compute pipeline before it exists.
#[non_exhaustive]
#[derive(Clone)]
pub struct ComputePipelineDescriptor {
    /// Diagnostic label. Excluded from every canonical hash (section 19.8).
    pub label: Label,
    /// The compute entry point.
    pub shader: ShaderModule,
    /// The logical layout contract bound to this entry point.
    pub interface: PipelineInterface,
}

impl ComputePipelineDescriptor {
    /// Describes a compute pipeline.
    pub fn new(shader: ShaderModule, interface: PipelineInterface) -> Self {
        Self {
            label: Label::default(),
            shader,
            interface,
        }
    }

    /// Attaches a diagnostic label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Label(Some(label.into()));
        self
    }
}

/// A created compute pipeline.
///
/// Opaque, cloneable, and identified by [`ObjectId`] plus the [`DeviceIdentity`]
/// that created it. It exists only on a device that enabled
/// [`OptionalFeature::Compute`]; section 28 makes the whole chapter
/// capability-gated even though its API shape is frozen.
#[derive(Clone)]
pub struct ComputePipeline {
    id: ObjectId,
    device: DeviceIdentity,
    descriptor: ComputePipelineDescriptor,
}

impl ComputePipeline {
    /// Assembles a created compute pipeline.
    ///
    /// Crate-private: section 3 gives identity to the object that created it, so
    /// only `Device::create_compute_pipeline` may produce one.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Device::create_compute_pipeline calls this once api::platform is declared"
        )
    )]
    pub(crate) fn new(
        id: ObjectId,
        device: DeviceIdentity,
        descriptor: ComputePipelineDescriptor,
    ) -> Self {
        Self {
            id,
            device,
            descriptor,
        }
    }

    /// This pipeline's process-local object ID.
    pub fn id(&self) -> ObjectId {
        self.id
    }

    /// The device that created this pipeline.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.device
    }

    /// The descriptor this pipeline was created from.
    pub fn descriptor(&self) -> &ComputePipelineDescriptor {
        &self.descriptor
    }

    /// The pipeline interface this pipeline was created with.
    pub fn interface(&self) -> &PipelineInterface {
        &self.descriptor.interface
    }
}

/// Prints portable identity only.
///
/// Written by hand rather than derived (defect D6 of the 0.16 plan): section 28
/// declares `#[derive(Clone)]` and no `Debug`, and a pipeline is exactly the
/// object a caller needs to name in a log. The descriptor is one call away through
/// [`ComputePipeline::descriptor`].
impl fmt::Debug for ComputePipeline {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ComputePipeline")
            .field("id", &self.id)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

/// Checks everything about a compute pipeline that does not need a backend.
///
/// Section 28's creation list, in its own order:
///
/// ```text
/// OptionalFeature::Compute enabled
/// shader / interface DeviceIdentity are the same
/// shader.stage == Compute
/// ShaderInterface: location inputs/outputs are empty
///                  resources compatible with PipelineInterface
/// ShaderRequirements: features / limits satisfied
/// ComputeWorkgroupRequirements: present, non-zero, x*y*z == total_invocations,
///                               and within the five workgroup limits
/// ```
///
/// The workgroup shape rules — non-zero dimensions and the checked
/// `x * y * z == total_invocations` identity — are applied by the shader
/// artifact's own validator and are applied again here through the same function,
/// because section 28 states them as conditions of *pipeline creation*: an
/// artifact that reached a device by another path must fail here rather than
/// become a pipeline whose reflection disagrees with itself.
///
/// These are reflection requirements of the entry point, not dispatch dimensions
/// (section 28), which is why the device limits they are compared against are the
/// workgroup-size limits and not the workgroups-per-dimension limit.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "Device::create_compute_pipeline validates through this once api::platform is \
                  declared"
    )
)]
pub(crate) fn validate_compute_pipeline_descriptor(
    desc: &ComputePipelineDescriptor,
    facts: PipelineDeviceFacts<'_>,
) -> RhiResult<()> {
    if !(facts.feature_supported)(OptionalFeature::Compute) {
        return Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "compute pipelines require OptionalFeature::Compute, which this device did not enable",
        ));
    }

    if desc.shader.device_identity() != desc.interface.device_identity() {
        return Err(RhiError::new(
            RhiErrorKind::WrongDevice,
            "the compute shader belongs to a different device than the pipeline interface",
        )
        .with_object(desc.shader.id()));
    }

    if desc.shader.stage() != ShaderStage::Compute {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            format!(
                "a compute pipeline's entry point has stage {:?}",
                desc.shader.stage()
            ),
        ));
    }

    let artifact = desc.shader.artifact();
    let interface = &artifact.interface;
    if !interface.inputs().is_empty() || !interface.outputs().is_empty() {
        return Err(RhiError::new(
            RhiErrorKind::IncompatibleInterface,
            "a compute entry point has no stage inputs or outputs",
        ));
    }

    validate_pipeline_interface_descriptor(
        desc.interface.descriptor(),
        facts.limit,
        facts.binding_limit,
    )?;
    let merged = merge_shader_resources([(ShaderStage::Compute, interface)])?;
    validate_shader_resource_requirements(&merged, &desc.interface, facts.binding_support)?;

    // Section 28's "ShaderRequirements: features / limits satisfied". The rule
    // itself belongs to section 19.7, so it is asked of its owner rather than
    // re-derived here.
    let acceptance = (facts.shader_acceptance)(artifact);
    if acceptance != ArtifactAcceptance::Accepted {
        return Err(RhiError::new(
            RhiErrorKind::Unsupported,
            format!("this device does not accept the compute shader artifact: {acceptance:?}"),
        ));
    }

    let Some(workgroup) = artifact.requirements.compute_workgroup() else {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "a compute entry point must declare its workgroup requirements",
        ));
    };
    validate_workgroup_shape(&workgroup)?;

    let limit = facts.limit;
    for (key, value) in [
        (LimitKey::MaxComputeWorkgroupSizeX, workgroup.x as u64),
        (LimitKey::MaxComputeWorkgroupSizeY, workgroup.y as u64),
        (LimitKey::MaxComputeWorkgroupSizeZ, workgroup.z as u64),
        (
            LimitKey::MaxComputeInvocationsPerWorkgroup,
            workgroup.total_invocations as u64,
        ),
        (
            LimitKey::MaxComputeWorkgroupStorageSize,
            workgroup.workgroup_storage_bytes,
        ),
    ] {
        if let Some(max) = limit(key) {
            if value > max {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    format!(
                        "the compute entry point requires {value} for {key:?}, over the device \
                         maximum of {max}"
                    ),
                ));
            }
        }
    }

    Ok(())
}

/// The workgroup shape rules, shared with the shader artifact's own validator.
///
/// Section 28 lists them among the conditions of pipeline creation and section
/// 19.7 among the conditions of an artifact being internally consistent; both
/// call [`validate_compute_workgroup`], so the two readings cannot drift.
fn validate_workgroup_shape(workgroup: &ComputeWorkgroupRequirements) -> RhiResult<()> {
    validate_compute_workgroup(workgroup)
}
