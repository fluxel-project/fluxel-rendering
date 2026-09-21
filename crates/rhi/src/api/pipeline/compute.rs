//! Section 28: `ComputePipeline` and its validator.
//!
//! The capability gate and the interface and merged-requirement checks shared with
//! the raster path. A compute pipeline has no target
//! signature and no fixed state, and that absence is the whole of the difference
//! between the two descriptors.
//!
//! Not owned here: the raster-only rules (section 27, `raster.rs`) and the
//! shader artifact validation, which remains owned by the shader chapter.

use core::fmt;
use std::sync::Arc;

use crate::api::binding::{BindingLimitClass, BindingSupportQuery};
use crate::api::error::{RhiError, RhiErrorKind, RhiResult};
use crate::api::format::{TextureFormat, TextureSupportQuery};
use crate::api::identity::{DeviceIdentity, Label, ObjectId};
use crate::api::pipeline::PipelineCache;
use crate::api::platform::Device;
use crate::api::platform::requirements::{LimitKey, OptionalFeature};
use crate::api::shader::{ArtifactAcceptance, ShaderArtifact, ShaderModule, ShaderStage};

use crate::api::pipeline::backend::ComputePipelineBackend;

use super::interface::{PipelineInterface, validate_pipeline_interface_descriptor};
use super::resources::{merge_shader_resources, validate_shader_resource_requirements};
use super::{ColorTargetFacts, PipelineDeviceFacts};

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
    /// Optional native cache consulted while building this pipeline.
    pub cache: Option<PipelineCache>,
}

impl ComputePipelineDescriptor {
    /// Describes a compute pipeline.
    pub fn new(shader: ShaderModule, interface: PipelineInterface) -> Self {
        Self {
            label: Label::default(),
            shader,
            interface,
            cache: None,
        }
    }

    /// Attaches a diagnostic label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Label(Some(label.into()));
        self
    }

    /// Associates a same-device native pipeline cache with this creation.
    pub fn with_cache(mut self, cache: PipelineCache) -> Self {
        self.cache = Some(cache);
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
    inner: Arc<ComputePipelineInner>,
}

/// The one ownership domain of a created compute pipeline.
struct ComputePipelineInner {
    id: ObjectId,
    device: DeviceIdentity,
    descriptor: ComputePipelineDescriptor,
    /// The driver's pipeline state object, in the same shape
    /// [`crate::api::resource::Buffer`] holds its allocation: it is directly
    /// owned by this handle's single shared inner domain, and behind `dyn` so
    /// that no native type reaches the exported surface (section 59).
    ///
    /// Nothing portable reads it. A dispatch is lowered by the *device's* backend,
    /// which downcasts this and the bound groups in one place, which is why
    /// `crate::api::pipeline::backend` carries no dispatch verb.
    #[cfg_attr(not(feature = "dx12"), allow(dead_code))]
    native: Box<dyn ComputePipelineBackend>,
}

impl ComputePipeline {
    /// Assembles a created compute pipeline.
    ///
    /// Crate-private: section 3 gives identity to the object that created it, so
    /// only `Device::create_compute_pipeline` may produce one.
    pub(crate) fn new(
        id: ObjectId,
        device: DeviceIdentity,
        descriptor: ComputePipelineDescriptor,
        native: Box<dyn ComputePipelineBackend>,
    ) -> Self {
        Self {
            inner: Arc::new(ComputePipelineInner {
                id,
                device,
                descriptor,
                native,
            }),
        }
    }

    /// The driver's pipeline state object.
    ///
    /// Crate-private because section 59 keeps native lowering out of the exported
    /// surface: the returned trait is `pub(crate)`, so this is the seam's ordinary
    /// inside face. Its caller is the command lowering of the backend that built
    /// the pipeline, which downcasts to its own type — a pipeline handed to
    /// another backend's device is refused portably, by device identity, long
    /// before a downcast is attempted.
    #[cfg_attr(not(feature = "dx12"), allow(dead_code))]
    pub(crate) fn native(&self) -> &dyn ComputePipelineBackend {
        self.inner.native.as_ref()
    }

    /// This pipeline's process-local object ID.
    pub fn id(&self) -> ObjectId {
        self.inner.id
    }

    /// The device that created this pipeline.
    pub fn device_identity(&self) -> DeviceIdentity {
        self.inner.device
    }

    /// The descriptor this pipeline was created from.
    pub fn descriptor(&self) -> &ComputePipelineDescriptor {
        &self.inner.descriptor
    }

    /// The pipeline interface this pipeline was created with.
    pub fn interface(&self) -> &PipelineInterface {
        &self.inner.descriptor.interface
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
            .field("id", &self.inner.id)
            .field("device", &self.inner.device)
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
/// ```
///
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

    Ok(())
}

/// Section 28's creation verb, defined in the chapter that owns the type it
/// produces.
///
/// The placement is the specification's own: section 28 writes this verb in an
/// `impl Device` in its own chapter, so the definition site is the owner.
impl Device {
    /// Creates a compute pipeline on this device from a descriptor.
    ///
    /// Section 3.1's O(1) identity step comes first, over the two device-owned
    /// objects the descriptor names: the shader module and the interface. Section
    /// 28's own rule compares the module against the *interface's* device, so
    /// proving the interface is this device's is the façade's half of it; without
    /// it, a descriptor whose two parts agree with each other but belong to
    /// another device would validate and say nothing about this one.
    ///
    /// The compute gate and merged-requirement checks then run through
    /// `validate_compute_pipeline_descriptor` against the
    /// seven device answers the descriptor-bag carries.
    ///
    /// The backend is asked last, and its failure is *not* wrapped or reworded:
    /// this is the first verb in the crate whose native call can fail for a reason
    /// about the program, and a driver refusing to build state for bytes that
    /// passed every portable check is a fact about the driver. Folding it into
    /// `InvalidUsage` would tell the caller its descriptor was wrong when the
    /// portable layer has already said otherwise (discipline 4 in
    /// the crate-private backend contract).
    ///
    /// The descriptor the backend receives is the caller's. Unlike a bind group
    /// there is no canonical form to hand it — an interface's order is already
    /// semantic and its groups are already canonical — so the packet validated is
    /// the packet stored and the packet lowered.
    pub async fn create_compute_pipeline(
        &self,
        desc: &ComputePipelineDescriptor,
    ) -> RhiResult<ComputePipeline> {
        let identity = self.identity();
        if desc.interface.device_identity() != identity {
            return Err(RhiError::new(
                RhiErrorKind::WrongDevice,
                "the pipeline interface belongs to a different device",
            )
            .with_object(desc.interface.id()));
        }
        if desc.shader.device_identity() != identity {
            return Err(RhiError::new(
                RhiErrorKind::WrongDevice,
                "the compute shader belongs to a different device",
            )
            .with_object(desc.shader.id()));
        }
        if let Some(cache) = desc.cache.as_ref()
            && cache.device_identity() != identity
        {
            return Err(RhiError::new(
                RhiErrorKind::WrongDevice,
                "the pipeline cache belongs to a different device",
            )
            .with_object(cache.id()));
        }

        // Section 6.5's liveness verdict, after every ownership comparison above
        // and before the device's seven answers are read. A descriptor naming a
        // foreign interface or shader is `WrongDevice` even on a lost device:
        // section 3.1 puts those comparisons first.
        self.require_active()?;

        let capabilities = self.capabilities();

        // Each closure is one of the seven questions `PipelineDeviceFacts` names,
        // answered by the method of the same name on `EnabledCapabilities`. They
        // are locals rather than inline struct-literal fields because the bag
        // holds `&dyn Fn` references, and a reference needs a binding to point at.
        let limit = |key: LimitKey| capabilities.limit(key);
        let binding_support = |query: &BindingSupportQuery| capabilities.binding_support(query);
        let binding_limit =
            |stage: ShaderStage, class: BindingLimitClass| capabilities.binding_limit(stage, class);
        let feature_supported = |feature: OptionalFeature| capabilities.supports_feature(feature);
        let shader_acceptance =
            |artifact: &ShaderArtifact| capabilities.shader_acceptance(artifact);
        let color_target_facts = |format: TextureFormat| {
            capabilities.format(format).map(|facts| {
                ColorTargetFacts::new(
                    facts.color_attachment(),
                    facts.blendable(),
                    facts.has_alpha_channel(),
                    facts.color_output_type(),
                )
            })
        };
        let texture_support = |query: &TextureSupportQuery| capabilities.texture_support(query);

        validate_compute_pipeline_descriptor(
            desc,
            PipelineDeviceFacts {
                limit: &limit,
                binding_support: &binding_support,
                binding_limit: &binding_limit,
                feature_supported: &feature_supported,
                shader_acceptance: &shader_acceptance,
                color_target_facts: &color_target_facts,
                texture_support: &texture_support,
            },
        )?;

        // A backend may compile asynchronously.  No portable pipeline exists
        // until the native request resolves, preserving the invariant that a
        // rejected compile cannot be observed as a usable pipeline handle.
        let mut request = self.native().create_compute_pipeline_request(desc)?;
        let native =
            std::future::poll_fn(
                |context| match request.poll_or_register_waker(context.waker()) {
                    Ok(crate::api::platform::backend::CreationRequestProgress::Pending) => {
                        std::task::Poll::Pending
                    }
                    Ok(crate::api::platform::backend::CreationRequestProgress::Ready(value)) => {
                        std::task::Poll::Ready(Ok(value))
                    }
                    Err(error) => std::task::Poll::Ready(Err(error)),
                },
            )
            .await?;
        // A native compiler completion and device loss can race.  Terminal loss
        // wins over publication of a fresh logical handle.
        self.require_active()?;

        Ok(ComputePipeline::new(
            ObjectId::next(),
            identity,
            desc.clone(),
            native,
        ))
    }
}
