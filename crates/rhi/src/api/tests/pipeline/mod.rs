//! Pipeline contract tests (specification sections 23 through 28).
//!
//! The chapter is one connected argument — an interface constrains a pipeline, a
//! pipeline constrains a shader, and both are constrained by the device — so the
//! tests are grouped by the section that states the rule rather than by the type
//! the rule is about. Each test names the rule it drives, and every one asserts
//! the exact [`RhiErrorKind`]: the three kinds that appear here mean different
//! things to a caller, and a test that accepted any error would not notice the
//! difference between "you described this wrongly" ([`RhiErrorKind::InvalidUsage`]),
//! "these two things disagree" ([`RhiErrorKind::IncompatibleInterface`]), and
//! "this device cannot do it" ([`RhiErrorKind::Unsupported`]).
//!
//! `Facts` below is the device this file tests against: a stand-in whose every
//! answer is permissive, with one override per test that needs to make the device
//! the reason for a refusal. That is exactly why
//! [`crate::api::pipeline::PipelineDeviceFacts`] takes its answers as closures —
//! the rules are decidable without hardware, so they are tested without it.

use crate::api::binding::{
    BindGroupIndex, BindGroupLayout, BindGroupLayoutCompatibilityId, BindGroupLayoutDescriptor,
    BindingCount, BindingKind, BindingLimitClass, BindingSlot, BindingSlotId, BindingSupport,
    BindingSupportQuery, BufferBindingAccess, LayoutFingerprint, StorageAccess,
};
use crate::api::command::IndexFormat;
use crate::api::error::{RhiErrorKind, RhiResult};
use crate::api::format::{
    TextureFormat, TextureSupport, TextureSupportLimits, TextureSupportQuery,
};
use crate::api::identity::{DeviceIdentity, DeviceInstanceId, ObjectId};
use crate::api::pipeline::compute::validate_compute_pipeline_descriptor;
use crate::api::pipeline::interface::validate_pipeline_interface_descriptor;
use crate::api::pipeline::mesh::validate_mesh_pipeline_descriptor;
use crate::api::pipeline::raster::validate_raster_pipeline_descriptor;
use crate::api::pipeline::ray_tracing::validate_ray_tracing_pipeline_descriptor;
use crate::api::pipeline::resources::{
    merge_shader_resources, validate_shader_resource_requirements,
};
use crate::api::pipeline::vertex_input::{
    validate_vertex_input_against_interface, validate_vertex_input_state,
};
use crate::api::pipeline::{
    BlendComponent, BlendFactor, BlendOperation, BlendState, ColorTargetFacts, ColorTargetState,
    ColorWriteMask, ComputePipeline, ComputePipelineDescriptor, DepthBiasState, DepthState,
    DepthStencilState, MultisampleState, PipelineDeviceFacts, PipelineInterface,
    PipelineInterfaceCompatibilityId, PipelineInterfaceDescriptor, PrimitiveState,
    PrimitiveTopology, RasterPipeline, RasterPipelineDescriptor, StencilFaceState,
    StencilOperation, StencilState, VertexAttribute, VertexBufferLayout, VertexFormat,
    VertexInputState, VertexStepMode,
};
use crate::api::platform::requirements::{LimitKey, OptionalFeature};
use crate::api::resource::sampler::CompareFunction;
use crate::api::resource::texture::{Extent3d, TextureDimension, TextureUsage};
use crate::api::resource::view::TextureViewDimension;
use crate::api::shader::{
    ArtifactAcceptance, ArtifactHash, ArtifactProducerVersion, InterpolationMode,
    InterpolationSampling, ShaderAbiVersion, ShaderArtifact, ShaderCode, ShaderInterface,
    ShaderInterpolation, ShaderLocation, ShaderLocationInterface, ShaderModule, ShaderNumericType,
    ShaderRequirements, ShaderResourceRequirement, ShaderStage, ShaderStages,
};

// ---------------------------------------------------------------------------
// Fixtures.
// ---------------------------------------------------------------------------

fn identity(instance: u64) -> DeviceIdentity {
    DeviceIdentity::new(DeviceInstanceId::new(instance))
}

fn device() -> DeviceIdentity {
    identity(1)
}

fn other_device() -> DeviceIdentity {
    identity(2)
}

fn object(value: u64) -> ObjectId {
    ObjectId::new(value)
}

fn assert_kind(result: RhiResult<()>, expected: RhiErrorKind) {
    match result {
        Ok(()) => panic!("expected {expected}, but the operation was accepted"),
        Err(error) => assert_eq!(error.kind(), expected, "{}", error.message()),
    }
}

/// The device every test starts from: every limit absent (which means "not
/// exposed", not zero), every binding expressible, every optional feature
/// enabled, every artifact accepted, every format a blendable color attachment
/// with an alpha channel that outputs `Float32`, every texture creatable.
///
/// Each field is a boxed closure so that an override can wrap the answer it
/// replaces, which is what keeps a test from having to restate the other six.
struct Facts {
    limit: Box<dyn Fn(LimitKey) -> Option<u64>>,
    binding_support: Box<dyn Fn(&BindingSupportQuery) -> BindingSupport>,
    binding_limit: Box<dyn Fn(ShaderStage, BindingLimitClass) -> Option<u32>>,
    feature_supported: Box<dyn Fn(OptionalFeature) -> bool>,
    shader_acceptance: Box<dyn Fn(&ShaderArtifact) -> ArtifactAcceptance>,
    color_target_facts: Box<dyn Fn(TextureFormat) -> Option<ColorTargetFacts>>,
    texture_support: Box<dyn Fn(&TextureSupportQuery) -> TextureSupport>,
}

impl Facts {
    fn new() -> Self {
        Self {
            limit: Box::new(|_| None),
            binding_support: Box::new(|_| BindingSupport::Supported),
            binding_limit: Box::new(|_, _| None),
            feature_supported: Box::new(|_| true),
            shader_acceptance: Box::new(|_| ArtifactAcceptance::Accepted),
            color_target_facts: Box::new(|_| {
                Some(ColorTargetFacts::new(
                    true,
                    true,
                    true,
                    Some(ShaderNumericType::Float32),
                ))
            }),
            texture_support: Box::new(|_| {
                TextureSupport::Supported(TextureSupportLimits::new(
                    Extent3d::d3(16384, 16384, 2048),
                    15,
                    2048,
                ))
            }),
        }
    }

    fn device(&self) -> PipelineDeviceFacts<'_> {
        PipelineDeviceFacts {
            limit: &*self.limit,
            binding_support: &*self.binding_support,
            binding_limit: &*self.binding_limit,
            feature_supported: &*self.feature_supported,
            shader_acceptance: &*self.shader_acceptance,
            color_target_facts: &*self.color_target_facts,
            texture_support: &*self.texture_support,
        }
    }

    fn limit(mut self, key: LimitKey, value: u64) -> Self {
        let previous = self.limit;
        self.limit = Box::new(move |queried| {
            if queried == key {
                Some(value)
            } else {
                previous(queried)
            }
        });
        self
    }

    fn binding_limit(mut self, stage: ShaderStage, class: BindingLimitClass, value: u32) -> Self {
        let previous = self.binding_limit;
        self.binding_limit = Box::new(move |queried_stage, queried_class| {
            if queried_stage == stage && queried_class == class {
                Some(value)
            } else {
                previous(queried_stage, queried_class)
            }
        });
        self
    }

    fn refuses_binding(mut self, query: BindingSupportQuery) -> Self {
        let previous = self.binding_support;
        self.binding_support = Box::new(move |queried| {
            if queried == &query {
                BindingSupport::Unsupported
            } else {
                previous(queried)
            }
        });
        self
    }

    fn without_feature(mut self, feature: OptionalFeature) -> Self {
        let previous = self.feature_supported;
        self.feature_supported = Box::new(move |queried| {
            if queried == feature {
                false
            } else {
                previous(queried)
            }
        });
        self
    }

    fn refuses_artifacts(mut self) -> Self {
        self.shader_acceptance = Box::new(|_| ArtifactAcceptance::UnsupportedCodeFormat);
        self
    }

    fn not_a_color_attachment(mut self, format: TextureFormat) -> Self {
        let previous = self.color_target_facts;
        self.color_target_facts = Box::new(move |queried| {
            previous(queried).map(|mut facts| {
                if queried == format {
                    facts.color_attachment = false;
                }
                facts
            })
        });
        self
    }

    fn not_blendable(mut self, format: TextureFormat) -> Self {
        let previous = self.color_target_facts;
        self.color_target_facts = Box::new(move |queried| {
            previous(queried).map(|mut facts| {
                if queried == format {
                    facts.blendable = false;
                }
                facts
            })
        });
        self
    }

    fn without_alpha(mut self, format: TextureFormat) -> Self {
        let previous = self.color_target_facts;
        self.color_target_facts = Box::new(move |queried| {
            previous(queried).map(|mut facts| {
                if queried == format {
                    facts.has_alpha_channel = false;
                }
                facts
            })
        });
        self
    }

    fn refuses_texture(mut self, query: TextureSupportQuery) -> Self {
        let previous = self.texture_support;
        self.texture_support = Box::new(move |queried| {
            if queried == &query {
                TextureSupport::Unsupported
            } else {
                previous(queried)
            }
        });
        self
    }
}

/// The default device, used by the tests whose subject is not the device.
fn permissive() -> Facts {
    Facts::new()
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

fn artifact(
    stage: ShaderStage,
    interface: ShaderInterface,
    requirements: ShaderRequirements,
) -> ShaderArtifact {
    ShaderArtifact::new(
        stage,
        "main",
        ShaderCode::Wgsl(std::sync::Arc::from("@vertex fn main() {}")),
        ShaderAbiVersion { major: 1, minor: 0 },
        interface,
        requirements,
        ArtifactHash([3; 32]),
        ArtifactProducerVersion {
            major: 0,
            minor: 16,
        },
    )
}

fn module_on(
    device: DeviceIdentity,
    id: u64,
    stage: ShaderStage,
    interface: ShaderInterface,
    requirements: ShaderRequirements,
) -> ShaderModule {
    let artifact = artifact(stage, interface, requirements);
    ShaderModule::new(
        object(id),
        device,
        artifact.clone(),
        crate::api::tests::mock::module_backend_for_test(&artifact),
    )
}

/// A vertex entry point that writes the position built-in and the outputs given.
fn vertex_module(id: u64, outputs: Vec<ShaderLocationInterface>) -> ShaderModule {
    let mut interface = ShaderInterface::new().with_writes_position(true);
    for output in outputs {
        interface = interface.with_output(output);
    }
    module_on(
        device(),
        id,
        ShaderStage::Vertex,
        interface,
        ShaderRequirements::new(),
    )
}

/// A fragment entry point with the inputs and outputs given.
fn fragment_module(
    id: u64,
    inputs: Vec<ShaderLocationInterface>,
    outputs: Vec<ShaderLocationInterface>,
) -> ShaderModule {
    let mut interface = ShaderInterface::new();
    for input in inputs {
        interface = interface.with_input(input);
    }
    for output in outputs {
        interface = interface.with_output(output);
    }
    module_on(
        device(),
        id,
        ShaderStage::Fragment,
        interface,
        ShaderRequirements::new(),
    )
}

/// A fragment entry point that also writes the fragment depth built-in.
fn depth_writing_fragment(id: u64) -> ShaderModule {
    module_on(
        device(),
        id,
        ShaderStage::Fragment,
        ShaderInterface::new().with_writes_frag_depth(true),
        ShaderRequirements::new(),
    )
}

fn compute_module(id: u64) -> ShaderModule {
    module_on(
        device(),
        id,
        ShaderStage::Compute,
        ShaderInterface::new()
            .with_compute_workgroup_size(crate::api::shader::ComputeWorkgroupSize::new(1, 1, 1)),
        ShaderRequirements::new(),
    )
}

/// The requirements one stage declares, as a shader interface carries them.
fn requirements(resources: Vec<ShaderResourceRequirement>) -> ShaderInterface {
    let mut interface = ShaderInterface::new();
    for resource in resources {
        interface = interface.with_resource(resource);
    }
    interface
}

fn uniform(group: u32, slot: u32, min_size: u64) -> ShaderResourceRequirement {
    ShaderResourceRequirement {
        group: BindGroupIndex::new(group),
        slot: BindingSlotId::new(slot),
        kind: BindingKind::UniformBuffer { min_size },
        count: BindingCount::One,
    }
}

fn storage_buffer(
    group: u32,
    slot: u32,
    access: BufferBindingAccess,
    min_size: u64,
) -> ShaderResourceRequirement {
    ShaderResourceRequirement {
        group: BindGroupIndex::new(group),
        slot: BindingSlotId::new(slot),
        kind: BindingKind::StorageBuffer { access, min_size },
        count: BindingCount::One,
    }
}

fn storage_texture(group: u32, slot: u32, access: StorageAccess) -> ShaderResourceRequirement {
    ShaderResourceRequirement {
        group: BindGroupIndex::new(group),
        slot: BindingSlotId::new(slot),
        kind: BindingKind::StorageTexture {
            dimension: TextureViewDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            access,
        },
        count: BindingCount::One,
    }
}

fn layout_slot(value: u32, visibility: ShaderStages, kind: BindingKind) -> BindingSlot {
    BindingSlot::new(BindingSlotId::new(value), visibility, kind)
}

fn layout(entries: Vec<BindingSlot>) -> BindGroupLayout {
    BindGroupLayout::new(
        object(30),
        device(),
        BindGroupLayoutDescriptor::new(entries).canonicalized(),
        BindGroupLayoutCompatibilityId::new(1),
        LayoutFingerprint([1; 32]),
    )
}

fn interface_of(groups: Vec<BindGroupLayout>) -> PipelineInterface {
    PipelineInterface::new(
        object(40),
        device(),
        PipelineInterfaceDescriptor::new(groups),
        PipelineInterfaceCompatibilityId::new(1),
        LayoutFingerprint([2; 32]),
    )
}

fn no_bindings() -> PipelineInterface {
    interface_of(Vec::new())
}

/// A permissive 2D color-target format for the tests whose subject is not the
/// format.
const TARGET: TextureFormat = TextureFormat::Rgba8Unorm;

fn raster_with(vertex: ShaderModule) -> RasterPipelineDescriptor {
    RasterPipelineDescriptor::new(vertex, no_bindings())
}

fn check_raster(desc: &RasterPipelineDescriptor, facts: &Facts) -> RhiResult<()> {
    validate_raster_pipeline_descriptor(desc, facts.device())
}

fn check_compute(desc: &ComputePipelineDescriptor, facts: &Facts) -> RhiResult<()> {
    validate_compute_pipeline_descriptor(desc, facts.device())
}

// ---------------------------------------------------------------------------
// Files.
// ---------------------------------------------------------------------------

mod compute;
mod interface;
mod mesh_ray_tracing;
mod raster;
mod raster_state;
mod resources;
mod vertex_input;
