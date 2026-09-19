//! Captured object definitions (rhi-design section 54).
//!
//! # What it is
//!
//! The reconstructable definition graph of every object a capture can observe.
//! An object is described by the same descriptor the caller created it from,
//! plus [`ObjectId`] references to the objects it depends on, so a definition
//! graph can be walked without any live RHI handle.
//!
//! ```text
//! TextureView   -> Texture ObjectId
//! BindGroup     -> BindGroupLayout ObjectId, Buffer/TextureView/Sampler ObjectIds
//! PipelineInterface -> BindGroupLayout ObjectIds
//! RasterPipeline    -> shader ObjectIds, PipelineInterface ObjectId, fixed state
//! ```
//!
//! # What it deliberately does not own
//!
//! A definition contains value types, [`ObjectId`], and [`DeviceIdentity`] only.
//! No live handle, native resource, pointer, descriptor index, or GPU address
//! appears anywhere, and no Rust memory layout of these values is an encoding
//! (section 58.8). The artifact layer maps runtime `ObjectId`s to its own
//! capture-local typed ids; RHI defines no on-disk encoding (section 54, tail).

use super::super::binding::{BindGroupLayoutDescriptor, BindingSlotId};
use super::super::pipeline::{
    ColorTargetState, DepthStencilState, MultisampleState, PrimitiveState, VertexInputState,
};
use super::super::platform::{DeviceIdentity, Label, ObjectId};
use super::super::presentation::PresentationConfiguration;
use super::super::resource::{
    BufferDescriptor, BufferRange, SamplerDescriptor, TextureDescriptor, TextureViewDescriptor,
};
use super::super::shader::ShaderArtifact;

/// One resource a captured bind group entry binds.
///
/// It is the definition-side counterpart of `binding::BindingResource`: the
/// same shape with every live handle replaced by the `ObjectId` that names it.
#[non_exhaustive]
#[derive(Clone)]
pub enum CapturedBindingResource {
    /// A single buffer range.
    Buffer {
        /// The buffer.
        buffer: ObjectId,
        /// The visible range.
        range: BufferRange,
    },

    /// A single texture view.
    TextureView {
        /// The view.
        view: ObjectId,
    },

    /// A single sampler.
    Sampler {
        /// The sampler.
        sampler: ObjectId,
    },

    /// A fixed-length buffer range array.
    BufferArray(Vec<(ObjectId, BufferRange)>),
    /// A fixed-length texture view array.
    TextureViewArray(Vec<ObjectId>),
    /// A fixed-length sampler array.
    SamplerArray(Vec<ObjectId>),
}

/// One captured bind group entry.
#[derive(Clone)]
pub struct CapturedBindGroupEntry {
    /// The slot this entry binds.
    pub slot: BindingSlotId,
    /// What it binds.
    pub resource: CapturedBindingResource,
}

/// The definition of a captured bind group.
#[derive(Clone)]
pub struct CapturedBindGroupDefinition {
    /// A diagnostic label. It never participates in identity.
    pub label: Label,
    /// The bind group layout this group is built against.
    pub layout: ObjectId,
    /// The entries, in the order the group was created with.
    pub entries: Vec<CapturedBindGroupEntry>,
}

/// The definition of a captured pipeline interface.
#[derive(Clone)]
pub struct CapturedPipelineInterfaceDefinition {
    /// A diagnostic label.
    pub label: Label,
    /// The group layouts, indexed by bind group index.
    ///
    /// A hole is an explicit empty layout, exactly as in
    /// `PipelineInterfaceDescriptor`: the vector index is the group index.
    pub groups: Vec<ObjectId>,
}

/// The fixed state of a captured raster pipeline.
///
/// Fixed state is captured by value because it is exactly what a replaying
/// device must reproduce: the same entry points, the same interface, and the
/// same assembly, depth, multisample, and blend facts.
#[derive(Clone)]
pub struct CapturedRasterPipelineDefinition {
    /// A diagnostic label.
    pub label: Label,

    /// The vertex entry point.
    pub vertex: ObjectId,
    /// The fragment entry point, when the pipeline has one.
    pub fragment: Option<ObjectId>,
    /// The logical pipeline interface the shaders are written against.
    pub interface: ObjectId,

    /// The vertex fetch layout.
    pub vertex_input: VertexInputState,
    /// Primitive assembly state.
    pub primitive: PrimitiveState,
    /// Depth/stencil state, when the pipeline has one.
    pub depth_stencil: Option<DepthStencilState>,
    /// Multisample state.
    pub multisample: MultisampleState,
    /// Vector index is the color output location, as in the live descriptor.
    pub color_targets: Vec<Option<ColorTargetState>>,
}

/// The definition of a captured compute pipeline.
#[derive(Clone)]
pub struct CapturedComputePipelineDefinition {
    /// A diagnostic label.
    pub label: Label,
    /// The compute entry point.
    pub shader: ObjectId,
    /// The logical pipeline interface the entry point is written against.
    pub interface: ObjectId,
}

/// A capture-local key for a host object supplied by ReplayRuntime or its
/// Artifact-layer fixture provider. It is not an OS or native graphics
/// handle.
///
/// The target itself is host-owned and cannot be reconstructed from RHI state:
/// a window, canvas, or composed layer is created by the host, not by the
/// device. RHI therefore names it by an opaque capture-local key, and the
/// replaying side binds that key to a fixture it created itself.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CapturedPresentationTargetFixture {
    /// The capture-local fixture key.
    pub key: String,
}

/// The definition of a captured presentation target.
#[derive(Clone)]
pub struct CapturedPresentationTargetDefinition {
    /// The fixture this target is reconstructed from.
    pub fixture: CapturedPresentationTargetFixture,
}

/// The definition of a captured configured-presentation lease.
///
/// The lease references its target by `ObjectId` rather than embedding it, so
/// the graph has one node per real object and a target shared by two
/// configurations is described once.
#[derive(Clone)]
pub struct CapturedConfiguredPresentationDefinition {
    /// The device identity the lease belongs to.
    pub device: DeviceIdentity,
    /// The presentation target the lease configures.
    pub target: ObjectId,
    /// The configuration that was accepted.
    pub configuration: PresentationConfiguration,
}

/// The complete definition of one observable object.
///
/// Every `ObjectId` a semantic event can carry has a variant here, including
/// [`CapturedObjectDefinition::PresentationTarget`] and
/// [`CapturedObjectDefinition::ConfiguredPresentation`]: a `FrameAcquired`
/// event names both, and each must be describable without serializing a host or
/// native presentation handle.
#[non_exhaustive]
#[derive(Clone)]
pub enum CapturedObjectDefinition {
    /// The target is reconstructed by binding `fixture` to an external
    /// presentation fixture; tooling never serializes a native target.
    PresentationTarget {
        /// The target's object id.
        id: ObjectId,
        /// Its fixture-keyed definition.
        definition: CapturedPresentationTargetDefinition,
    },

    /// A configured-presentation lease.
    ConfiguredPresentation {
        /// The lease's object id.
        id: ObjectId,
        /// Its definition.
        definition: CapturedConfiguredPresentationDefinition,
    },

    /// A buffer.
    Buffer {
        /// The buffer's object id.
        id: ObjectId,
        /// The descriptor it was created from.
        descriptor: BufferDescriptor,
    },

    /// A texture.
    Texture {
        /// The texture's object id.
        id: ObjectId,
        /// The descriptor it was created from.
        descriptor: TextureDescriptor,
    },

    /// A texture view.
    TextureView {
        /// The view's object id.
        id: ObjectId,
        /// The texture this view reads.
        texture: ObjectId,
        /// The descriptor it was created from.
        descriptor: TextureViewDescriptor,
    },

    /// A sampler.
    Sampler {
        /// The sampler's object id.
        id: ObjectId,
        /// The descriptor it was created from.
        descriptor: SamplerDescriptor,
    },

    /// A shader entry point, with its full provenance.
    Shader {
        /// The entry point's object id.
        id: ObjectId,
        /// The artifact it was created from.
        artifact: ShaderArtifact,
    },

    /// A bind group layout.
    BindGroupLayout {
        /// The layout's object id.
        id: ObjectId,
        /// The descriptor it was created from.
        descriptor: BindGroupLayoutDescriptor,
    },

    /// A bind group.
    BindGroup {
        /// The bind group's object id.
        id: ObjectId,
        /// Its definition.
        definition: CapturedBindGroupDefinition,
    },

    /// A pipeline interface.
    PipelineInterface {
        /// The interface's object id.
        id: ObjectId,
        /// Its definition.
        definition: CapturedPipelineInterfaceDefinition,
    },

    /// A raster pipeline.
    RasterPipeline {
        /// The pipeline's object id.
        id: ObjectId,
        /// Its definition.
        definition: CapturedRasterPipelineDefinition,
    },

    /// A compute pipeline.
    ComputePipeline {
        /// The pipeline's object id.
        id: ObjectId,
        /// Its definition.
        definition: CapturedComputePipelineDefinition,
    },
}
