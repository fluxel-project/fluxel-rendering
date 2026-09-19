//! Specification section 54: the captured object definitions.
//!
//! One responsibility: **what an object is, stated without a handle to it.**
//! Every type here is the tooling-side projection of a live RHI object's
//! descriptor, and together they are the object reconstruction graph of section
//! 52.3 — a TextureView names its Texture, a BindGroup names its layout and its
//! resources, a PipelineInterface names its layouts, a RasterPipeline names its
//! modules, its interface, and its fixed state.
//!
//! Not owned here: the events that carry these definitions ([`super::event`]) and
//! the mutation and command values that are *not* object definitions
//! ([`super::mutation`]). Nor is the capture-local typed ID: section 54 states
//! that the Capture Artifact Layer maps a runtime [`ObjectId`] to its own
//! capture-local ID, and that RHI tooling defines no disk typed-ID encoding. So
//! the identity in every type below is the runtime one, and the mapping is
//! deliberately absent rather than stubbed.
//!
//! # The invariant this file exists to keep
//!
//! Section 54 restricts a tooling definition to value types, [`ObjectId`], and
//! [`AcquiredFrameId`](crate::api::presentation::AcquiredFrameId). A live RHI
//! handle must not be included. That is not
//! tidiness: a definition holding a `Buffer` would keep the buffer alive and
//! make the record outlive the device by accident, and it would let a captured
//! record be read back as a *usable* resource, which is Replay's job and not a
//! reader's. Every field below is therefore one of:
//!
//! ```text
//! ObjectId              a name, not a handle
//! AcquiredFrameId       a name, not a handle
//! a descriptor          a value type, and one from the section that owns it
//! a Label               a value type
//! ```
//!
//! # Why the descriptors are the live ones
//!
//! [`CapturedObjectDefinition::Buffer`] carries a
//! [`crate::api::resource::buffer::BufferDescriptor`] and not a tooling-local
//! copy of it. Two shapes for "what a buffer is" would be two things to keep in
//! step, and the live descriptor is already a value type with public fields, so
//! the copy would buy nothing and cost the drift. Sections 55 and 56 are written
//! the same way, except where a live type is *directional* and the captured one
//! must not be — which is a note for the file that hits it.

use crate::api::binding::layout::BindGroupLayoutDescriptor;
use crate::api::binding::vocabulary::BindingSlotId;
use crate::api::identity::{DeviceIdentity, Label, ObjectId};
use crate::api::pipeline::{
    ColorTargetState, DepthStencilState, MultisampleState, PrimitiveState, VertexInputState,
};
use crate::api::presentation::PresentationConfiguration;
use crate::api::resource::buffer::{BufferDescriptor, BufferRange};
use crate::api::resource::sampler::SamplerDescriptor;
use crate::api::resource::texture::TextureDescriptor;
use crate::api::resource::view::TextureViewDescriptor;
use crate::api::shader::ShaderArtifact;

/// A captured resource reference, as a bind group entry holds it.
///
/// The tooling-side counterpart of
/// [`crate::api::binding::group::BindingResource`], and the difference between
/// the two is the whole point of section 54: the live type holds
/// `BufferBinding`, `TextureView`, and `Sampler` *handles*, and this one holds
/// their [`ObjectId`]s and the ranges that describe the binding.
///
/// The scalar and array variants stay separate here for the same reason they are
/// separate in the live type: [`crate::api::binding::vocabulary::BindingCount`]
/// says which one a slot takes, and an array of length one cannot stand in for a
/// scalar. Collapsing them into a `Vec` variant would erase that correspondence
/// from the captured record, and a Replay rebuilding the packet would then have
/// to re-derive a rule the record no longer states.
#[non_exhaustive]
#[derive(Clone)]
pub enum CapturedBindingResource {
    /// One buffer range.
    Buffer {
        /// The buffer's identity.
        buffer: ObjectId,
        /// The range of the buffer this binding sees.
        range: BufferRange,
    },

    /// One texture view.
    TextureView {
        /// The view's identity. The view's own definition names its texture.
        view: ObjectId,
    },

    /// One sampler.
    Sampler {
        /// The sampler's identity.
        sampler: ObjectId,
    },

    /// A fixed-length array of buffer ranges.
    BufferArray(Vec<(ObjectId, BufferRange)>),

    /// A fixed-length array of texture views.
    TextureViewArray(Vec<ObjectId>),

    /// A fixed-length array of samplers.
    SamplerArray(Vec<ObjectId>),
}

/// One slot of a captured bind group packet.
///
/// `slot` is a [`BindingSlotId`] rather than a raw index because the slot is a
/// logical position the caller chose while writing the layout, and the captured
/// record has to agree with the layout it names about which position that is.
#[derive(Clone)]
pub struct CapturedBindGroupEntry {
    /// The slot being filled.
    pub slot: BindingSlotId,

    /// The resource filling it, named rather than held.
    pub resource: CapturedBindingResource,
}

/// What a captured bind group packet is.
///
/// `layout` is an [`ObjectId`] and not a
/// [`BindGroupLayoutDescriptor`](crate::api::binding::layout::BindGroupLayoutDescriptor):
/// the layout has a definition of its own in this graph, and inlining it here
/// would duplicate the layout once per group bound to it — which is how a
/// reconstruction graph becomes a tree and stops being a graph.
#[derive(Clone)]
pub struct CapturedBindGroupDefinition {
    /// Diagnostic label, excluded from every canonical hash.
    pub label: Label,

    /// The layout this packet was validated against.
    pub layout: ObjectId,

    /// The resources, in the canonical ascending-slot order the packet was
    /// created with.
    pub entries: Vec<CapturedBindGroupEntry>,
}

/// What a captured pipeline interface is.
///
/// The layout *order* is the interface's identity —
/// [`crate::api::pipeline::PipelineInterface`] indexes its groups by
/// [`crate::api::binding::BindGroupIndex`] — so this is a `Vec<ObjectId>` and
/// not a set.
#[derive(Clone)]
pub struct CapturedPipelineInterfaceDefinition {
    /// Diagnostic label.
    pub label: Label,

    /// One layout identity per group, in group-index order.
    pub groups: Vec<ObjectId>,
}

/// What a captured raster pipeline is.
///
/// The three identity fields are section 52.3's edges — the two shader modules
/// and the interface — and the five state fields are the "fixed state" that
/// section 52.3 lists alongside them. Both halves are needed: the identities say
/// what the pipeline is made of, and the state says what it does, and neither is
/// derivable from the other.
///
/// The state types are the live ones rather than tooling copies, because each is
/// already a plain value the live pipeline stores, and a tooling copy would be a
/// second place for a fixed-state field to be forgotten.
///
/// [`crate::api::pipeline::RenderTargetSignature`] is deliberately *not* a
/// field. Section 26 derives the target signature from `color_targets` and the
/// multisample state, both of which are here, so carrying it would be carrying a
/// computed value beside its inputs.
#[derive(Clone)]
pub struct CapturedRasterPipelineDefinition {
    /// Diagnostic label.
    pub label: Label,

    /// The vertex-stage module.
    pub vertex: ObjectId,

    /// The fragment-stage module, when the pipeline has one.
    pub fragment: Option<ObjectId>,

    /// The interface this pipeline's groups are validated against.
    pub interface: ObjectId,

    /// Vertex input state (section 24).
    pub vertex_input: VertexInputState,

    /// Primitive assembly and rasterization state (section 25).
    pub primitive: PrimitiveState,

    /// Depth and stencil state, when the pipeline has any.
    pub depth_stencil: Option<DepthStencilState>,

    /// Multisample state.
    pub multisample: MultisampleState,

    /// One colour target state per attachment location, with `None` for a hole.
    ///
    /// A hole is preserved rather than compacted: section 26's canonicalization
    /// removes trailing holes only, and a record that dropped interior ones would
    /// describe a pipeline with fewer attachment locations than the one that ran.
    pub color_targets: Vec<Option<ColorTargetState>>,
}

/// What a captured compute pipeline is.
#[derive(Clone)]
pub struct CapturedComputePipelineDefinition {
    /// Diagnostic label.
    pub label: Label,

    /// The compute-stage module.
    pub shader: ObjectId,

    /// The interface this pipeline's groups are validated against.
    pub interface: ObjectId,
}

/// A capture-local key for a host object supplied by an external fixture.
///
/// Explicitly not an OS or native graphics handle (section 54). It is a string
/// key into whatever the ReplayRuntime or its artifact layer is holding, which is
/// why it is the one type in this module that a caller may construct freely: it
/// names *their* object, not ours, and minting one forges nothing.
///
/// This is the seam that lets a captured presentation target be described
/// without serializing a surface: the record says "the target the fixture
/// provider calls `main-window`", and the provider decides what that is.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CapturedPresentationTargetFixture {
    /// The fixture provider's key.
    pub key: String,
}

/// What a captured presentation target is.
///
/// One field, and that is the whole answer: a target has no portable descriptor
/// to record, because section 42.1 keeps every native surface type out of the
/// portable API. Reconstructing the target is binding the fixture, and there is
/// nothing else to say about it.
#[derive(Clone)]
pub struct CapturedPresentationTargetDefinition {
    /// The external fixture this target is reconstructed from.
    pub fixture: CapturedPresentationTargetFixture,
}

/// What a captured configured presentation is.
///
/// The edge back to the target is `target`, which is why section 58.1 can require
/// a `FrameAcquired` event's `configured_presentation` to be describable: it
/// names its own target, and the target names its fixture, so the chain reaches
/// an external fixture without ever holding a host handle.
#[derive(Clone)]
pub struct CapturedConfiguredPresentationDefinition {
    /// The device that holds the lease.
    pub device: DeviceIdentity,

    /// The target the lease is over.
    pub target: ObjectId,

    /// The configuration the lease was taken with.
    ///
    /// The whole configuration, not its parts:
    /// [`PresentationConfiguration`] is the value a reconfigure replaces and the
    /// value a present was made under, so recording it as a whole is what lets a
    /// record say which configuration a frame was presented with.
    pub configuration: PresentationConfiguration,
}

/// One object's complete definition, as the tooling side sees it.
///
/// Section 54's enum, and the answer to both halves of the lazy seam:
/// [`super::event::SemanticEvent::ObjectCreated`] carries one by reference, and
/// [`super::ToolingAccess::describe_object`] returns one by value.
///
/// # The variants are not a mirror of the resource taxonomy
///
/// There is no variant for a readback ticket, an upload job, a recorded work
/// item, or a submission, even though section 52.1 makes all of those observable
/// objects. That is deliberate rather than an omission: each of them has a
/// *definition type of its own* in this chapter (section 55 for upload and
/// readback, section 56 for recorded work, section 57 for plans and receipts),
/// and their events carry those types directly. Folding them in here would give
/// two ways to name the same thing, and a reader would then have to know which
/// event used which.
///
/// # `Shader` carries the artifact, not a lowered form
///
/// Section 52.7's list is code, ABI version, interface, requirements, and
/// provenance — which is exactly a [`ShaderArtifact`]. The record therefore
/// carries the artifact whole, because the decision section 52.7 gives the
/// ReplayRuntime ("can the target backend accept it directly, must it be
/// recompiled, or is it executable-only and therefore unsupported") is a decision
/// made *from* the provenance by a layer that has a cross-backend toolchain. RHI
/// does not have one, so it does not summarize.
#[non_exhaustive]
#[derive(Clone)]
pub enum CapturedObjectDefinition {
    /// The target is reconstructed by binding `fixture` to an external
    /// presentation fixture; tooling never serializes a native target.
    PresentationTarget {
        /// The target's identity.
        id: ObjectId,
        /// What it is.
        definition: CapturedPresentationTargetDefinition,
    },

    /// A configured presentation lease.
    ConfiguredPresentation {
        /// The lease's identity.
        id: ObjectId,
        /// What it is.
        definition: CapturedConfiguredPresentationDefinition,
    },

    /// A buffer.
    Buffer {
        /// The buffer's identity.
        id: ObjectId,
        /// The descriptor it was created from.
        descriptor: BufferDescriptor,
    },

    /// A texture.
    Texture {
        /// The texture's identity.
        id: ObjectId,
        /// The descriptor it was created from.
        descriptor: TextureDescriptor,
    },

    /// A texture view. The edge to its texture is the reconstruction graph's
    /// first link.
    TextureView {
        /// The view's identity.
        id: ObjectId,
        /// The texture it views.
        texture: ObjectId,
        /// The descriptor it was created from.
        descriptor: TextureViewDescriptor,
    },

    /// A sampler.
    Sampler {
        /// The sampler's identity.
        id: ObjectId,
        /// The descriptor it was created from.
        descriptor: SamplerDescriptor,
    },

    /// A shader module, carrying its whole artifact.
    Shader {
        /// The module's identity.
        id: ObjectId,
        /// The artifact it was created from.
        artifact: ShaderArtifact,
    },

    /// A bind group layout.
    ///
    /// The descriptor here is the *canonicalized* one, because that is what the
    /// live layout stores and therefore what the layout is: a record showing the
    /// caller's pre-canonicalization ordering would not reconstruct the object
    /// that existed.
    BindGroupLayout {
        /// The layout's identity.
        id: ObjectId,
        /// The canonicalized descriptor.
        descriptor: BindGroupLayoutDescriptor,
    },

    /// A bind group packet.
    BindGroup {
        /// The packet's identity.
        id: ObjectId,
        /// What it is.
        definition: CapturedBindGroupDefinition,
    },

    /// A pipeline interface.
    PipelineInterface {
        /// The interface's identity.
        id: ObjectId,
        /// What it is.
        definition: CapturedPipelineInterfaceDefinition,
    },

    /// A raster pipeline.
    RasterPipeline {
        /// The pipeline's identity.
        id: ObjectId,
        /// What it is.
        definition: CapturedRasterPipelineDefinition,
    },

    /// A compute pipeline.
    ComputePipeline {
        /// The pipeline's identity.
        id: ObjectId,
        /// What it is.
        definition: CapturedComputePipelineDefinition,
    },
}

// No `expect(unused_imports)` above, unlike the sibling chapters: every import in
// this file is named by a field, and the three types named only in doc links are
// written out as full paths rather than imported for that reason. A reader who
// has seen such an expectation elsewhere should not add one here.
