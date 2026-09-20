//! Pipeline interface, vertex input, raster fixed state, target signature, and
//! the two pipeline objects (specification sections 23 through 28).
//!
//! # What this module owns
//!
//! - [`PipelineInterface`], the logical layout contract between shader entry
//!   points and bind-group packets, and the two tokens of section 23.2 that
//!   describe it: [`PipelineInterfaceCompatibilityId`] (exact, same-Device,
//!   unforgeable) and [`LayoutFingerprint`](crate::api::binding::LayoutFingerprint)
//!   (a cache hint, reusing the one shared type that lives in
//!   [`crate::api::binding`]).
//! - Vertex input: [`VertexFormat`], [`VertexAttribute`], [`VertexBufferLayout`],
//!   [`VertexInputState`] (section 24).
//! - Raster fixed state: primitive, blend, depth/stencil, multisample (section
//!   25). The chapter deliberately excludes polygon mode, depth clip control,
//!   depth bounds, conservative raster, programmable sample positions, and VRS;
//!   they arrive later by capability family and are not reserved here.
//! - [`RenderTargetSignature`] (section 26), including its trailing-`None`
//!   canonicalization.
//! - [`RasterPipeline`] and [`ComputePipeline`] (sections 27 and 28).
//!
//! # What this module deliberately does not own
//!
//! - [`LayoutFingerprint`](crate::api::binding::LayoutFingerprint)'s algorithm, or either compatibility token's value.
//!   Both are outcomes of the Device's interning step (section 23.2), so they are
//!   constructor parameters here, exactly as in [`crate::api::binding`].
//! - [`crate::api::command::IndexFormat`]. Section 25.1 back-references it for
//!   `strip_index_format`, and section 04 owns it.
//! - Inline parameters. Section 23.4 freezes *nothing* in `PipelineInterface` for
//!   push constants, root constants, or immediate bytes, and forbids pre-adding a
//!   reserved range, so there is no field for them and no placeholder verb.
//! - The persistent pipeline cache format. Section 28.1 requires every descriptor
//!   to be re-described by artifact, interface, fixed state, and target signature,
//!   which the types below satisfy by construction; the file format itself is not
//!   frozen in 0.16. Pipeline creation remains the single async public seam: a
//!   backend may transparently use in-memory, driver, or persistent caches, but a
//!   cache key, blob, import/export operation, and hit/miss observation are not
//!   portable API until cross-platform semantics require them.
//!
//! # The device seam
//!
//! The three verbs of this chapter are `Device::create_pipeline_interface`,
//! `Device::create_raster_pipeline`, and `Device::create_compute_pipeline`.
//! Interface creation is synchronous logical creation; raster and compute
//! pipeline creation are async. Their portable halves live here as validators
//! that take the device's answers as `PipelineDeviceFacts`:
//!
//! ```text
//! limit               EnabledCapabilities::limit(key) -> Option<u64>
//! binding_support     EnabledCapabilities::binding_support(query)
//! binding_limit       EnabledCapabilities::binding_limit(stage, class)
//! feature_supported   EnabledCapabilities::supports_feature(feature)
//! shader_acceptance   EnabledCapabilities::shader_acceptance(artifact)
//! color_target_facts  FormatFacts::color_attachment / blendable /
//!                     has_alpha_channel / color_output_type  (section 27.3)
//! texture_support     EnabledCapabilities::texture_support(query)
//! ```
//!
//! `color_target_facts` is a small carrier rather than a direct `FormatFacts`
//! reference so validation fixtures can state the exact target facts they need
//! without constructing an entire capability snapshot. Production construction
//! maps it directly from the probed `FormatFacts` record.
//!
//! # The rules this module decides
//!
//! Section 23.1's aggregate counts, section 23.3's merge lattice, section 24.2's
//! vertex-input checks, section 27.3's ten validation blocks, and section 28's
//! capability gate. Every one of them runs before a backend
//! is touched, because section 4 forbids handing a backend a problem portable
//! validation could have found.
//!
//! # Files
//!
//! One section of the chapter per file, so that each file states its own rules
//! and no file states two sections' rules at once:
//!
//! ```text
//! mod.rs           the device-answer carrier both pipeline validators take
//! interface.rs     section 23.1-23.2, PipelineInterface and its descriptor
//! resources.rs     section 23.3, the shader requirement merge lattice
//! vertex_input.rs  section 24, vertex input
//! raster_state.rs  section 25, the raster fixed-function state
//! raster.rs        section 26-27, the raster pipeline and its target signature
//! compute.rs       section 28, the compute pipeline
//! ```
//!
//! This file keeps only the `PipelineDeviceFacts` and `ColorTargetFacts`
//! carriers that more than one of them takes. It decides
//! no rule of its own — every rule lives in the file for the section that states
//! it.
//!
//! The submodules are `pub(crate)` and nothing crate-private is re-exported, so a
//! crate-internal caller names the file that defines the item, e.g.
//! `merge_shader_resources`. A
//! `pub(crate) use` of one would only obscure which file owns the validation
//! rule, so this module does not carry lint attributes as a substitute for a
//! caller.

// The submodules stay crate-visible rather than private because the validators
// they own are crate-private entry points of their own: a `pub(crate) use` of one
// would obscure the owner of the rule, and this file does not carry lint
// attributes as a substitute for a caller. Paths point at the module that
// defines the item.
pub(crate) mod backend;
pub(crate) mod compute;
pub(crate) mod interface;
pub(crate) mod raster;
pub(crate) mod raster_state;
pub(crate) mod resources;
pub(crate) mod vertex_input;

pub use compute::{ComputePipeline, ComputePipelineDescriptor};
pub use interface::{
    PipelineInterface, PipelineInterfaceCompatibilityId, PipelineInterfaceDescriptor,
};
pub use raster::{RasterPipeline, RasterPipelineDescriptor, RenderTargetSignature};
pub use raster_state::{
    BlendComponent, BlendFactor, BlendOperation, BlendState, ColorTargetState, ColorWriteMask,
    CullMode, DepthBiasState, DepthState, DepthStencilState, FrontFace, MultisampleState,
    PrimitiveState, PrimitiveTopology, StencilFaceState, StencilOperation, StencilState,
};
pub use vertex_input::{
    VertexAttribute, VertexBufferLayout, VertexFormat, VertexInputState, VertexStepMode,
};

use crate::api::binding::{BindingLimitClass, BindingSupport, BindingSupportQuery};
use crate::api::format::{TextureFormat, TextureSupport, TextureSupportQuery};
use crate::api::platform::requirements::{LimitKey, OptionalFeature};
use crate::api::shader::{ArtifactAcceptance, ShaderArtifact, ShaderNumericType, ShaderStage};

/// The device answers the pipeline rules of this chapter need.
///
/// One struct rather than seven parameters, because every field names a question
/// the device already answers in another module's vocabulary, and a call site
/// that reads `binding_limit: &limits` is a call site a reviewer can check against
/// the specification's list without counting argument positions.
///
/// Each field is a `dyn` reference so that the two validators can share one value
/// without each monomorphizing a different closure tuple.
pub(crate) struct PipelineDeviceFacts<'a> {
    /// A device limit. `None` means the device does not expose that key, which
    /// section 23.1 and section 27.3 both treat as "not applicable" rather than as
    /// zero.
    pub(crate) limit: &'a dyn Fn(LimitKey) -> Option<u64>,

    /// Whether, and how, the device can express a binding (section 20.4).
    pub(crate) binding_support: &'a dyn Fn(&BindingSupportQuery) -> BindingSupport,

    /// The device's binding-count ceiling for one stage and resource class.
    pub(crate) binding_limit: &'a dyn Fn(ShaderStage, BindingLimitClass) -> Option<u32>,

    /// Whether an optional feature is enabled on this device (section 28's
    /// `OptionalFeature::Compute` gate).
    pub(crate) feature_supported: &'a dyn Fn(OptionalFeature) -> bool,

    /// Whether the device accepts one shader artifact (section 19.7's features and
    /// limits, answered by their owner rather than re-derived here).
    pub(crate) shader_acceptance: &'a dyn Fn(&ShaderArtifact) -> ArtifactAcceptance,

    /// The probed per-format facts section 27.3's "Target facts" block compares
    /// against.
    pub(crate) color_target_facts: &'a dyn Fn(TextureFormat) -> Option<ColorTargetFacts>,

    /// Whether a texture key can be created (section 8.3).
    pub(crate) texture_support: &'a dyn Fn(&TextureSupportQuery) -> TextureSupport,
}

/// The four format facts section 27.3's "Target facts" block names.
///
/// A compact validation carrier. The façade fills it from the probed
/// [`crate::api::format::FormatFacts`], while fixtures can state only the four
/// facts a target-validation case varies.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ColorTargetFacts {
    /// `FormatFacts::color_attachment`.
    pub(crate) color_attachment: bool,
    /// `FormatFacts::blendable`.
    pub(crate) blendable: bool,
    /// `FormatFacts::has_alpha_channel`.
    pub(crate) has_alpha_channel: bool,
    /// `FormatFacts::color_output_type`, or `None` when the format has no shader
    /// color output type.
    pub(crate) color_output_type: Option<ShaderNumericType>,
}

impl ColorTargetFacts {
    /// Records one format's target facts.
    pub(crate) fn new(
        color_attachment: bool,
        blendable: bool,
        has_alpha_channel: bool,
        color_output_type: Option<ShaderNumericType>,
    ) -> Self {
        Self {
            color_attachment,
            blendable,
            has_alpha_channel,
            color_output_type,
        }
    }
}
