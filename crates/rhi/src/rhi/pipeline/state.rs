//! The raster fixed-state and vertex-input vocabulary.
//!
//! This module owns rhi-design sections 24 to 26.
//!
//! # What it is
//!
//! These are plain value types. They describe the closed, portable subset of
//! raster state that Fluxel freezes: vertex fetch layout, primitive assembly,
//! blend, depth/stencil, multisample, and the render-target signature that ties
//! a pipeline to the attachments it may be used with.
//!
//! # What it deliberately does not own
//!
//! The chapter freezes the commonly used state and nothing beyond it. Polygon
//! mode, depth clip control, depth bounds, conservative raster, programmable
//! sample positions, and variable-rate shading are not reserved here: adding a
//! field "for later" would freeze a contract with no consumer.
//!
//! Inline parameters, specialization constants, and a persistent pipeline cache
//! file format are likewise absent, by decision rather than by omission.
//!
//! # Why the vocabulary is shaped this way
//!
//! `strip_index_format` lives on [`PrimitiveState`] rather than being inferred
//! at draw time because WebGPU requires a strip draw's index format to be known
//! when the pipeline is created, and D3D12 carries a strip-cut value in the PSO.
//! Leaving it to the draw call would make the same pipeline mean two different
//! things on two backends.
//!
//! `ColorWriteMask` and the sparse `Option` slots in [`RenderTargetSignature`]
//! exist so that Vulkan and D3D12 sparse MRT locations are expressible without
//! inventing a second representation of the same target set.

use super::super::format::{FormatFacts, TextureFormat, format_facts};
use super::super::platform::{RhiError, RhiErrorKind, RhiResult};
use super::super::resource::{CompareFunction, TextureAspects};
use super::super::shader::{ShaderLocation, ShaderNumericType};

/// The index width of an indexed draw.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum IndexFormat {
    /// A 16-bit index.
    Uint16,
    /// A 32-bit index.
    Uint32,
}

impl IndexFormat {
    /// The byte size of one index.
    pub fn byte_size(self) -> u32 {
        match self {
            Self::Uint16 => 2,
            Self::Uint32 => 4,
        }
    }
}

/// The portable format of one vertex attribute.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum VertexFormat {
    /// One 32-bit float.
    Float32,
    /// Two 32-bit floats.
    Float32x2,
    /// Three 32-bit floats.
    Float32x3,
    /// Four 32-bit floats.
    Float32x4,

    /// One unsigned 32-bit integer.
    Uint32,
    /// Two unsigned 32-bit integers.
    Uint32x2,
    /// Three unsigned 32-bit integers.
    Uint32x3,
    /// Four unsigned 32-bit integers.
    Uint32x4,

    /// One signed 32-bit integer.
    Sint32,
    /// Two signed 32-bit integers.
    Sint32x2,
    /// Three signed 32-bit integers.
    Sint32x3,
    /// Four signed 32-bit integers.
    Sint32x4,

    /// Two normalized unsigned 8-bit values.
    Unorm8x2,
    /// Four normalized unsigned 8-bit values.
    Unorm8x4,
}

impl VertexFormat {
    /// The byte size of one attribute element.
    pub fn byte_size(self) -> u32 {
        match self {
            Self::Float32 | Self::Uint32 | Self::Sint32 => 4,
            Self::Float32x2 | Self::Uint32x2 | Self::Sint32x2 => 8,
            Self::Float32x3 | Self::Uint32x3 | Self::Sint32x3 => 12,
            Self::Float32x4 | Self::Uint32x4 | Self::Sint32x4 => 16,
            Self::Unorm8x2 => 2,
            Self::Unorm8x4 => 4,
        }
    }

    /// The numeric type this attribute presents to its shader location.
    ///
    /// Normalized formats present as floats; everything else keeps its
    /// signedness, because that is what the shader variable is declared as.
    pub fn shader_numeric_type(self) -> ShaderNumericType {
        match self {
            Self::Float32
            | Self::Float32x2
            | Self::Float32x3
            | Self::Float32x4
            | Self::Unorm8x2
            | Self::Unorm8x4 => ShaderNumericType::Float32,
            Self::Uint32 | Self::Uint32x2 | Self::Uint32x3 | Self::Uint32x4 => {
                ShaderNumericType::Uint32
            }
            Self::Sint32 | Self::Sint32x2 | Self::Sint32x3 | Self::Sint32x4 => {
                ShaderNumericType::Sint32
            }
        }
    }

    /// The number of components this attribute presents.
    pub fn components(self) -> u8 {
        match self {
            Self::Float32 | Self::Uint32 | Self::Sint32 => 1,
            Self::Float32x2 | Self::Uint32x2 | Self::Sint32x2 | Self::Unorm8x2 => 2,
            Self::Float32x3 | Self::Uint32x3 | Self::Sint32x3 => 3,
            Self::Float32x4 | Self::Uint32x4 | Self::Sint32x4 | Self::Unorm8x4 => 4,
        }
    }
}

/// Whether a vertex buffer advances per vertex or per instance.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VertexStepMode {
    /// One element per vertex.
    Vertex,
    /// One element per instance.
    Instance,
}

/// One attribute inside a vertex buffer.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VertexAttribute {
    /// The shader input location this attribute feeds.
    pub location: ShaderLocation,
    /// The attribute's format.
    pub format: VertexFormat,
    /// The byte offset of the attribute inside one element.
    pub offset: u64,
}

impl VertexAttribute {
    /// An attribute at `location` with `format`, starting at `offset`.
    pub fn new(location: ShaderLocation, format: VertexFormat, offset: u64) -> Self {
        Self {
            location,
            format,
            offset,
        }
    }

    fn end(&self) -> u64 {
        self.offset + u64::from(self.format.byte_size())
    }
}

/// The stride, step mode, and attributes of one vertex buffer slot.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VertexBufferLayout {
    /// The byte distance between consecutive elements.
    pub stride: u64,
    /// Whether elements advance per vertex or per instance.
    pub step_mode: VertexStepMode,
    /// The attributes fetched from this buffer.
    pub attributes: Vec<VertexAttribute>,
}

impl VertexBufferLayout {
    /// An empty layout with `stride` and `step_mode`.
    pub fn new(stride: u64, step_mode: VertexStepMode) -> Self {
        Self {
            stride,
            step_mode,
            attributes: Vec::new(),
        }
    }

    /// Adds one attribute.
    pub fn with_attribute(mut self, attribute: VertexAttribute) -> Self {
        self.attributes.push(attribute);
        self
    }
}

/// The complete vertex fetch layout of a raster pipeline.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VertexInputState {
    /// The vertex buffer slots, whose vector index is the slot number.
    pub buffers: Vec<VertexBufferLayout>,
}

impl VertexInputState {
    /// An empty layout.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds one buffer slot.
    pub fn with_buffer(mut self, layout: VertexBufferLayout) -> Self {
        self.buffers.push(layout);
        self
    }
}

/// The portable vertex-fetch limits a pipeline must fit inside.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct VertexInputLimits {
    /// `LimitKey::MaxVertexBuffers`.
    pub max_buffers: u64,
    /// `LimitKey::MaxVertexAttributes`.
    pub max_attributes: u64,
    /// `LimitKey::MaxVertexBufferArrayStride`.
    pub max_stride: u64,
}

/// Validates a vertex layout against the device's vertex-fetch limits.
pub(crate) fn validate_vertex_input(
    state: &VertexInputState,
    limits: VertexInputLimits,
) -> RhiResult<()> {
    if state.buffers.len() as u64 > limits.max_buffers {
        return Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "vertex input declares more buffers than MaxVertexBuffers",
        ));
    }
    let mut attribute_count: u64 = 0;
    for (slot, buffer) in state.buffers.iter().enumerate() {
        if buffer.stride > limits.max_stride {
            return Err(RhiError::new(
                RhiErrorKind::Unsupported,
                format!("vertex buffer slot {slot} stride exceeds MaxVertexBufferArrayStride"),
            ));
        }
        attribute_count += buffer.attributes.len() as u64;
        for attribute in &buffer.attributes {
            if attribute.end() > buffer.stride {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    format!(
                        "vertex buffer slot {slot} attribute at location {} runs past the stride",
                        attribute.location.get()
                    ),
                ));
            }
        }
        for window in buffer.attributes.windows(2) {
            if window[0].location == window[1].location {
                return Err(RhiError::new(
                    RhiErrorKind::InvalidUsage,
                    format!(
                        "vertex buffer slot {slot} declares location {} twice",
                        window[0].location.get()
                    ),
                ));
            }
        }
    }
    if attribute_count > limits.max_attributes {
        return Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "vertex input declares more attributes than MaxVertexAttributes",
        ));
    }
    Ok(())
}

/// Looks up the attribute feeding `location` across every buffer slot.
pub(crate) fn find_vertex_attribute(
    state: &VertexInputState,
    location: ShaderLocation,
) -> Option<&VertexAttribute> {
    state
        .buffers
        .iter()
        .flat_map(|buffer| buffer.attributes.iter())
        .find(|attribute| attribute.location == location)
}

/// The portable subset of primitive assembly state.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PrimitiveTopology {
    /// One point per vertex.
    PointList,
    /// Independent line segments.
    LineList,
    /// A connected line strip.
    LineStrip,
    /// Independent triangles.
    TriangleList,
    /// A connected triangle strip.
    TriangleStrip,
}

impl PrimitiveTopology {
    /// Whether this topology is a strip and therefore consumes a strip index
    /// format when drawn indexed.
    pub fn is_strip(self) -> bool {
        matches!(self, Self::LineStrip | Self::TriangleStrip)
    }
}

/// The winding that counts as front-facing.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FrontFace {
    /// Counter-clockwise is front-facing.
    Ccw,
    /// Clockwise is front-facing.
    Cw,
}

/// Which faces are discarded.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CullMode {
    /// No face is discarded.
    None,
    /// Front faces are discarded.
    Front,
    /// Back faces are discarded.
    Back,
}

/// Portable depth bias.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthBiasState {
    /// The constant depth bias.
    pub constant: i32,
    /// The slope-scaled depth bias.
    pub slope_scale: f32,
}

impl DepthBiasState {
    /// A bias with the given constant and slope scale.
    pub fn new(constant: i32, slope_scale: f32) -> Self {
        Self {
            constant,
            slope_scale,
        }
    }
}

/// Portable primitive assembly state.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct PrimitiveState {
    /// How vertices become primitives.
    pub topology: PrimitiveTopology,
    /// Which winding is front-facing.
    pub front_face: FrontFace,
    /// Which faces are discarded.
    pub cull_mode: CullMode,
    /// Optional depth bias.
    pub depth_bias: Option<DepthBiasState>,
    /// The strip index format, legal only for strip topologies.
    pub strip_index_format: Option<IndexFormat>,
}

impl PrimitiveState {
    /// Triangle-list state with counter-clockwise front faces and no culling.
    pub fn new(topology: PrimitiveTopology) -> Self {
        Self {
            topology,
            front_face: FrontFace::Ccw,
            cull_mode: CullMode::None,
            depth_bias: None,
            strip_index_format: None,
        }
    }

    /// Sets the front face.
    pub fn with_front_face(mut self, front_face: FrontFace) -> Self {
        self.front_face = front_face;
        self
    }

    /// Sets the cull mode.
    pub fn with_cull_mode(mut self, cull_mode: CullMode) -> Self {
        self.cull_mode = cull_mode;
        self
    }

    /// Sets the depth bias.
    pub fn with_depth_bias(mut self, bias: DepthBiasState) -> Self {
        self.depth_bias = Some(bias);
        self
    }

    /// Sets the strip index format.
    pub fn with_strip_index_format(mut self, format: IndexFormat) -> Self {
        self.strip_index_format = Some(format);
        self
    }
}

/// Validates primitive state in isolation.
pub(crate) fn validate_primitive_state(state: &PrimitiveState) -> RhiResult<()> {
    if state.strip_index_format.is_some() && !state.topology.is_strip() {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "strip index format is legal only for strip topologies",
        ));
    }
    if let Some(bias) = state.depth_bias {
        if !state.topology.is_triangle() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "portable depth bias is legal only for triangle topologies",
            ));
        }
        if !bias.slope_scale.is_finite() {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "depth bias slope scale must be finite",
            ));
        }
    }
    Ok(())
}

impl PrimitiveTopology {
    /// Whether this topology assembles triangles.
    pub(crate) fn is_triangle(self) -> bool {
        matches!(self, Self::TriangleList | Self::TriangleStrip)
    }
}

/// A blend factor.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlendFactor {
    /// Zero.
    Zero,
    /// One.
    One,

    /// The source component.
    Src,
    /// One minus the source component.
    OneMinusSrc,
    /// The source alpha.
    SrcAlpha,
    /// One minus the source alpha.
    OneMinusSrcAlpha,

    /// The destination component.
    Dst,
    /// One minus the destination component.
    OneMinusDst,
    /// The destination alpha.
    DstAlpha,
    /// One minus the destination alpha.
    OneMinusDstAlpha,

    /// The saturated source alpha.
    SrcAlphaSaturated,

    /// The current dynamic blend constant.
    Constant,
    /// One minus the current dynamic blend constant.
    OneMinusConstant,
}

/// A blend operation.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlendOperation {
    /// Source plus destination.
    Add,
    /// Source minus destination.
    Subtract,
    /// Destination minus source.
    ReverseSubtract,
    /// The smaller value.
    Min,
    /// The larger value.
    Max,
}

impl BlendOperation {
    /// Whether this operation requires unit factors.
    ///
    /// `Min` and `Max` select an operand rather than combining two, so a
    /// non-unit factor has no meaning any backend could agree on.
    pub fn requires_unit_factors(self) -> bool {
        matches!(self, Self::Min | Self::Max)
    }
}

/// One blend component.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlendComponent {
    /// The source factor.
    pub src_factor: BlendFactor,
    /// The destination factor.
    pub dst_factor: BlendFactor,
    /// The combining operation.
    pub operation: BlendOperation,
}

impl BlendComponent {
    /// A component with the given factors and operation.
    pub fn new(
        src_factor: BlendFactor,
        dst_factor: BlendFactor,
        operation: BlendOperation,
    ) -> Self {
        Self {
            src_factor,
            dst_factor,
            operation,
        }
    }

    pub(crate) fn is_well_formed(&self) -> bool {
        if !self.operation.requires_unit_factors() {
            return true;
        }
        self.src_factor == BlendFactor::One && self.dst_factor == BlendFactor::One
    }
}

/// The color and alpha blend components of one target.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlendState {
    /// The color component.
    pub color: BlendComponent,
    /// The alpha component.
    pub alpha: BlendComponent,
}

impl BlendState {
    /// A blend state from a color and an alpha component.
    pub fn new(color: BlendComponent, alpha: BlendComponent) -> Self {
        Self { color, alpha }
    }

    pub(crate) fn is_well_formed(&self) -> bool {
        self.color.is_well_formed() && self.alpha.is_well_formed()
    }
}

/// A per-target write mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ColorWriteMask(u8);

impl ColorWriteMask {
    /// No channel is written.
    pub const NONE: Self = Self(0);
    /// The red channel is written.
    pub const RED: Self = Self(1 << 0);
    /// The green channel is written.
    pub const GREEN: Self = Self(1 << 1);
    /// The blue channel is written.
    pub const BLUE: Self = Self(1 << 2);
    /// The alpha channel is written.
    pub const ALPHA: Self = Self(1 << 3);
    /// Every channel is written.
    pub const ALL: Self = Self(0x0f);

    /// Whether every channel in `other` is present in `self`.
    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// The union of two masks.
    pub fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// The bits.
    pub fn bits(self) -> u8 {
        self.0
    }
}

/// The fixed state of one color attachment location.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct ColorTargetState {
    /// The attachment format this location must have.
    pub format: TextureFormat,
    /// Optional blending.
    pub blend: Option<BlendState>,
    /// The channels written at this location.
    pub write_mask: ColorWriteMask,
}

impl ColorTargetState {
    /// A target that writes every channel without blending.
    pub fn new(format: TextureFormat) -> Self {
        Self {
            format,
            blend: None,
            write_mask: ColorWriteMask::ALL,
        }
    }

    /// Enables blending.
    pub fn with_blend(mut self, blend: BlendState) -> Self {
        self.blend = Some(blend);
        self
    }

    /// Sets the write mask.
    pub fn with_write_mask(mut self, mask: ColorWriteMask) -> Self {
        self.write_mask = mask;
        self
    }
}

/// A stencil operation.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StencilOperation {
    /// Keep the current value.
    Keep,
    /// Write zero.
    Zero,
    /// Write the reference value.
    Replace,
    /// Bitwise-invert the current value.
    Invert,
    /// Increment, clamping at the maximum.
    IncrementClamp,
    /// Decrement, clamping at zero.
    DecrementClamp,
    /// Increment, wrapping to zero.
    IncrementWrap,
    /// Decrement, wrapping to the maximum.
    DecrementWrap,
}

/// The stencil state of one face.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StencilFaceState {
    /// The comparison function.
    pub compare: CompareFunction,
    /// The operation when the stencil test fails.
    pub fail_op: StencilOperation,
    /// The operation when the stencil test passes but depth fails.
    pub depth_fail_op: StencilOperation,
    /// The operation when both tests pass.
    pub pass_op: StencilOperation,
}

impl StencilFaceState {
    /// A face state that keeps the value on every path.
    pub fn new(compare: CompareFunction) -> Self {
        Self {
            compare,
            fail_op: StencilOperation::Keep,
            depth_fail_op: StencilOperation::Keep,
            pass_op: StencilOperation::Keep,
        }
    }
}

/// The front, back, and mask stencil state.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StencilState {
    /// The front-face state.
    pub front: StencilFaceState,
    /// The back-face state.
    pub back: StencilFaceState,
    /// The read mask.
    pub read_mask: u32,
    /// The write mask.
    pub write_mask: u32,
}

impl StencilState {
    /// A state with the given front and back faces and full masks.
    pub fn new(front: StencilFaceState, back: StencilFaceState) -> Self {
        Self {
            front,
            back,
            read_mask: u32::MAX,
            write_mask: u32::MAX,
        }
    }
}

/// The depth test state.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DepthState {
    /// Whether passing fragments write depth.
    pub write_enabled: bool,
    /// The comparison function.
    pub compare: CompareFunction,
}

impl DepthState {
    /// A read-only depth state with the given comparison.
    pub fn new(compare: CompareFunction) -> Self {
        Self {
            write_enabled: false,
            compare,
        }
    }

    /// Enables or disables depth writes.
    pub fn with_write_enabled(mut self, enabled: bool) -> Self {
        self.write_enabled = enabled;
        self
    }
}

/// The depth and stencil state of a raster pipeline.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct DepthStencilState {
    /// The attachment format both states must agree with.
    pub format: TextureFormat,
    /// Optional depth state.
    pub depth: Option<DepthState>,
    /// Optional stencil state.
    pub stencil: Option<StencilState>,
}

impl DepthStencilState {
    /// A state over `format` with both halves disabled.
    pub fn new(format: TextureFormat) -> Self {
        Self {
            format,
            depth: None,
            stencil: None,
        }
    }

    /// Enables depth.
    pub fn with_depth(mut self, depth: DepthState) -> Self {
        self.depth = Some(depth);
        self
    }

    /// Enables stencil.
    pub fn with_stencil(mut self, stencil: StencilState) -> Self {
        self.stencil = Some(stencil);
        self
    }
}

/// Validates depth/stencil state against the format's aspects.
pub(crate) fn validate_depth_stencil_state(
    state: &DepthStencilState,
) -> RhiResult<Option<FormatFacts>> {
    let facts = format_facts(state.format);
    if state.depth.is_none() && state.stencil.is_none() {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "depth/stencil state enables neither depth nor stencil",
        ));
    }
    if state.depth.is_some() && !facts.aspects().contains(TextureAspects::DEPTH) {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "depth is enabled on a format with no depth aspect",
        ));
    }
    if state.stencil.is_some() && !facts.aspects().contains(TextureAspects::STENCIL) {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "stencil is enabled on a format with no stencil aspect",
        ));
    }
    if !facts.depth_attachment() {
        return Err(RhiError::new(
            RhiErrorKind::Unsupported,
            "the depth/stencil format is not usable as an attachment",
        ));
    }
    Ok(Some(facts))
}

/// Multisample state.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MultisampleState {
    /// The sample count.
    pub count: u32,
    /// The portable sample mask.
    pub mask: u32,
    /// Whether alpha-to-coverage is enabled.
    pub alpha_to_coverage_enabled: bool,
}

impl MultisampleState {
    /// A state with the given sample count, the full mask, and no
    /// alpha-to-coverage.
    pub fn new(count: u32) -> Self {
        Self {
            count,
            mask: u32::MAX,
            alpha_to_coverage_enabled: false,
        }
    }

    /// Sets the sample mask.
    pub fn with_mask(mut self, mask: u32) -> Self {
        self.mask = mask;
        self
    }

    /// Enables or disables alpha-to-coverage.
    pub fn with_alpha_to_coverage(mut self, enabled: bool) -> Self {
        self.alpha_to_coverage_enabled = enabled;
        self
    }
}

/// Validates multisample state in isolation.
pub(crate) fn validate_multisample_state(state: &MultisampleState) -> RhiResult<()> {
    if state.count == 0 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "multisample count must be at least one",
        ));
    }
    if state.alpha_to_coverage_enabled && state.count <= 1 {
        return Err(RhiError::new(
            RhiErrorKind::InvalidUsage,
            "alpha-to-coverage requires a sample count above one",
        ));
    }
    Ok(())
}

/// The color and depth/stencil formats a raster pipeline is built against.
///
/// Trailing empty color locations are removed so that one target set has
/// exactly one representation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RenderTargetSignature {
    /// Vector index is the color attachment location; `None` is an empty slot.
    pub color_formats: Vec<Option<TextureFormat>>,
    /// The depth/stencil format, when the pipeline uses one.
    pub depth_stencil_format: Option<TextureFormat>,
    /// The sample count every active attachment must have.
    pub sample_count: u32,
}

impl RenderTargetSignature {
    /// Builds a signature, dropping trailing empty color locations.
    pub fn new(
        color_formats: Vec<Option<TextureFormat>>,
        depth_stencil_format: Option<TextureFormat>,
        sample_count: u32,
    ) -> Self {
        let mut color_formats = color_formats;
        while matches!(color_formats.last(), Some(None)) {
            color_formats.pop();
        }
        Self {
            color_formats,
            depth_stencil_format,
            sample_count,
        }
    }

    /// The number of active color locations, including holes below the last
    /// active one.
    pub fn active_color_count(&self) -> usize {
        self.color_formats
            .iter()
            .filter(|format| format.is_some())
            .count()
    }
}

/// Validates that a shader input location is served by a vertex attribute.
pub(crate) fn validate_vertex_shader_inputs(
    interface: &super::super::shader::ShaderInterface,
    state: &VertexInputState,
) -> RhiResult<()> {
    for input in interface.inputs() {
        if input.components == 0 || input.components > 4 {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "shader interface declares an input with an out-of-range component count",
            ));
        }
        let Some(attribute) = find_vertex_attribute(state, input.location) else {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "vertex shader input location {} has no vertex attribute",
                    input.location.get()
                ),
            ));
        };
        if attribute.format.shader_numeric_type() != input.numeric_type {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "vertex attribute at location {} does not match the shader input numeric type",
                    input.location.get()
                ),
            ));
        }
        if attribute.format.components() != input.components {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                format!(
                    "vertex attribute at location {} does not match the shader input component count",
                    input.location.get()
                ),
            ));
        }
    }
    Ok(())
}
