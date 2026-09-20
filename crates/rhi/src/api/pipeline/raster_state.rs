//! Section 25: raster fixed-function state.
//!
//! Primitive, blend, depth/stencil and multisample state as data: what a caller
//! states, with constructors that make the common legal state cheap to write. The
//! chapter keeps depth bounds, programmable sample positions, and VRS absent
//! rather than reserving placeholders. Polygon mode, depth clip control, and
//! conservative rasterization are present and capability-gated at pipeline
//! creation.
//!
//! Not owned here: every rule that compares this state against a device, a target
//! signature or a shader. Those live with the pipeline, in `raster.rs`, because
//! they are conditions of pipeline creation rather than properties of the state.

use crate::api::command::IndexFormat;
use crate::api::format::TextureFormat;
use crate::api::resource::sampler::CompareFunction;

// ---------------------------------------------------------------------------
// Section 25 - Raster fixed state
// ---------------------------------------------------------------------------

/// How vertices are assembled into primitives.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrimitiveTopology {
    /// Isolated points.
    PointList,
    /// Isolated line segments.
    LineList,
    /// A connected line strip.
    LineStrip,
    /// Isolated triangles.
    TriangleList,
    /// A connected triangle strip.
    TriangleStrip,
}

impl PrimitiveTopology {
    /// Whether this topology is a strip.
    ///
    /// The question section 27.3's strip rule is written against, stated once so
    /// that "legal only for LineStrip / TriangleStrip" cannot drift between the
    /// two places that need it.
    pub fn is_strip(self) -> bool {
        match self {
            Self::PointList | Self::LineList | Self::TriangleList => false,
            Self::LineStrip | Self::TriangleStrip => true,
        }
    }
}

/// Which winding order is front-facing.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrontFace {
    /// Counter-clockwise in framebuffer coordinates.
    Ccw,
    /// Clockwise in framebuffer coordinates.
    Cw,
}

/// Which faces are discarded.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CullMode {
    /// No culling.
    None,
    /// Discard front-facing primitives.
    Front,
    /// Discard back-facing primitives.
    Back,
}

/// How rasterization covers a primitive.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolygonMode {
    /// Fill the primitive interior.
    Fill,
    /// Rasterize polygon edges.
    Line,
    /// Rasterize polygon vertices.
    Point,
}

/// A portable depth bias.
///
/// Constant and slope are portable baseline fields. A non-zero clamp is an
/// explicit `DepthBiasClamp` capability request; line/point-specific bias forms
/// remain absent because the RHI has no portable lowering contract for them.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthBiasState {
    /// Constant offset added to the depth value.
    pub constant: i32,
    /// Offset scaled by the primitive's depth slope.
    pub slope_scale: f32,
    /// Maximum absolute depth-bias contribution. Zero is the portable baseline.
    pub clamp: f32,
}

impl DepthBiasState {
    /// States a depth bias.
    pub fn new(constant: i32, slope_scale: f32) -> Self {
        Self {
            constant,
            slope_scale,
            clamp: 0.0,
        }
    }

    /// Sets a finite depth-bias clamp.
    pub fn with_clamp(mut self, clamp: f32) -> Self {
        self.clamp = clamp;
        self
    }
}

/// The primitive assembly and rasterization state.
///
/// `DepthBiasState::slope_scale` must be finite and the portable contract admits
/// bias only for triangle topology, both of which section 27.3's strip-topology block
/// checks; neither is decided here, because a state value carries no other value
/// to compare against.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct PrimitiveState {
    /// How vertices are assembled.
    pub topology: PrimitiveTopology,
    /// Which winding order is front-facing.
    pub front_face: FrontFace,
    /// Which faces are discarded.
    pub cull_mode: CullMode,

    /// Polygon coverage mode.
    pub polygon_mode: PolygonMode,
    /// When true, primitives are not clipped against the depth range.
    pub unclipped_depth: bool,
    /// When true, rasterization conservatively covers touched pixels.
    pub conservative: bool,

    /// Portable depth bias. A non-zero clamp requires `DepthBiasClamp`.
    pub depth_bias: Option<DepthBiasState>,

    /// Legal only for LineStrip / TriangleStrip.
    ///
    /// If Some, the IndexFormat of indexed strip draws must match,
    /// and the corresponding fixed primitive-restart value is enabled.
    pub strip_index_format: Option<IndexFormat>,
}

impl PrimitiveState {
    /// States a topology with the portable defaults.
    ///
    /// `Ccw` front face, no culling, no depth bias, no strip index format. Section
    /// 25.1 fixes only the topology parameter, so the other four are the values a
    /// caller would otherwise have to write to get a pipeline that draws.
    pub fn new(topology: PrimitiveTopology) -> Self {
        Self {
            topology,
            front_face: FrontFace::Ccw,
            cull_mode: CullMode::None,
            polygon_mode: PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
            depth_bias: None,
            strip_index_format: None,
        }
    }

    /// Sets the front-face winding.
    pub fn with_front_face(mut self, front_face: FrontFace) -> Self {
        self.front_face = front_face;
        self
    }

    /// Sets the cull mode.
    pub fn with_cull_mode(mut self, cull_mode: CullMode) -> Self {
        self.cull_mode = cull_mode;
        self
    }

    /// Sets the polygon coverage mode.
    pub fn with_polygon_mode(mut self, polygon_mode: PolygonMode) -> Self {
        self.polygon_mode = polygon_mode;
        self
    }
    /// Enables or disables unclipped depth.
    pub fn with_unclipped_depth(mut self, enabled: bool) -> Self {
        self.unclipped_depth = enabled;
        self
    }
    /// Enables or disables conservative rasterization.
    pub fn with_conservative(mut self, enabled: bool) -> Self {
        self.conservative = enabled;
        self
    }

    /// Enables a depth bias.
    pub fn with_depth_bias(mut self, bias: DepthBiasState) -> Self {
        self.depth_bias = Some(bias);
        self
    }

    /// Declares the index format of this strip's indexed draws.
    ///
    /// Section 25.1's reason is backend-visible rather than tidiness: WebGPU
    /// requires the pipeline of an indexed strip draw to fix the strip index
    /// format, and D3D12's PSO carries the strip-cut value, so leaving it to the
    /// draw call would be a fact no backend could recover.
    pub fn with_strip_index_format(mut self, format: IndexFormat) -> Self {
        self.strip_index_format = Some(format);
        self
    }
}

/// One factor of a blend equation.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendFactor {
    /// Zero.
    Zero,
    /// One.
    One,

    /// The source color.
    Src,
    /// One minus the source color.
    OneMinusSrc,
    /// The source alpha.
    SrcAlpha,
    /// One minus the source alpha.
    OneMinusSrcAlpha,

    /// The second source color output.
    Src1,
    /// One minus the second source color output.
    OneMinusSrc1,
    /// The second source alpha output.
    Src1Alpha,
    /// One minus the second source alpha output.
    OneMinusSrc1Alpha,

    /// The destination color.
    Dst,
    /// One minus the destination color.
    OneMinusDst,
    /// The destination alpha.
    DstAlpha,
    /// One minus the destination alpha.
    OneMinusDstAlpha,

    /// The source alpha, saturated.
    SrcAlphaSaturated,

    /// Uses the current dynamic blend constant.
    Constant,
    /// One minus the current dynamic blend constant.
    OneMinusConstant,
}

/// How two blend factors are combined.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendOperation {
    /// `src + dst`.
    Add,
    /// `src - dst`.
    Subtract,
    /// `dst - src`.
    ReverseSubtract,
    /// The smaller of the two.
    Min,
    /// The larger of the two.
    Max,
}

/// One blend equation: two factors and an operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlendComponent {
    /// The factor applied to the source.
    pub src_factor: BlendFactor,
    /// The factor applied to the destination.
    pub dst_factor: BlendFactor,
    /// How the two are combined.
    pub operation: BlendOperation,
}

impl BlendComponent {
    /// States one equation.
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
}

/// Color and alpha blending for one color target.
///
/// Two components rather than one, because the alpha equation is independent of
/// the color equation in every backend this layer targets, and collapsing them
/// would make the common "blend color but leave alpha alone" case inexpressible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlendState {
    /// The color equation.
    pub color: BlendComponent,
    /// The alpha equation.
    pub alpha: BlendComponent,
}

impl BlendState {
    /// States both equations.
    pub fn new(color: BlendComponent, alpha: BlendComponent) -> Self {
        Self { color, alpha }
    }
}

/// Which color channels a target writes.
///
/// A hand-rolled bitset rather than the `bitflags` crate, for the reason section
/// 11.1 gives for [`crate::api::resource::buffer::BufferUsage`]: the public
/// surface does not depend on a macro crate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ColorWriteMask(u8);

impl ColorWriteMask {
    /// Writes nothing.
    pub const NONE: Self = Self(0);
    /// Writes the red channel.
    pub const RED: Self = Self(1 << 0);
    /// Writes the green channel.
    pub const GREEN: Self = Self(1 << 1);
    /// Writes the blue channel.
    pub const BLUE: Self = Self(1 << 2);
    /// Writes the alpha channel.
    pub const ALPHA: Self = Self(1 << 3);
    /// Writes all four channels.
    pub const ALL: Self = Self(0x0f);

    /// Whether every bit set in `other` is set in `self`.
    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// The union of two masks.
    pub fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

/// How one color attachment is written.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct ColorTargetState {
    /// The format of the attachment this state is written to.
    pub format: TextureFormat,
    /// The blend equation, or `None` to overwrite.
    pub blend: Option<BlendState>,
    /// Which channels are written.
    pub write_mask: ColorWriteMask,
}

impl ColorTargetState {
    /// States a target format, overwriting every channel with no blending.
    ///
    /// `ALL` rather than `NONE`, because a target that writes nothing is the
    /// exceptional case: section 27.3 requires `NONE` exactly when the fragment
    /// stage has no output at that location, so a caller who wants that says so
    /// with [`Self::with_write_mask`].
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

/// One stencil operation.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StencilOperation {
    /// Keep the current value.
    Keep,
    /// Set to zero.
    Zero,
    /// Replace with the reference value.
    Replace,
    /// Bitwise inversion.
    Invert,
    /// Increment, clamped at the maximum.
    IncrementClamp,
    /// Decrement, clamped at zero.
    DecrementClamp,
    /// Increment, wrapping to zero.
    IncrementWrap,
    /// Decrement, wrapping to the maximum.
    DecrementWrap,
}

/// The stencil state of one face.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StencilFaceState {
    /// The comparison against the reference value.
    pub compare: CompareFunction,
    /// Applied when the stencil test fails.
    pub fail_op: StencilOperation,
    /// Applied when the stencil test passes and the depth test fails.
    pub depth_fail_op: StencilOperation,
    /// Applied when both tests pass.
    pub pass_op: StencilOperation,
}

impl StencilFaceState {
    /// States a comparison, leaving all three operations at `Keep`.
    ///
    /// `Keep` is the only default that cannot surprise: every other operation is a
    /// write to the stencil buffer that the caller did not ask for. Section 25.3
    /// fixes only the comparison parameter, and section 25.5's builders are the way
    /// to set the other three.
    pub fn new(compare: CompareFunction) -> Self {
        Self {
            compare,
            fail_op: StencilOperation::Keep,
            depth_fail_op: StencilOperation::Keep,
            pass_op: StencilOperation::Keep,
        }
    }
}

/// The complete stencil state: both faces plus the masks.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StencilState {
    /// The front-face state.
    pub front: StencilFaceState,
    /// The back-face state.
    pub back: StencilFaceState,
    /// The read mask applied to the stencil value before comparison.
    pub read_mask: u32,
    /// The write mask applied to every write.
    pub write_mask: u32,
}

impl StencilState {
    /// States both faces with both masks fully open.
    ///
    /// `0xffff_ffff` for both masks is the value every backend's default already
    /// is, so a caller who does not mention masks gets masking that changes
    /// nothing.
    pub fn new(front: StencilFaceState, back: StencilFaceState) -> Self {
        Self {
            front,
            back,
            read_mask: u32::MAX,
            write_mask: u32::MAX,
        }
    }
}

/// The depth test and write state.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DepthState {
    /// Whether passing fragments write depth.
    pub write_enabled: bool,
    /// The comparison against the depth buffer.
    pub compare: CompareFunction,
}

impl DepthState {
    /// States a comparison, with depth writes disabled.
    ///
    /// `false` because writing depth is the more surprising default of the two: a
    /// caller who wants it says so with [`Self::with_write_enabled`], and a caller
    /// who only wants a depth *test* gets exactly that.
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

/// The depth and stencil attachment state of a raster pipeline.
///
/// "Must be consistent with `FormatFacts.aspects()`" (section 25.3): a depth-only
/// format cannot carry a stencil state, and a combined format may enable either
/// independently. Like [`PrimitiveState`], this type carries no format facts, so
/// the consistency rule is checked by
/// `validate_raster_pipeline_descriptor`
/// rather than by a constructor.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct DepthStencilState {
    /// The format of the depth and/or stencil attachment.
    pub format: TextureFormat,
    /// The depth state, or `None` for no depth test or write.
    pub depth: Option<DepthState>,
    /// The stencil state, or `None` for no stencil test or write.
    pub stencil: Option<StencilState>,
}

impl DepthStencilState {
    /// Says a pipeline has a depth/stencil attachment of this format, and uses
    /// neither aspect of it.
    ///
    /// Both members start as `None` because they are independently optional, and
    /// because "the pipeline declares the attachment but tests nothing" is a legal
    /// and sometimes wanted pipeline — a render pass may have a depth attachment
    /// that this pipeline neither reads nor writes.
    pub fn new(format: TextureFormat) -> Self {
        Self {
            format,
            depth: None,
            stencil: None,
        }
    }

    /// Enables the depth test.
    pub fn with_depth(mut self, depth: DepthState) -> Self {
        self.depth = Some(depth);
        self
    }

    /// Enables the stencil test.
    pub fn with_stencil(mut self, stencil: StencilState) -> Self {
        self.stencil = Some(stencil);
        self
    }
}

/// Multisample state.
///
/// `count` must be consistent with every active render target's sample count, and
/// `alpha_to_coverage_enabled` is valid only when `count > 1` (section 25.4).
/// Both are checked where the targets are known — the raster pipeline and the
/// render pass — because this type has no target to compare against.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MultisampleState {
    /// The number of samples per pixel.
    pub count: u32,

    /// The portable sample mask aligns with the core mask width of Vulkan/WebGPU/D3D12.
    pub mask: u32,

    /// Whether alpha-to-coverage is enabled.
    pub alpha_to_coverage_enabled: bool,
}

impl MultisampleState {
    /// States a sample count with the portable mask and alpha-to-coverage off.
    ///
    /// The mask starts fully open (`u32::MAX`), which is the value the core mask of
    /// every backend is when nothing narrower was asked for.
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
