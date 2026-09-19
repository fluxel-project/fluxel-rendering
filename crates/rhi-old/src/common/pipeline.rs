//! The fixed-function raster pipeline state a backend creates a pipeline from.
//!
//! # Why this vocabulary is closed
//!
//! The same rule as [`super::vertex`] and [`super::binding`]: the variants here are
//! exactly what a retained artifact declares, so a caller cannot describe a
//! pipeline no artifact uses. A new possibility arrives with the artifact that
//! needs it.
//!
//! # What is deliberately absent, and why
//!
//! - **Blending.** No retained recipe blends; every one replaces its colour
//!   target's contents. A `BlendState` whose default was "no blend" would be
//!   vocabulary for a recipe that does not exist, and *independent blend* is a
//!   named refusal in the plan's capability triage.
//! - **Stencil.** Every depth-stencil attachment is refused by name today and the
//!   plan defers stencil to its own slice, so a stencil state here would describe a
//!   pipeline the pass model cannot reach.
//! - **Polygon mode, conservative rasterization, unclipped depth, depth bounds and
//!   depth bias.** None is declared by a retained artifact, and polygon mode is a
//!   named refusal in the triage.
//! - **Multiview.** Its ledger row is closed on every profile; a multiview mask
//!   here would let a caller ask for something the ledger deliberately refuses.
//!
//! # The one refusal this type owns
//!
//! A raster pass admits **exactly one colour attachment, at index 0**, and that is
//! a preserved semantic of this RHI rather than a limit of any one API (plan
//! section 4). [`PipelineState::validate`] therefore refuses more than one colour
//! target before a backend sees the description, so a pipeline and the pass it is
//! used in cannot disagree about how many attachments exist.
//!
//! It also refuses a description with no attachment at all -- a pipeline that
//! renders nowhere -- and a zero sample count, which names no sample pattern.
//!
//! # Format facts are not decided here
//!
//! Whether `depth_stencil.format` carries depth, and whether a colour target's
//! format is a colour format, are capability questions a backend answers from its
//! own format table. [`TextureFormat`] is `#[non_exhaustive]`, so a classification
//! written here would need a wildcard arm that invents an answer for a format this
//! layer has not been taught. Each backend refuses the mismatch by name instead.
//!
//! # The comparison ordering is the sampler's
//!
//! [`DepthStencilState::depth_compare`] is [`CompareFunction`] from
//! [`super::sampler`]: the eight orderings are one vocabulary, and the sampler
//! module already owns them because a shadow sampler compares with the same eight.
//! A second eight-variant enum here would be a second spelling of one shape, which
//! is what this layer exists to remove.

use fluxel_rendergraph::TextureFormat;

use super::sampler::CompareFunction;

/// How vertices are assembled into primitives.
///
/// The five the retained recipes and the GL family both name. Fan and
/// adjacency topologies are absent because no artifact declares one.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum PrimitiveTopology {
    /// Each vertex is an independent point.
    Points,
    /// Each pair of vertices is an independent line.
    Lines,
    /// Consecutive vertices form a connected line.
    LineStrip,
    /// Each triple of vertices is an independent triangle.
    Triangles,
    /// Consecutive vertices form a connected triangle strip.
    TriangleStrip,
}

/// Which facing primitives are discarded.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CullMode {
    /// No primitive is discarded.
    None,
    /// Front-facing primitives are discarded.
    Front,
    /// Back-facing primitives are discarded.
    Back,
}

/// Which winding order is considered front-facing.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum FrontFace {
    /// Clockwise winding is front-facing.
    Clockwise,
    /// Counter-clockwise winding is front-facing.
    CounterClockwise,
}

/// How primitives are assembled and which of them survive rasterization.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct PrimitiveState {
    /// How vertices become primitives.
    pub topology: PrimitiveTopology,
    /// Which facing primitives are discarded.
    pub cull_mode: CullMode,
    /// Which winding order is front-facing.
    pub front_face: FrontFace,
}

impl PrimitiveState {
    /// The state every retained raster recipe declares.
    ///
    /// Triangle lists, no culling and counter-clockwise front faces. It is a named
    /// constructor rather than a `Default` impl so that the values a recipe
    /// depends on are visible at the definition site instead of inferred from a
    /// derive.
    pub(crate) const fn triangle_list() -> Self {
        Self {
            topology: PrimitiveTopology::Triangles,
            cull_mode: CullMode::None,
            front_face: FrontFace::CounterClockwise,
        }
    }
}

/// Whether and how depth is tested, for a pipeline that has a depth attachment.
///
/// The presence of this value *is* the depth test: a depth-stencil state that does
/// not test depth is not a state a retained recipe declares, so there is no
/// separate enable flag whose default could disagree with the pass.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct DepthStencilState {
    /// The format of the depth-stencil attachment this pipeline is built for.
    pub format: TextureFormat,
    /// Whether passing samples write depth.
    pub depth_write_enabled: bool,
    /// The ordering the depth test applies. Every variant is meaningful, and
    /// `Always` is a real comparison rather than the absent one.
    pub depth_compare: CompareFunction,
}

/// Which colour components a draw writes to its colour target.
///
/// A bitset rather than a struct of four booleans because a mask composes and the
/// GL family already models it as one integer, which is what makes the two
/// implementations converge without a translation step. The representation is
/// private and only the named constants and [`Self::union`] produce a value.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct ColorWriteMask(u8);

impl ColorWriteMask {
    /// The red component.
    pub(crate) const R: Self = Self(0b0001);
    /// The green component.
    pub(crate) const G: Self = Self(0b0010);
    /// The blue component.
    pub(crate) const B: Self = Self(0b0100);
    /// The alpha component.
    pub(crate) const A: Self = Self(0b1000);
    /// Every component: what an opaque colour target declares.
    pub(crate) const ALL: Self = Self(0b1111);

    /// Every component in both values is written.
    pub(crate) const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether every component in `other` is also written here.
    pub(crate) const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}

/// One colour target a raster pipeline writes.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct ColorTargetState {
    /// The format of the target, which the pipeline bakes in.
    pub format: TextureFormat,
    /// Which components the draw writes.
    pub write_mask: ColorWriteMask,
}

/// The complete fixed-function state one raster pipeline is created from.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PipelineState {
    /// How primitives are assembled and culled.
    pub primitive: PrimitiveState,
    /// The depth test, when the pipeline has a depth-stencil attachment.
    pub depth_stencil: Option<DepthStencilState>,
    /// Samples per texel the pipeline is built for.
    pub sample_count: u32,
    /// The colour attachments, in index order.
    pub color_targets: Vec<ColorTargetState>,
}

/// Why a description cannot describe a pipeline.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PipelineStateError {
    /// Neither a colour target nor a depth-stencil attachment was named.
    NoAttachments,
    /// More colour targets were named than a raster pass admits.
    MultipleColorTargets {
        /// How many were named.
        count: usize,
    },
    /// The description names zero samples per texel.
    ZeroSamples,
}

impl PipelineState {
    /// Rejects a description no retained pass could use.
    ///
    /// The whole check is local: it compares the description against the pass model
    /// and never against a device, because a limit or capability question belongs
    /// to the ledger. What it deliberately does not check is whether a named format
    /// is a colour or depth format; that is each backend's format table, and see
    /// the module docs for why.
    pub(crate) fn validate(&self) -> Result<(), PipelineStateError> {
        if self.color_targets.len() > 1 {
            return Err(PipelineStateError::MultipleColorTargets {
                count: self.color_targets.len(),
            });
        }
        if self.color_targets.is_empty() && self.depth_stencil.is_none() {
            return Err(PipelineStateError::NoAttachments);
        }
        if self.sample_count == 0 {
            return Err(PipelineStateError::ZeroSamples);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The state the retained colour-only recipes declare.
    fn colour_only() -> PipelineState {
        PipelineState {
            primitive: PrimitiveState::triangle_list(),
            depth_stencil: None,
            sample_count: 1,
            color_targets: vec![ColorTargetState {
                format: TextureFormat::Rgba8Unorm,
                write_mask: ColorWriteMask::ALL,
            }],
        }
    }

    #[test]
    fn the_retained_colour_only_description_is_accepted() {
        assert_eq!(colour_only().validate(), Ok(()));
        assert_eq!(
            colour_only().primitive,
            PrimitiveState {
                topology: PrimitiveTopology::Triangles,
                cull_mode: CullMode::None,
                front_face: FrontFace::CounterClockwise,
            }
        );
    }

    #[test]
    fn a_depth_sibling_description_is_accepted() {
        // The second realization every retained raster recipe has: the same colour
        // target plus the depth-stencil state, which the pass model refuses today
        // but the vocabulary must be able to express.
        let state = PipelineState {
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: CompareFunction::LessEqual,
            }),
            ..colour_only()
        };
        assert_eq!(state.validate(), Ok(()));
    }

    #[test]
    fn a_second_colour_target_is_refused() {
        // The preserved semantic this type owns: one colour attachment at index 0.
        let state = PipelineState {
            color_targets: vec![
                ColorTargetState {
                    format: TextureFormat::Rgba8Unorm,
                    write_mask: ColorWriteMask::ALL,
                },
                ColorTargetState {
                    format: TextureFormat::Rgba8Unorm,
                    write_mask: ColorWriteMask::ALL,
                },
            ],
            ..colour_only()
        };
        assert_eq!(
            state.validate(),
            Err(PipelineStateError::MultipleColorTargets { count: 2 })
        );
    }

    #[test]
    fn a_pipeline_with_no_attachment_at_all_is_refused() {
        let state = PipelineState {
            color_targets: Vec::new(),
            ..colour_only()
        };
        assert_eq!(state.validate(), Err(PipelineStateError::NoAttachments));
    }

    #[test]
    fn a_zero_sample_count_is_refused() {
        // Zero names no sample pattern rather than a one-sample one.
        let state = PipelineState {
            sample_count: 0,
            ..colour_only()
        };
        assert_eq!(state.validate(), Err(PipelineStateError::ZeroSamples));
    }

    #[test]
    fn the_write_mask_composes_without_gaining_a_component() {
        let rgb = ColorWriteMask::R
            .union(ColorWriteMask::G)
            .union(ColorWriteMask::B);
        assert!(rgb.contains(ColorWriteMask::R));
        assert!(rgb.contains(ColorWriteMask::G));
        assert!(rgb.contains(ColorWriteMask::B));
        assert!(!rgb.contains(ColorWriteMask::A));
        assert!(ColorWriteMask::ALL.contains(rgb));
        // The named full mask and the fold of the four components agree, so a
        // constant and the union cannot disagree about what is written.
        assert_eq!(rgb.union(ColorWriteMask::A), ColorWriteMask::ALL);
        assert_ne!(rgb, ColorWriteMask::ALL);
    }

    #[test]
    fn the_three_cull_modes_are_distinct_values() {
        assert_ne!(CullMode::None, CullMode::Front);
        assert_ne!(CullMode::Front, CullMode::Back);
        assert_ne!(CullMode::None, CullMode::Back);
    }
}
