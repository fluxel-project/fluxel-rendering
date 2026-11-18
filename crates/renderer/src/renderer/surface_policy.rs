//! Renderer-owned realization of Material's static surface policy.
//!
//! Material owns the authoring facts. This module turns those facts and the
//! actual world-from-local transform into fixed-function pipeline state. The
//! transform is deliberately part of the result: a negative linear determinant
//! reverses the effective winding, so a cached pipeline cannot reuse the
//! opposite front-face state.

use fluxel_material::{SurfaceAlphaMode, SurfaceBlendMode, SurfaceSideOrientation};
use fluxel_rhi::{
    api::{
        BlendComponent, BlendFactor, BlendOperation, BlendState, CullMode, Error, FrontFace,
        PrimitiveState, Support, TextureDesc,
    },
    backend::ResourceBackend,
};

use crate::{ForwardDrawInput, FrameTopology, RenderTransform};

/// Hashable fixed-function facts selected for one material draw.
///
/// RHI's public descriptor enums intentionally do not expose `Hash`; this
/// private representation is the cache identity and converts to RHI state only
/// when a pipeline is created.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct RasterSurfaceState {
    front_face: FrontFaceState,
    culling: Culling,
    blend_mode: Option<SurfaceBlendMode>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum FrontFaceState {
    CounterClockwise,
    Clockwise,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum Culling {
    None,
    Front,
    Back,
    SeparateBack,
}

impl RasterSurfaceState {
    /// Applies the selected material/transform facts to triangle primitives.
    pub(crate) fn apply_to(self, primitive: &mut PrimitiveState) {
        primitive.front_face = match self.front_face {
            FrontFaceState::CounterClockwise => FrontFace::Ccw,
            FrontFaceState::Clockwise => FrontFace::Cw,
        };
        primitive.cull_mode = match self.culling {
            Culling::None => CullMode::None,
            Culling::Front => CullMode::Front,
            Culling::Back | Culling::SeparateBack => CullMode::Back,
        };
    }

    /// Returns the preliminary primitive with the opposite front-face winding.
    pub(crate) fn preliminary_primitive(
        self,
        mut primary: PrimitiveState,
    ) -> Option<PrimitiveState> {
        if self.culling != Culling::SeparateBack {
            return None;
        }
        primary.front_face = match primary.front_face {
            FrontFace::Ccw => FrontFace::Cw,
            FrontFace::Cw => FrontFace::Ccw,
        };
        Some(primary)
    }

    /// Babylon peeling disables ordinary culling before each mesh draw.
    pub(in crate::renderer) fn for_oit(mut self, separate_culling_pass: bool) -> Self {
        self.culling = if separate_culling_pass {
            Culling::SeparateBack
        } else {
            Culling::None
        };
        self
    }

    /// Returns Babylon's selected fixed-function blend factors.
    pub(crate) fn blend_state(self) -> Option<BlendState> {
        let mode = self.blend_mode?;
        Some(BlendState {
            color: BlendComponent {
                src_factor: if mode.uses_premultiplied_rgb() {
                    BlendFactor::One
                } else {
                    BlendFactor::SrcAlpha
                },
                dst_factor: BlendFactor::OneMinusSrcAlpha,
                operation: BlendOperation::Add,
            },
            alpha: BlendComponent {
                src_factor: BlendFactor::One,
                dst_factor: if mode == SurfaceBlendMode::PremultipliedPorterDuff {
                    BlendFactor::OneMinusSrcAlpha
                } else {
                    BlendFactor::One
                },
                operation: BlendOperation::Add,
            },
        })
    }

    /// Opaque and masked fragments can write the selected depth attachment.
    pub(crate) const fn depth_write_enabled(self) -> bool {
        self.blend_mode.is_none()
    }
}

/// Selects the actual front-face and cull state for one resolved draw.
///
/// Like Babylon Mesh::_getDrawSideOrientation, the material
/// override takes precedence over the mesh winding. The world determinant
/// then reverses that choice for a negative determinant.
/// The zero case remains the canonical orientation: it has no inverse for PBR
/// normal transforms and is already rejected by the PBR preparation boundary.
pub(crate) fn raster_surface_state(input: &ForwardDrawInput) -> RasterSurfaceState {
    let policy = input.prepared().material().surface_policy();
    let orientation = policy
        .side_orientation
        .unwrap_or(input.mesh_side_orientation());
    let clockwise = (orientation == SurfaceSideOrientation::Clockwise)
        ^ (linear_determinant(input.transform()) < 0.0);
    RasterSurfaceState {
        front_face: if clockwise {
            FrontFaceState::Clockwise
        } else {
            FrontFaceState::CounterClockwise
        },
        culling: match (policy.back_face_culling, input.view().mirrored) {
            (false, _) if policy.separate_culling_pass => Culling::SeparateBack,
            (false, _) => Culling::None,
            (true, true) => Culling::Front,
            (true, false) => Culling::Back,
        },
        blend_mode: matches!(
            input.effective_alpha_mode(),
            SurfaceAlphaMode::Blend | SurfaceAlphaMode::MaskBlend
        )
        .then_some(policy.blend_mode),
    }
}

/// Rejects alpha blending until its Graph phase owns ordering and composition.
///
/// A blend material must never be emitted through the opaque Forward or
/// Deferred G-buffer paths: those paths neither sort transparent draws nor
/// retain a destination composition contract.
pub(crate) fn require_opaque_or_mask(inputs: &[ForwardDrawInput]) -> Result<(), Error> {
    if inputs.iter().any(|input| {
        matches!(
            input.effective_alpha_mode(),
            SurfaceAlphaMode::Blend | SurfaceAlphaMode::MaskBlend
        )
    }) {
        return Err(Error::Unsupported(
            "transparent material requires the transparent Graph phase",
        ));
    }
    Ok(())
}

/// Confines blend modes to the Graph phase that provides their ordering and
/// destination read/write contract.
pub(crate) fn require_supported_surface_modes(
    inputs: &[ForwardDrawInput],
    topology: FrameTopology,
) -> Result<(), Error> {
    for input in inputs {
        let prepared = input.prepared();
        let requested = prepared.material().transmission_extensions().is_some();
        if requested != prepared.uses_transmission() {
            return Err(Error::Unsupported(
                "Forward transmission requires its selected refraction shader route",
            ));
        }
    }
    if topology == FrameTopology::SortedForward {
        return Ok(());
    }
    require_opaque_or_mask(inputs)
}

/// Admits the actual color attachment descriptor before its owner is allocated.
/// Sorted Forward has a destination-read/write blend contract, so support must
/// come from the selected device's descriptor facts rather than backend names.
pub(crate) fn require_blendable_target<B: ResourceBackend>(
    device: &fluxel_rhi::api::Device<B>,
    target: &TextureDesc,
    topology: FrameTopology,
) -> Result<(), Error> {
    if topology != FrameTopology::SortedForward {
        return Ok(());
    }
    match device.texture_support(target) {
        Support::Supported(facts) if facts.blendable => Ok(()),
        Support::Supported(_) => Err(Error::Unsupported("blendable Forward color target")),
        Support::Unsupported(reason) => Err(Error::Unsupported(reason)),
    }
}

fn linear_determinant(transform: RenderTransform) -> f64 {
    let matrix = transform.columns().map(|column| column.map(f64::from));
    matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[1][0] * (matrix[0][1] * matrix[2][2] - matrix[0][2] * matrix[2][1])
        + matrix[2][0] * (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1])
}
