//! Renderer-owned realization of Material's static surface policy.
//!
//! Material owns the authoring facts. This module turns those facts and the
//! actual world-from-local transform into fixed-function pipeline state. The
//! transform is deliberately part of the result: a negative linear determinant
//! reverses the effective winding, so a cached pipeline cannot reuse the
//! opposite front-face state.

use fluxel_material::SurfaceAlphaMode;
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
    cull_back_faces: bool,
    alpha_blending: bool,
    read_only_depth: bool,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum FrontFaceState {
    CounterClockwise,
    Clockwise,
}

impl RasterSurfaceState {
    /// Applies the selected material/transform facts to triangle primitives.
    pub(crate) fn apply_to(self, primitive: &mut PrimitiveState) {
        primitive.front_face = match self.front_face {
            FrontFaceState::CounterClockwise => FrontFace::Ccw,
            FrontFaceState::Clockwise => FrontFace::Cw,
        };
        primitive.cull_mode = if self.cull_back_faces {
            CullMode::Back
        } else {
            CullMode::None
        };
    }

    /// Returns Babylon's `ALPHA_COMBINE` fixed-function blend factors.
    ///
    /// Color uses straight alpha. Babylon separately adds source and
    /// destination alpha; it does not use the Porter-Duff alpha factors used
    /// by its distinct `ALPHA_LAYER_ACCUMULATE` mode.
    pub(crate) fn blend_state(self) -> Option<BlendState> {
        self.alpha_blending.then_some(BlendState {
            color: BlendComponent {
                src_factor: BlendFactor::SrcAlpha,
                dst_factor: BlendFactor::OneMinusSrcAlpha,
                operation: BlendOperation::Add,
            },
            alpha: BlendComponent {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::One,
                operation: BlendOperation::Add,
            },
        })
    }

    /// Blend draws and capture refraction receivers compare existing depth.
    /// The selected depth attachment stays available for later consumers.
    pub(crate) const fn depth_write_enabled(self) -> bool {
        !self.alpha_blending && !self.read_only_depth
    }
}

/// Selects the actual front-face and cull state for one resolved draw.
///
/// Babylon changes the effective orientation for a negative world determinant.
/// The zero case remains the canonical orientation: it has no inverse for PBR
/// normal transforms and is already rejected by the PBR preparation boundary.
pub(crate) fn raster_surface_state(input: &ForwardDrawInput) -> RasterSurfaceState {
    let policy = input.prepared().material().surface_policy();
    RasterSurfaceState {
        front_face: if linear_determinant(input.transform()) < 0.0 {
            FrontFaceState::Clockwise
        } else {
            FrontFaceState::CounterClockwise
        },
        cull_back_faces: policy.back_face_culling,
        read_only_depth: input.prepared().uses_scene_refraction(),
        alpha_blending: matches!(
            policy.alpha_mode,
            SurfaceAlphaMode::Blend | SurfaceAlphaMode::MaskBlend
        ),
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
            input.prepared().material().surface_policy().alpha_mode,
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
        if requested != prepared.uses_transmission()
            || (prepared.uses_transmission() && !prepared.uses_specular_ibl())
        {
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
