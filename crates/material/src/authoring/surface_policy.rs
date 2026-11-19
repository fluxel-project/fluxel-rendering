//! Static raster and alpha facts for a material surface.

/// The alpha handling selected for a material surface.
///
/// This is a material fact. Consumers select the actual raster pipeline and
/// must still reject modes for which their pass has no correct composition.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum SurfaceAlphaMode {
    /// Write an opaque surface alpha of one.
    Opaque,
    /// Reject fragments below [`SurfacePolicy::alpha_cutoff`] and write alpha one.
    Mask,
    /// Preserve authored alpha for a transparency-capable consumer.
    Blend,
    /// Reject fragments below the cutoff and preserve accepted authored alpha.
    MaskBlend,
}

/// Color and alpha blend factors used for a blend-capable surface.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum SurfaceBlendMode {
    /// Straight RGB and additive alpha, matching Babylon's `ALPHA_COMBINE`.
    #[default]
    Combine,
    /// Adds source RGB weighted by source alpha and preserves destination alpha.
    /// Matches Babylon's `ALPHA_ADD` particle blend mode.
    Add,
    /// Premultiplied RGB and additive alpha.
    Premultiplied,
    /// Premultiplied RGB and source-over alpha.
    PremultipliedPorterDuff,
}

impl SurfaceBlendMode {
    /// Returns whether the fragment must multiply RGB by final alpha.
    pub const fn uses_premultiplied_rgb(self) -> bool {
        matches!(self, Self::Premultiplied | Self::PremultipliedPorterDuff)
    }
}

/// The winding that identifies the front side of a surface.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum SurfaceSideOrientation {
    /// Counterclockwise triangles are front-facing.
    #[default]
    CounterClockwise,
    /// Clockwise triangles are front-facing.
    Clockwise,
}

/// Static alpha, culling, and two-sided-lighting choices for one surface.
///
/// Defaults follow Babylon's opaque, back-face-culling policy. They are
/// deliberately independent choices: enabling two-sided lighting does not by
/// itself disable culling, and a blend mode does not promise a transparent pass.
#[derive(Clone, Copy, Debug)]
pub struct SurfacePolicy {
    /// Alpha mode selected by the author.
    pub alpha_mode: SurfaceAlphaMode,
    /// RGB association and blend factors for transparent composition.
    pub blend_mode: SurfaceBlendMode,
    /// The threshold used by mask modes.
    pub alpha_cutoff: f32,
    /// Whether a consumer should cull back-facing primitives.
    pub back_face_culling: bool,
    /// Optional front-side winding override for the mesh using this material.
    pub side_orientation: Option<SurfaceSideOrientation>,
    /// Whether a consumer should reverse the final shading normal on back faces.
    pub two_sided_lighting: bool,
    /// Draws each side with its own culling state when ordinary culling is disabled.
    pub separate_culling_pass: bool,
}

impl SurfacePolicy {
    /// Babylon-compatible default cutoff used when a mask mode is selected.
    pub const DEFAULT_ALPHA_CUTOFF: f32 = 0.4;

    /// Returns a policy that accepts both faces and reverses lighting normals
    /// for the back face.
    #[must_use]
    pub fn double_sided() -> Self {
        Self::default().with_double_sided(true)
    }

    /// Selects the blend factors and fragment RGB association.
    #[must_use]
    pub fn with_blend_mode(mut self, mode: SurfaceBlendMode) -> Self {
        self.blend_mode = mode;
        self
    }

    /// Selects the authored front-side winding for this material.
    #[must_use]
    pub fn with_side_orientation(mut self, orientation: SurfaceSideOrientation) -> Self {
        self.side_orientation = Some(orientation);
        self
    }

    /// Selects ordered back-side and front-side draws for a two-sided surface.
    #[must_use]
    pub fn with_separate_culling_pass(mut self, enabled: bool) -> Self {
        self.separate_culling_pass = enabled;
        self
    }

    /// Selects whether this surface is rendered and lit as double-sided.
    #[must_use]
    pub fn with_double_sided(mut self, enabled: bool) -> Self {
        self.back_face_culling = !enabled;
        self.two_sided_lighting = enabled;
        self
    }
}

impl Default for SurfacePolicy {
    fn default() -> Self {
        Self {
            alpha_mode: SurfaceAlphaMode::Opaque,
            blend_mode: SurfaceBlendMode::Combine,
            alpha_cutoff: Self::DEFAULT_ALPHA_CUTOFF,
            back_face_culling: true,
            side_orientation: None,
            two_sided_lighting: false,
            separate_culling_pass: false,
        }
    }
}

impl PartialEq for SurfacePolicy {
    fn eq(&self, other: &Self) -> bool {
        self.alpha_mode == other.alpha_mode
            && self.blend_mode == other.blend_mode
            && self.alpha_cutoff.to_bits() == other.alpha_cutoff.to_bits()
            && self.back_face_culling == other.back_face_culling
            && self.side_orientation == other.side_orientation
            && self.two_sided_lighting == other.two_sided_lighting
            && self.separate_culling_pass == other.separate_culling_pass
    }
}

impl Eq for SurfacePolicy {}

/// The face discarded when a material enables triangle culling.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum SurfaceCullFace {
    /// Cull back-facing triangles, as Babylon does by default.
    #[default]
    Back,
    /// Cull front-facing triangles.
    Front,
}
