//! Typed CPU preparation and real Graph contribution for Forward features.

use super::{FeatureContext, FeatureError, FeatureHandle, FeatureIssuer, TypedFeatureSnapshot};
use crate::{
    ForwardDrawRequest, ForwardFeaturePass, ForwardResources, PipelineError, PreparedFrame,
    RenderBounds,
};

/// A statically composed feature that produces its own bounded Forward draws.
///
/// The input and preparation remain application-owned types. The renderer
/// validates their frame identity, resolves the emitted logical resources, and
/// owns every upload, Graph execution and completion. CPU input values are copied
/// into immutable resolved draws and frame data; this trait grants no backend
/// object or independent submission authority.
pub trait RenderFeature {
    /// Application-owned immutable batch input.
    type Input;
    /// Application-owned result retained through preparation and declaration.
    type Prepared;

    /// Returns this feature's scene-issued identity.
    fn handle(&self) -> FeatureHandle;

    /// Prepares a feature batch for the selected immutable frame and volume.
    fn prepare(
        &self,
        context: &FeatureContext<'_, ForwardResources>,
        input: &TypedFeatureSnapshot<Self::Input>,
        frame: &PreparedFrame<'_, ForwardResources>,
        visibility: RenderBounds,
    ) -> Result<Self::Prepared, FeatureError>;

    /// Borrows the feature's own logical draw requests, not scene record IDs.
    fn draws<'a>(&self, prepared: &'a Self::Prepared) -> &'a [ForwardDrawRequest];

    /// Declares this batch's actual accesses inside the real opaque Graph pass.
    ///
    /// The scope cannot name another feature's draws or declare output writes.
    /// All three read classes must be declared once, including an empty batch.
    fn contribute(
        &self,
        _prepared: &Self::Prepared,
        pass: &mut ForwardFeaturePass<'_, '_>,
    ) -> Result<(), PipelineError> {
        pass.read_vertex_streams()?;
        pass.read_frame_uniforms()?;
        pass.read_sampled_textures()
    }
}

/// One concrete feature and its immutable input, bound by a scene issuer.
///
/// Construction performs no GPU work. Submission validates the issuer, scene,
/// feature handle and snapshot version before invoking feature preparation.
pub struct FeatureBinding<'a, F: RenderFeature> {
    pub(crate) issuer: &'a FeatureIssuer,
    pub(crate) feature: &'a F,
    pub(crate) input: &'a TypedFeatureSnapshot<F::Input>,
}

impl<'a, F: RenderFeature> FeatureBinding<'a, F> {
    /// Binds one typed input to its concrete feature for a frame submission.
    pub const fn new(
        issuer: &'a FeatureIssuer,
        feature: &'a F,
        input: &'a TypedFeatureSnapshot<F::Input>,
    ) -> Self {
        Self {
            issuer,
            feature,
            input,
        }
    }
}
