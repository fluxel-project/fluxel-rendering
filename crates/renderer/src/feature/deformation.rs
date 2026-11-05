//! Typed external position deformation for the bounded Deferred GPU product.

use super::{FeatureContext, FeatureError, FeatureHandle, FeatureIssuer, TypedFeatureSnapshot};
use crate::{DeferredDeformationPass, ForwardResources, PipelineError, PreparedFrame};

/// A position-transform function and the immutable parameters for one submission.
///
/// The source supplies `deform_position(vec3<f32>, vec4<f32>) -> vec3<f32>`.
/// Renderer composes and validates that function with its compute template;
/// it owns source/product buffers, dispatch, Graph accesses and completion.
#[derive(Clone, Debug)]
pub struct PositionDeformation {
    /// Named WGSL source unit implementing the position transform.
    pub source: fluxel_shader::SourceUnit,
    /// Finite per-submission values passed to the position transform.
    pub parameters: [f32; 4],
}

/// A concrete external Feature producing a GPU position stream for Deferred.
///
/// Application payloads remain typed and scene/issuer/version bound. This narrow
/// contract grants no Graph handle, buffer, encoder, allocator or submit access.
pub trait DeferredPositionDeformer {
    /// Application-owned immutable deformation parameters.
    type Input;

    /// Returns the scene-issued Feature identity.
    fn handle(&self) -> FeatureHandle;

    /// Resolves one frame's typed input to the checked position-function request.
    fn prepare(
        &self,
        context: &FeatureContext<'_, ForwardResources>,
        input: &TypedFeatureSnapshot<Self::Input>,
        frame: &PreparedFrame<'_, ForwardResources>,
    ) -> Result<PositionDeformation, FeatureError>;

    /// Declares this Feature's accesses inside the actual compute Graph pass.
    fn contribute(
        &self,
        pass: &mut DeferredDeformationPass<'_, '_, '_>,
    ) -> Result<(), PipelineError> {
        pass.read_source()?;
        pass.read_parameters()?;
        pass.write_vertices()
    }
}

/// One typed Feature and its immutable input, validated through a scene issuer.
pub struct DeformationBinding<'a, F: DeferredPositionDeformer> {
    pub(crate) issuer: &'a FeatureIssuer,
    pub(crate) feature: &'a F,
    pub(crate) input: &'a TypedFeatureSnapshot<F::Input>,
}

impl<'a, F: DeferredPositionDeformer> DeformationBinding<'a, F> {
    /// Binds a concrete position deformer without issuing device work.
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
