//! Typed external position deformation for the bounded Deferred product.
//!
//! The Feature contributes WGSL source and per-frame parameters only. Renderer
//! owns the producer Graph pass, resource allocation, recording and completion.

use fluxel_renderer::{
    DeferredPositionDeformer, FeatureContext, FeatureError, FeatureHandle, ForwardResources,
    PositionDeformation, PreparedFrame, TypedFeatureSnapshot,
};
use fluxel_shader::SourceUnit;

/// Exact source retained by the native consumer before submission.
pub const BEND_SOURCE: &str = r#"
fn deform_position(position: vec3<f32>, parameters: vec4<f32>) -> vec3<f32> {
    return vec3<f32>(
        position.x + parameters.x * position.y * position.y + parameters.y,
        position.y,
        position.z + parameters.z * position.x + parameters.w,
    );
}
"#;

/// Application-owned typed parameters for one Deferred deformation submission.
#[derive(Clone, Debug)]
pub struct BendInput {
    /// `x` bend, `y` x offset, `z` z slope, and `w` z offset.
    pub parameters: [f32; 4],
}

/// Statically selected external bend implementation.
#[derive(Clone, Debug)]
pub struct BendFeature {
    handle: FeatureHandle,
}

impl BendFeature {
    /// Issues a scene-bound typed deformation handle.
    pub fn new(issuer: &fluxel_renderer::FeatureIssuer) -> Result<Self, FeatureError> {
        Ok(Self {
            handle: issuer.issue("external-bend-position")?,
        })
    }
}

impl DeferredPositionDeformer for BendFeature {
    type Input = BendInput;

    fn handle(&self) -> FeatureHandle {
        self.handle
    }

    fn prepare(
        &self,
        context: &FeatureContext<'_, ForwardResources>,
        input: &TypedFeatureSnapshot<Self::Input>,
        _frame: &PreparedFrame<'_, ForwardResources>,
    ) -> Result<PositionDeformation, FeatureError> {
        fluxel_renderer::validate_context(context, input, self.handle)?;
        Ok(PositionDeformation {
            source: SourceUnit::new("external-bend-position", BEND_SOURCE),
            parameters: input.data().parameters,
        })
    }
}
