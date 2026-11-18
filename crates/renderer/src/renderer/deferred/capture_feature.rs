//! Actual typed producer read declarations for Deferred capture.

use super::*;
use crate::{
    FeatureHandle, ForwardDrawInput, ForwardDrawSource, ForwardPhaseContext, PipelineError,
    RenderFeature, graph::ForwardOitReadContext, pipeline::read_all,
};

#[derive(Default)]
pub(super) struct CaptureDrawGroups {
    meshes: Vec<usize>,
    features: [Vec<usize>; 2],
}

impl CaptureDrawGroups {
    pub(super) fn from_draws(
        draws: &[ForwardDrawInput],
        features: &[FeatureHandle],
    ) -> Result<Self, PipelineError> {
        let mut groups = Self::default();
        if !features.is_empty() {
            for (index, input) in draws.iter().enumerate() {
                match input.source() {
                    ForwardDrawSource::Scene(_) => groups.meshes.push(index),
                    ForwardDrawSource::Feature { feature, .. } => {
                        let slot = features
                            .iter()
                            .position(|handle| *handle == feature)
                            .and_then(|slot| groups.features.get_mut(slot))
                            .ok_or(PipelineError::InvalidDrawOrder)?;
                        slot.push(index);
                    }
                }
            }
        }
        Ok(groups)
    }
}

pub(super) trait CaptureProducerReads<B: ResourceBackend> {
    fn feature_handles(&self) -> &[FeatureHandle] {
        &[]
    }

    fn deferred(
        &self,
        context: &mut crate::DeferredPhaseContext<'_, '_>,
        groups: &CaptureDrawGroups,
    ) -> Result<(), PipelineError>;

    fn sorted(
        &self,
        context: &mut ForwardPhaseContext<'_, '_, '_, B>,
        _: &CaptureDrawGroups,
    ) -> Result<(), PipelineError> {
        context.sorted(|pass| {
            pass.depends_on_uploads()?;
            pass.read_vertex_streams()?;
            pass.read_frame_uniforms()?;
            pass.read_sampled_textures()?;
            pass.write_selected_depth()?;
            pass.read_write_main_color()
        })
    }

    fn overlay(
        &self,
        context: &mut ForwardOitReadContext<'_, '_>,
        _: &CaptureDrawGroups,
    ) -> Result<(), PipelineError> {
        context.all_reads()
    }
}

pub(super) struct SceneCaptureProducer<'a, 'frame, P> {
    pub(super) pipeline: &'a P,
    pub(super) selected: &'a crate::PreparedFrame<'frame, crate::ForwardResources>,
}

impl<B: ResourceBackend, P: FramePipeline<crate::ForwardResources>> CaptureProducerReads<B>
    for SceneCaptureProducer<'_, '_, P>
{
    fn deferred(
        &self,
        context: &mut crate::DeferredPhaseContext<'_, '_>,
        _: &CaptureDrawGroups,
    ) -> Result<(), PipelineError> {
        self.pipeline.contribute_deferred(self.selected, context)
    }
}

pub(super) struct FeatureCaptureProducer<'a, 'frame, P, F: RenderFeature> {
    pub(super) pipeline: &'a P,
    pub(super) selected: &'a crate::PreparedFrame<'frame, crate::ForwardResources>,
    pub(super) feature: PreparedCaptureFeature<'a, F>,
}

impl<B: ResourceBackend, P: FramePipeline<crate::ForwardResources>, F: RenderFeature>
    CaptureProducerReads<B> for FeatureCaptureProducer<'_, '_, P, F>
{
    fn feature_handles(&self) -> &[FeatureHandle] {
        std::slice::from_ref(&self.feature.handle)
    }

    fn deferred(
        &self,
        context: &mut crate::DeferredPhaseContext<'_, '_>,
        groups: &CaptureDrawGroups,
    ) -> Result<(), PipelineError> {
        declare_geometry(context, groups, |pass| {
            pass.scope(&groups.features[0], |scope| self.feature.contribute(scope))
        })
    }

    fn sorted(
        &self,
        context: &mut ForwardPhaseContext<'_, '_, '_, B>,
        groups: &CaptureDrawGroups,
    ) -> Result<(), PipelineError> {
        let mut composed = crate::ForwardCompositionContext::new(
            context,
            self.feature.feature,
            &self.feature.prepared,
            &groups.meshes,
            &groups.features[0],
        );
        self.pipeline
            .contribute_composed(self.selected, &mut composed)
    }

    fn overlay(
        &self,
        context: &mut ForwardOitReadContext<'_, '_>,
        groups: &CaptureDrawGroups,
    ) -> Result<(), PipelineError> {
        context.scope(&groups.meshes, read_all)?;
        context.scope(&groups.features[0], |scope| self.feature.contribute(scope))
    }
}

pub(super) struct PairCaptureProducer<'a, 'frame, P, F: RenderFeature, G: RenderFeature> {
    pub(super) pipeline: &'a P,
    pub(super) selected: &'a crate::PreparedFrame<'frame, crate::ForwardResources>,
    pub(super) handles: [FeatureHandle; 2],
    pub(super) first: PreparedCaptureFeature<'a, F>,
    pub(super) second: PreparedCaptureFeature<'a, G>,
}

impl<B, P, F, G> CaptureProducerReads<B> for PairCaptureProducer<'_, '_, P, F, G>
where
    B: ResourceBackend,
    P: FramePipeline<crate::ForwardResources>,
    F: RenderFeature,
    G: RenderFeature,
{
    fn feature_handles(&self) -> &[FeatureHandle] {
        &self.handles
    }

    fn deferred(
        &self,
        context: &mut crate::DeferredPhaseContext<'_, '_>,
        groups: &CaptureDrawGroups,
    ) -> Result<(), PipelineError> {
        declare_geometry(context, groups, |pass| {
            pass.scope(&groups.features[0], |scope| self.first.contribute(scope))?;
            pass.scope(&groups.features[1], |scope| self.second.contribute(scope))
        })
    }

    fn sorted(
        &self,
        context: &mut ForwardPhaseContext<'_, '_, '_, B>,
        groups: &CaptureDrawGroups,
    ) -> Result<(), PipelineError> {
        let mut composed = crate::ForwardPairCompositionContext::new(
            context,
            self.first.feature,
            &self.first.prepared,
            self.second.feature,
            &self.second.prepared,
            &groups.meshes,
            &groups.features[0],
            &groups.features[1],
        );
        self.pipeline
            .contribute_composed_pair(self.selected, &mut composed)
    }

    fn overlay(
        &self,
        context: &mut ForwardOitReadContext<'_, '_>,
        groups: &CaptureDrawGroups,
    ) -> Result<(), PipelineError> {
        context.scope(&groups.meshes, read_all)?;
        context.scope(&groups.features[0], |scope| self.first.contribute(scope))?;
        context.scope(&groups.features[1], |scope| self.second.contribute(scope))
    }
}

fn declare_geometry(
    context: &mut crate::DeferredPhaseContext<'_, '_>,
    groups: &CaptureDrawGroups,
    declare_features: impl FnOnce(
        &mut crate::DeferredGeometryPass<'_, '_, '_>,
    ) -> Result<(), PipelineError>,
) -> Result<(), PipelineError> {
    context.geometry(|pass| {
        pass.depends_on_uploads()?;
        pass.scope(&groups.meshes, read_all)?;
        declare_features(pass)?;
        pass.write_gbuffer()
    })?;
    context.lighting(|pass| {
        pass.read_gbuffer()?;
        pass.write_main_color()
    })
}

pub(super) struct PreparedCaptureFeature<'a, F: RenderFeature> {
    pub(super) feature: &'a F,
    pub(super) handle: FeatureHandle,
    pub(super) prepared: F::Prepared,
}

impl<F: RenderFeature> PreparedCaptureFeature<'_, F> {
    fn contribute(
        &self,
        scope: &mut crate::ForwardFeaturePass<'_, '_>,
    ) -> Result<(), PipelineError> {
        self.feature.contribute(&self.prepared, scope)
    }
}
