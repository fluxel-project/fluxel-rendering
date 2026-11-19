//! Raster phase declaration for capture namespaces.

use super::*;

pub(in crate::graph::forward) fn declare_raster<B, F>(
    graph: &mut GraphBuilder,
    spec: &ForwardGraphSpec,
    topology: FrameTopology,
    upload: GraphPass,
    handles: &FrameHandles,
    contribute: F,
) -> std::result::Result<
    (
        GraphPass,
        crate::graph::definition::TextureHandle,
        Option<crate::graph::definition::TextureHandle>,
    ),
    ForwardPlanBuildError,
>
where
    B: ResourceBackend,
    F: FnOnce(&mut ForwardPhaseContext<'_, '_, '_, B>) -> std::result::Result<(), PipelineError>,
{
    let mut driver = ForwardPhaseDriver {
        graph: Some(graph),
        marker: std::marker::PhantomData,
        expected: topology,
        upload: Some(upload),
        streams: handles.streams.clone(),
        uniform: Some(handles.uniform),
        textures: handles.textures.clone(),
        environment: handles.environment,
        specular_radiance: handles.specular_radiance,
        specular_brdf_lut: handles.specular_brdf_lut,
        specular_view_uniform: handles.specular_view_uniform,
        specular_mip_levels: spec
            .specular_radiance
            .as_ref()
            .map(|owner| owner.desc.mip_levels),
        depth: handles.depth,
        target: handles.target,
        resolve_target: None,
        draws: spec.draws.clone(),
        indirect: Vec::new(),
        particles: Vec::new(),
        declared: false,
        output: None,
    };
    let mut context = ForwardPhaseContext::new(&mut driver);
    contribute(&mut context).map_err(ForwardPlanBuildError::Pipeline)?;
    driver
        .finish()
        .map_err(|_| ForwardPlanBuildError::Pipeline(PipelineError::InvalidPhaseContribution))
}
