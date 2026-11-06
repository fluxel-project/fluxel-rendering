//! Bounded, opt-in companion readbacks for the visible demo.

use std::{error::Error, fs, path::Path};

use external_renderer::ExternalDeferredPipeline;
use fluxel_renderer::{
    ForwardAssets, ForwardCpuPipeline, ForwardResources, FrameSnapshot, RenderBounds,
    RenderViewId, Renderer,
};
use fluxel_rhi::backend::{
    ExecutionBackend, MappingBackend, PipelineBackend, RasterBackend, ReadbackBackend,
    ResourceBackend, ShaderBackend,
};

use crate::{block_on, demo::DemoRoute};

pub(super) const LAST_FRAME: usize = 10;

pub(super) fn enabled() -> bool {
    std::env::var_os("FLUXEL_DEMO_EVIDENCE").is_some_and(|value| value == "1")
}

#[allow(clippy::too_many_arguments)]
pub(super) fn write_input(
    route: DemoRoute,
    snapshot: &FrameSnapshot<ForwardResources>,
    frame: usize,
    extent: [u32; 2],
    transform: [[f32; 4]; 4],
    tint: [f32; 4],
    output: &Path,
) -> Result<(), Box<dyn Error>> {
    fs::write(
        output.join(format!("demo-evidence-frame-{frame}-input.json")),
        format!(
            concat!(
                "{{\"schemaVersion\":1,\"frame\":{frame},\"route\":\"{}\",",
                "\"sceneVersion\":{},\"extent\":{:?},",
                "\"transformColumns\":{:?},\"tint\":{:?},",
                "\"companion\":\"same-snapshot-offscreen-readback\"}}\n"
            ),
            route_name(route),
            snapshot.version(),
            extent,
            transform,
            tint,
            frame = frame,
        ),
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn readback<B>(
    route: DemoRoute,
    renderer: &mut Renderer<B>,
    forward: &ForwardCpuPipeline<fluxel_renderer::BoundsVisibility>,
    deferred: &ExternalDeferredPipeline,
    snapshot: FrameSnapshot<ForwardResources>,
    view: RenderViewId,
    visibility: RenderBounds,
    assets: &ForwardAssets,
    frame: usize,
    output: &Path,
) -> Result<(), Box<dyn Error>>
where
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + RasterBackend
        + PipelineBackend
        + ShaderBackend,
{
    let submission = match route {
        DemoRoute::Forward => block_on(renderer.submit_forward(
            forward, snapshot, view, visibility, assets,
        ))?,
        DemoRoute::Deferred => block_on(renderer.submit_deferred(
            deferred, snapshot, view, visibility, assets,
        ))?,
    };
    let raw = block_on(renderer.readback(&submission))?;
    fs::write(
        output.join(format!("demo-evidence-frame-{frame}-readback.rgba")),
        raw,
    )?;
    Ok(())
}

fn route_name(route: DemoRoute) -> &'static str {
    match route {
        DemoRoute::Forward => "forward",
        DemoRoute::Deferred => "deferred",
    }
}
