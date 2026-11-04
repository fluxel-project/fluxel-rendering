//! External preparation and controlled Forward-phase consumer.
use external_renderer::{ExternalClearPipeline, ExternalOpaquePipeline};
use fluxel_renderer::{
    BoundsVisibility, ForwardCpuPipeline, FramePipeline, FrameTopology, RenderBounds,
    RenderChange, RenderRecord, RenderScene, RenderTransform, RenderView,
};

/// Exercises two concrete compositions through the same public trait.
pub(crate) fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut scene = RenderScene::new()?;
    let mut changes = fluxel_renderer::RenderChangeSet::new();
    for (geometry, x) in [(0u32, 0.0), (1, 8.0), (2, -0.5)] {
        changes.push(RenderChange::Insert(RenderRecord {
            resources: (geometry, geometry % 2),
            transform: RenderTransform::identity(),
            bounds: RenderBounds::new([x, 0.0, 0.0], [x + 0.25, 0.25, 0.25])?,
        }));
    }
    changes.push(RenderChange::InsertView(RenderView::new(
        RenderTransform::identity(),
        [640, 360],
    )?));
    let receipt = scene.apply(&changes)?;
    let snapshot = scene.snapshot();
    let view = receipt.inserted_views()[0];
    let volume = RenderBounds::new([-1.0; 3], [1.0; 3])?;
    let builtin = ForwardCpuPipeline::new(BoundsVisibility);
    let external = ExternalOpaquePipeline::new();
    let clear = ExternalClearPipeline::new();
    let first = builtin.prepare(&snapshot, view, volume)?;
    let second = external.prepare(&snapshot, view, volume)?;
    let expected = vec![receipt.inserted()[0], receipt.inserted()[2]];
    if first.candidates() != expected || second.candidates() != expected {
        return Err("preparation differs from the fixed independent candidate oracle".into());
    }
    let clear_frame = clear.prepare(&snapshot, view, volume)?;
    if clear_frame.candidates() != expected
        || external.topology(&second)? != FrameTopology::OpaqueForward
        || clear.topology(&clear_frame)? != FrameTopology::ClearOnly
    {
        return Err("external frame policies did not retain their selected Forward products".into());
    }
    println!("S5 external pipelines select opaque and clear Forward products");
    Ok(())
}
