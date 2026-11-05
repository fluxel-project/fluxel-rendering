//! Native evidence for external Material code-node variants.

use std::{error::Error, fs, path::Path};

use external_renderer::{ChannelMapVariant, ExternalDeferredPipeline};
use fluxel_renderer::{
    BoundsVisibility, ForwardCpuPipeline, ForwardResources, FramePipeline, RenderBounds,
    RenderChange, RenderChangeSet, RenderScene, RenderTransform, RenderView, Renderer,
    prepare_forward,
};
use fluxel_rhi::backend::{
    ExecutionBackend, MappingBackend, PipelineBackend, RasterBackend, ReadbackBackend,
    ResourceBackend, ShaderBackend,
};
use fluxel_shader::ShaderComposer;

use super::{
    asset_json,
    fixtures::{ALBEDO, Assets, EXTENT, flat_positions},
    oracle, verify_facts, write_facts,
};
use crate::block_on;

pub(super) fn run<B>(
    renderer: &mut Renderer<B>,
    assets: &Assets,
    forward_assets: &fluxel_renderer::ForwardAssets,
    visibility: RenderBounds,
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
    let first = expected_color(ChannelMapVariant::First);
    let second = expected_color(ChannelMapVariant::Second);
    if first == second {
        return Err("S7 CodeNode variants do not change the analytic output".into());
    }
    run_case(
        renderer,
        assets,
        forward_assets,
        visibility,
        output,
        ChannelMapVariant::First,
        first,
    )?;
    run_case(
        renderer,
        assets,
        forward_assets,
        visibility,
        output,
        ChannelMapVariant::Second,
        second,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_case<B>(
    renderer: &mut Renderer<B>,
    assets: &Assets,
    forward_assets: &fluxel_renderer::ForwardAssets,
    visibility: RenderBounds,
    output: &Path,
    variant: ChannelMapVariant,
    color: [u8; 4],
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
    let name = format!("code-node-{}", variant.label());
    let record = assets.code_record(variant);
    let mut scene = RenderScene::<ForwardResources>::new()?;
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::Insert(record.clone()));
    changes.push(RenderChange::InsertView(RenderView::new(
        RenderTransform::identity(),
        EXTENT,
    )?));
    let receipt = scene.apply(&changes)?;
    let view = receipt.inserted_views()[0];
    let snapshot = scene.snapshot();
    let deferred_pipeline = ExternalDeferredPipeline::new();
    let prepared = deferred_pipeline.prepare(&snapshot, view, visibility)?;
    let draws = prepare_forward(&prepared, forward_assets, &mut ShaderComposer::new())?;
    let [draw] = draws.as_slice() else {
        return Err("S7 CodeNode expected one selected draw".into());
    };
    let shader = draw.prepared().shader();
    let source = shader.source();
    let function = match variant {
        ChannelMapVariant::First => "external_channel_map_first",
        ChannelMapVariant::Second => "external_channel_map_second",
    };
    if !source.contains(function) {
        return Err("S7 prepared shader omitted the selected CodeNode source".into());
    }
    let contribution = draw.prepared().material().contribution();
    let source_units = contribution.source_units();
    if !source_units
        .iter()
        .any(|unit| unit.source.contains(function))
    {
        return Err("S7 material contribution omitted the selected CodeNode unit".into());
    }
    let (expected, mask) =
        oracle::image_for_color(flat_positions(), oracle::Route::Deferred, color);
    let (forward_expected, forward_mask) =
        oracle::image_for_color(flat_positions(), oracle::Route::Forward, color);
    write_inputs(
        output,
        &name,
        &snapshot,
        &record,
        variant,
        color,
        &expected,
        &forward_expected,
        &mask,
        source,
        shader.source_map(),
        draw.prepared().material().key().as_str(),
        &source_units,
    )?;
    if renderer.pending() != 0 {
        return Err("S7 CodeNode began with retained Renderer work".into());
    }

    let deferred = block_on(renderer.submit_deferred(
        &deferred_pipeline,
        snapshot.clone(),
        view,
        visibility,
        forward_assets,
    ))?;
    let pending_after_deferred = renderer.pending();
    fs::write(
        output.join(format!("s7-{name}-graph-owners.txt")),
        renderer
            .deferred_probe_diagnostic(&deferred)?
            .ok_or("S7 CodeNode accepted without Graph owners")?,
    )?;
    let facts = renderer
        .deferred_probe_facts()
        .ok_or("S7 CodeNode accepted without facts")?;
    verify_facts(facts)?;
    write_facts(
        output,
        &name,
        0,
        pending_after_deferred,
        None,
        [None, None],
        facts,
    )?;
    let actual = block_on(renderer.readback(&deferred))?;
    fs::write(output.join(format!("s7-{name}-deferred.rgba")), &actual)?;
    fs::write(
        output.join(format!("s7-{name}-deferred.ppm")),
        oracle::ppm(&actual)?,
    )?;
    let pending_after_deferred_readback = renderer.pending();
    write_facts(
        output,
        &name,
        0,
        pending_after_deferred,
        Some(pending_after_deferred_readback),
        [None, None],
        facts,
    )?;
    if pending_after_deferred != 1 || pending_after_deferred_readback != 0 {
        return Err("S7 CodeNode Deferred pending transition differs from 0 -> 1 -> 0".into());
    }
    oracle::verify(&actual, &expected, &mask, &name, oracle::Route::Deferred)?;

    let forward = block_on(renderer.submit_forward(
        &ForwardCpuPipeline::new(BoundsVisibility),
        snapshot,
        view,
        visibility,
        forward_assets,
    ))?;
    let pending_after_forward = renderer.pending();
    let forward_actual = block_on(renderer.readback(&forward))?;
    fs::write(
        output.join(format!("s7-{name}-forward.rgba")),
        &forward_actual,
    )?;
    fs::write(
        output.join(format!("s7-{name}-forward.ppm")),
        oracle::ppm(&forward_actual)?,
    )?;
    let pending_after_forward_readback = renderer.pending();
    write_facts(
        output,
        &name,
        0,
        pending_after_deferred,
        Some(pending_after_deferred_readback),
        [
            Some(pending_after_forward),
            Some(pending_after_forward_readback),
        ],
        facts,
    )?;
    if pending_after_forward != 1 || pending_after_forward_readback != 0 {
        return Err("S7 CodeNode Forward pending transition differs from 0 -> 1 -> 0".into());
    }
    oracle::verify(
        &forward_actual,
        &forward_expected,
        &forward_mask,
        &format!("{name} Forward"),
        oracle::Route::Forward,
    )
}

#[allow(clippy::too_many_arguments)]
fn write_inputs(
    output: &Path,
    name: &str,
    snapshot: &fluxel_renderer::FrameSnapshot<ForwardResources>,
    record: &fluxel_renderer::RenderRecord<ForwardResources>,
    variant: ChannelMapVariant,
    color: [u8; 4],
    expected: &[u8],
    forward_expected: &[u8],
    mask: &[u8],
    source: &str,
    source_map: &[fluxel_shader::SourceSpan],
    variant_key: &str,
    source_units: &[fluxel_shader::SourceUnit],
) -> Result<(), Box<dyn Error>> {
    fs::write(
        output.join(format!("s7-{name}-input.json")),
        format!(
            concat!(
                "{{\"schemaVersion\":1,\"sceneVersion\":{},\"variant\":\"{}\",",
                "\"factors\":{:?},\"sourceColor\":{:?},\"analyticOutput\":{:?},",
                "\"tint\":{:?},\"sameSnapshotForDeferredAndForward\":true,",
                "\"savedShaderRoute\":\"forward-shared-preparation\",\"assets\":{{\"geometry\":{},",
                "\"material\":{},\"texture\":{}}}}}\n"
            ),
            snapshot.version(),
            variant.label(),
            expected_factors(variant),
            ALBEDO,
            color,
            record.resources.tint,
            asset_json(record.resources.geometry),
            asset_json(record.resources.material),
            asset_json(record.resources.texture),
        ),
    )?;
    fs::write(output.join(format!("s7-{name}-expected.rgba")), expected)?;
    fs::write(
        output.join(format!("s7-{name}-expected.ppm")),
        oracle::ppm(expected)?,
    )?;
    fs::write(
        output.join(format!("s7-{name}-forward-expected.rgba")),
        forward_expected,
    )?;
    fs::write(
        output.join(format!("s7-{name}-forward-expected.ppm")),
        oracle::ppm(forward_expected)?,
    )?;
    fs::write(output.join(format!("s7-{name}-stable-mask.bin")), mask)?;
    fs::write(output.join(format!("s7-{name}-composed.wgsl")), source)?;
    fs::write(
        output.join(format!("s7-{name}-source-map.txt")),
        format!("{source_map:#?}\nsource units:\n{source_units:#?}\nvariant key: {variant_key}\n"),
    )?;
    Ok(())
}

fn expected_color(variant: ChannelMapVariant) -> [u8; 4] {
    match variant {
        // Independent oracle constants: ALBEDO is [144, 80, 32, 255].
        ChannelMapVariant::First => [16, 80, 144, 255],
        ChannelMapVariant::Second => [32, 40, 144, 255],
    }
}

fn expected_factors(variant: ChannelMapVariant) -> [f32; 3] {
    match variant {
        ChannelMapVariant::First => [0.5, 1.0, 1.0],
        ChannelMapVariant::Second => [1.0, 0.5, 1.0],
    }
}
