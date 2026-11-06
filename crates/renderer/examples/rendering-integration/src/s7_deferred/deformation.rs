//! Native evidence for two in-flight typed external position deformations.

use std::{error::Error, fs, path::Path};

use external_renderer::{BEND_SOURCE, BendFeature, BendInput, ExternalDeferredPipeline};
use fluxel_renderer::{
    CpuPhase, DeferredPositionDeformer, DeformationBinding, ForwardAssets, ForwardResources,
    FrameSnapshot, RenderBounds, RenderChange, RenderChangeSet, RenderRecord, RenderScene,
    RenderTransform, RenderView, Renderer,
};
use fluxel_rhi::backend::{
    ExecutionBackend, MappingBackend, PipelineBackend, RasterBackend, ReadbackBackend,
    ResourceBackend, ShaderBackend,
};

use super::{
    fixtures::{Assets, EXTENT},
    oracle,
};
use crate::block_on;

const FIRST: [f32; 4] = [0.45, -0.15, 0.12, 0.0];
const SECOND: [f32; 4] = [-0.30, 0.15, -0.10, 0.05];

pub(super) fn run<B>(
    renderer: &mut Renderer<B>,
    assets: &Assets,
    forward_assets: &ForwardAssets,
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
    fs::write(output.join("s7-bend-feature.wgsl"), BEND_SOURCE)?;
    let mut scene = RenderScene::<ForwardResources>::new()?;
    let issuer = scene.feature_issuer()?;
    let feature = BendFeature::new(&issuer)?;
    let mut storage = issuer.storage::<BendInput>();
    let mut changes = RenderChangeSet::new();
    let record = assets.record(false);
    changes.push(RenderChange::Insert(record.clone()));
    changes.push(RenderChange::InsertView(RenderView::new(
        RenderTransform::identity(),
        EXTENT,
    )?));
    let receipt = scene.apply(&changes)?;
    let view = receipt.inserted_views()[0];
    let snapshot = scene.snapshot();
    let (first_expected, first_mask) =
        oracle::image_for_positions(deformed(FIRST), oracle::Route::Deferred);
    let (second_expected, second_mask) =
        oracle::image_for_positions(deformed(SECOND), oracle::Route::Deferred);
    if first_expected == second_expected {
        return Err("S7 bend parameter replacements did not change the analytic result".into());
    }
    write_input(
        output,
        "first",
        &snapshot,
        &record,
        FIRST,
        &first_expected,
        &first_mask,
    )?;
    write_input(
        output,
        "second",
        &snapshot,
        &record,
        SECOND,
        &second_expected,
        &second_mask,
    )?;
    if renderer.pending() != 0 {
        return Err("S7 bend began with retained Renderer work".into());
    }

    let context = issuer.context(&snapshot, CpuPhase::Reference, feature.handle())?;
    let first_input = storage.publish(&context, BendInput { parameters: FIRST })?;
    let first = block_on(renderer.submit_deferred_with_deformation(
        &ExternalDeferredPipeline::new(),
        snapshot.clone(),
        view,
        visibility,
        forward_assets,
        DeformationBinding::new(&issuer, &feature, &first_input),
    ))?;
    if renderer.pending() != 1 {
        return Err("S7 first bend submission was not retained".into());
    }
    fs::write(
        output.join("s7-bend-first-graph-owners.txt"),
        renderer
            .deferred_probe_diagnostic(&first)?
            .ok_or("S7 first bend lacks Graph owners")?,
    )?;

    let second_input = storage.publish(&context, BendInput { parameters: SECOND })?;
    let second = block_on(renderer.submit_deferred_with_deformation(
        &ExternalDeferredPipeline::new(),
        snapshot.clone(),
        view,
        visibility,
        forward_assets,
        DeformationBinding::new(&issuer, &feature, &second_input),
    ))?;
    if renderer.pending() != 2 {
        return Err("S7 second bend submission did not coexist with first".into());
    }
    fs::write(
        output.join("s7-bend-second-graph-owners.txt"),
        renderer
            .deferred_probe_diagnostic(&second)?
            .ok_or("S7 second bend lacks Graph owners")?,
    )?;
    write_facts(output, renderer, &[0, 1, 2])?;

    let first_actual = block_on(renderer.readback(&first))?;
    fs::write(output.join("s7-bend-first.rgba"), &first_actual)?;
    fs::write(
        output.join("s7-bend-first.ppm"),
        oracle::ppm(&first_actual)?,
    )?;
    if renderer.pending() != 1 {
        return Err("S7 first bend readback did not preserve second owner".into());
    }
    write_facts(output, renderer, &[0, 1, 2, 1])?;
    oracle::verify(
        &first_actual,
        &first_expected,
        &first_mask,
        "bend first",
        oracle::Route::Deferred,
    )?;

    let second_actual = block_on(renderer.readback(&second))?;
    fs::write(output.join("s7-bend-second.rgba"), &second_actual)?;
    fs::write(
        output.join("s7-bend-second.ppm"),
        oracle::ppm(&second_actual)?,
    )?;
    if renderer.pending() != 0 {
        return Err("S7 bend owners survived both readbacks".into());
    }
    oracle::verify(
        &second_actual,
        &second_expected,
        &second_mask,
        "bend second",
        oracle::Route::Deferred,
    )?;
    write_facts(output, renderer, &[0, 1, 2, 1, 0])?;
    Ok(())
}

fn write_input(
    output: &Path,
    name: &str,
    snapshot: &FrameSnapshot<ForwardResources>,
    record: &RenderRecord<ForwardResources>,
    parameters: [f32; 4],
    expected: &[u8],
    mask: &[u8],
) -> Result<(), Box<dyn Error>> {
    fs::write(
        output.join(format!("s7-bend-{name}-input.json")),
        format!(
            concat!(
                "{{\"schemaVersion\":1,\"sceneVersion\":{},\"parameters\":{:?},",
                "\"source\":\"s7-bend-feature.wgsl\",\"assets\":{{\"geometry\":{},",
                "\"material\":{},\"texture\":{}}}}}\n"
            ),
            snapshot.version(),
            parameters,
            super::asset_json(record.resources.geometry),
            super::asset_json(record.resources.material),
            super::asset_json(record.resources.texture),
        ),
    )?;
    fs::write(
        output.join(format!("s7-bend-{name}-expected.rgba")),
        expected,
    )?;
    fs::write(
        output.join(format!("s7-bend-{name}-expected.ppm")),
        oracle::ppm(expected)?,
    )?;
    fs::write(output.join(format!("s7-bend-{name}-stable-mask.bin")), mask)?;
    Ok(())
}

fn write_facts<B>(
    output: &Path,
    renderer: &Renderer<B>,
    pending: &[usize],
) -> Result<(), Box<dyn Error>>
where
    B: ExecutionBackend + ResourceBackend + MappingBackend + ReadbackBackend,
{
    let facts = renderer
        .deferred_probe_facts()
        .ok_or("S7 bend lacks Deferred facts")?;
    if facts.passes != 5 {
        return Err("S7 bend producer Graph did not contain five passes".into());
    }
    fs::write(
        output.join("s7-bend-facts.json"),
        format!(
            "{{\"schemaVersion\":1,\"pending\":{:?},\"passes\":{},\"bufferBindings\":{},\"textureBindings\":{}}}\n",
            pending, facts.passes, facts.buffer_bindings, facts.texture_bindings,
        ),
    )?;
    Ok(())
}

fn deformed(parameters: [f32; 4]) -> [[f32; 3]; 3] {
    super::fixtures::flat_positions().map(|[x, y, z]| {
        [
            x + parameters[0] * y * y + parameters[1],
            y,
            z + parameters[2] * x + parameters[3],
        ]
    })
}
