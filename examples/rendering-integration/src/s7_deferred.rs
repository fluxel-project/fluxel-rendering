//! Native offscreen consumer for the provisional position-only Deferred probe.
//!
//! This architecture probe intentionally uses one built-in mesh and the shared
//! Forward asset schema. Each exact scene snapshot is submitted both to the
//! private Deferred probe and the public Forward path; no new public pipeline
//! or Feature contract is claimed here.

mod cache;
mod cache_manifest;
mod code_node;
mod deformation;
mod fixtures;
mod oracle;
mod pbr;
pub(crate) mod pbr_oracle;

use std::{error::Error, fs, path::Path};

use external_renderer::ExternalDeferredPipeline;
use fluxel_assets::AssetKind;
use fluxel_renderer::{
    BoundsVisibility, ForwardCpuPipeline, ForwardResources, FrameSnapshot, RenderBounds,
    RenderChange, RenderChangeSet, RenderRecord, RenderScene, RenderTransform, RenderView,
    Renderer, SceneAssetRef,
};
use fluxel_rhi::api::TextureFormat;
use fluxel_rhi::{
    api::Device,
    backend::{
        ExecutionBackend, MappingBackend, PipelineBackend, RasterBackend, ReadbackBackend,
        ResourceBackend, ShaderBackend,
    },
};

use self::fixtures::{ALBEDO, Assets, EXTENT};
use crate::block_on;

/// Runs flat and tilted position-only meshes through the same Deferred and
/// Forward snapshot/asset inputs, retaining their independent raw products.
pub(crate) fn run<B>(device: Device<B>, output: &Path) -> Result<(), Box<dyn Error>>
where
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + RasterBackend
        + PipelineBackend
        + ShaderBackend,
{
    fs::create_dir_all(output)?;
    let assets = Assets::new()?;
    assets.write(output)?;
    fs::write(
        output.join("s7-oracle-contract.json"),
        concat!(
            "{\"schemaVersion\":1,\"deferredInteriorRgbTolerance\":1,",
            "\"forwardRgbTolerance\":0,\"alphaTolerance\":0,\"backgroundTolerance\":0,",
            "\"edgeExclusionNdc\":0.035,\"expected\":\"independent analytic ideal\",",
            "\"reason\":\"Deferred stores and samples RGBA8 G-buffer before final RGBA8 quantization\"}\n"
        ),
    )?;
    let forward_assets = assets.forward();
    let mut scene = RenderScene::<ForwardResources>::new()?;
    let mut changes = RenderChangeSet::new();
    let flat = assets.record(false);
    changes.push(RenderChange::Insert(flat.clone()));
    changes.push(RenderChange::InsertView(RenderView::new(
        RenderTransform::identity(),
        EXTENT,
    )?));
    let receipt = scene.apply(&changes)?;
    let id = receipt.inserted()[0];
    let view = receipt.inserted_views()[0];
    let mut renderer = Renderer::new(device)?;
    let visibility = RenderBounds::new([-1.0; 3], [1.0; 3])?;

    run_case(
        &mut renderer,
        scene.snapshot(),
        view,
        visibility,
        &forward_assets,
        &flat,
        false,
        "flat",
        output,
    )?;

    let tilted = assets.record(true);
    let mut update = RenderChangeSet::new();
    update.push(RenderChange::Update(id, tilted.clone()));
    scene.apply(&update)?;
    run_case(
        &mut renderer,
        scene.snapshot(),
        view,
        visibility,
        &forward_assets,
        &tilted,
        true,
        "tilt",
        output,
    )?;
    cache::run(&mut renderer, &assets, visibility, output)?;
    cache_manifest::write(output)?;
    deformation::run(&mut renderer, &assets, &forward_assets, visibility, output)?;
    code_node::run(&mut renderer, &assets, &forward_assets, visibility, output)?;
    pbr::run(&mut renderer, visibility, output)?;
    write_manifest(output)
}

#[allow(clippy::too_many_arguments)]
fn run_case<B>(
    renderer: &mut Renderer<B>,
    snapshot: FrameSnapshot<ForwardResources>,
    view: fluxel_renderer::RenderViewId,
    visibility: RenderBounds,
    assets: &fluxel_renderer::ForwardAssets,
    record: &RenderRecord<ForwardResources>,
    tilted: bool,
    name: &str,
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
    let (expected, mask) = oracle::image(tilted, oracle::Route::Deferred);
    let (forward_expected, forward_mask) = oracle::image(tilted, oracle::Route::Forward);
    fs::write(
        output.join(format!("s7-{name}-input.json")),
        input_json(&snapshot, record, tilted),
    )?;
    fs::write(output.join(format!("s7-{name}-expected.rgba")), &expected)?;
    fs::write(
        output.join(format!("s7-{name}-expected.ppm")),
        oracle::ppm(&expected)?,
    )?;
    fs::write(
        output.join(format!("s7-{name}-forward-expected.rgba")),
        &forward_expected,
    )?;
    fs::write(
        output.join(format!("s7-{name}-forward-expected.ppm")),
        oracle::ppm(&forward_expected)?,
    )?;
    fs::write(output.join(format!("s7-{name}-stable-mask.bin")), &mask)?;
    let pending_before = renderer.pending();
    if pending_before != 0 {
        return Err("S7 Deferred began with pending Renderer work".into());
    }

    let deferred = block_on(renderer.submit_deferred(
        &ExternalDeferredPipeline::new(),
        snapshot.clone(),
        view,
        visibility,
        assets,
    ))?;
    let pending_after_deferred = renderer.pending();
    fs::write(
        output.join(format!("s7-{name}-graph-owners.txt")),
        renderer
            .deferred_probe_diagnostic(&deferred)?
            .ok_or("S7 Deferred accepted without Graph/owner diagnostic")?,
    )?;
    let facts = renderer
        .deferred_probe_facts()
        .ok_or("S7 Deferred accepted without facts")?;
    write_facts(
        output,
        name,
        pending_before,
        pending_after_deferred,
        None,
        [None, None],
        facts,
    )?;
    verify_facts(facts)?;
    let actual = block_on(renderer.readback(&deferred))?;
    fs::write(output.join(format!("s7-{name}-deferred.rgba")), &actual)?;
    fs::write(
        output.join(format!("s7-{name}-deferred.ppm")),
        oracle::ppm(&actual)?,
    )?;
    let pending_after_deferred_readback = renderer.pending();
    write_facts(
        output,
        name,
        pending_before,
        pending_after_deferred,
        Some(pending_after_deferred_readback),
        [None, None],
        facts,
    )?;
    if pending_after_deferred != 1 || pending_after_deferred_readback != 0 {
        return Err("S7 Deferred pending ownership did not transition 0 -> 1 -> 0".into());
    }
    oracle::verify(&actual, &expected, &mask, name, oracle::Route::Deferred)?;

    // This is intentionally the same immutable snapshot and asset collection
    // as the Deferred submission above, not a reconstructed comparison input.
    let forward = block_on(renderer.submit_forward(
        &ForwardCpuPipeline::new(BoundsVisibility),
        snapshot,
        view,
        visibility,
        assets,
    ))?;
    let pending_after_forward = renderer.pending();
    write_facts(
        output,
        name,
        pending_before,
        pending_after_deferred,
        Some(pending_after_deferred_readback),
        [Some(pending_after_forward), None],
        facts,
    )?;
    let forward_actual = block_on(renderer.readback(&forward))?;
    fs::write(
        output.join(format!("s7-{name}-forward.rgba")),
        &forward_actual,
    )?;
    fs::write(
        output.join(format!("s7-{name}-forward.ppm")),
        oracle::ppm(&forward_actual)?,
    )?;
    let pending_after = renderer.pending();
    write_facts(
        output,
        name,
        pending_before,
        pending_after_deferred,
        Some(pending_after_deferred_readback),
        [Some(pending_after_forward), Some(pending_after)],
        facts,
    )?;
    oracle::verify(
        &forward_actual,
        &forward_expected,
        &forward_mask,
        &format!("{name} Forward"),
        oracle::Route::Forward,
    )?;
    if pending_after_forward != 1 || pending_after != 0 {
        return Err("S7 Forward pending ownership did not transition 0 -> 1 -> 0".into());
    }
    Ok(())
}

fn input_json(
    snapshot: &FrameSnapshot<ForwardResources>,
    record: &RenderRecord<ForwardResources>,
    tilted: bool,
) -> String {
    format!(
        concat!(
            "{{\"schemaVersion\":1,\"sceneVersion\":{},\"case\":\"{}\",\"extent\":{:?},",
            "\"positionOnly\":true,\"sameSnapshotForDeferredAndForward\":true,",
            "\"bounds\":{{\"minimum\":{:?},\"maximum\":{:?}}},\"transformColumns\":{:?},",
            "\"tint\":{:?},\"textureRgba\":{:?},\"positionPlane\":{:?},",
            "\"assets\":{{\"geometry\":{},\"material\":{},\"texture\":{}}}}}\n"
        ),
        snapshot.version(),
        if tilted { "tilt" } else { "flat" },
        EXTENT,
        record.bounds.minimum(),
        record.bounds.maximum(),
        record.transform.columns(),
        record.resources.tint,
        ALBEDO,
        oracle::plane(tilted),
        asset_json(record.resources.geometry),
        asset_json(record.resources.material),
        asset_json(record.resources.texture),
    )
}

pub(super) fn verify_facts(
    facts: fluxel_renderer::DeferredProbeFacts,
) -> Result<(), Box<dyn Error>> {
    if facts.passes != 4
        || facts.buffer_bindings == 0
        || facts.texture_bindings == 0
        || facts.gbuffer_formats
            != [
                TextureFormat::Rgba8Unorm,
                TextureFormat::Rgba8Unorm,
                TextureFormat::Depth32Float,
            ]
        || facts.gbuffer_extents != [[EXTENT[0], EXTENT[1], 1]; 3]
    {
        return Err("S7 Deferred Graph facts differ from the four-pass G-buffer contract".into());
    }
    Ok(())
}

pub(super) fn write_facts(
    output: &Path,
    name: &str,
    pending_before: usize,
    pending_after_deferred: usize,
    pending_after_deferred_readback: Option<usize>,
    pending_forward: [Option<usize>; 2],
    facts: fluxel_renderer::DeferredProbeFacts,
) -> Result<(), Box<dyn Error>> {
    fs::write(
        output.join(format!("s7-{name}-facts.json")),
        format!(
            concat!(
                "{{\"schemaVersion\":1,\"pendingBefore\":{},\"pendingAfterDeferred\":{},",
                "\"pendingAfterDeferredReadback\":{},\"pendingAfterForward\":{},",
                "\"pendingAfterForwardReadback\":{},\"plan\":{},",
                "\"passes\":{},\"bufferBindings\":{},\"textureBindings\":{},\"gbufferFormats\":[{},{},{}],",
                "\"gbufferExtents\":[{:?},{:?},{:?}]}}\n"
            ),
            pending_before,
            pending_after_deferred,
            pending_after_deferred_readback
                .map_or_else(|| "null".to_owned(), |value| value.to_string()),
            pending_forward[0].map_or_else(|| "null".to_owned(), |value| value.to_string()),
            pending_forward[1].map_or_else(|| "null".to_owned(), |value| value.to_string()),
            crate::json_string(&format!("{:?}", facts.plan)),
            facts.passes,
            facts.buffer_bindings,
            facts.texture_bindings,
            crate::json_string(&format!("{:?}", facts.gbuffer_formats[0])),
            crate::json_string(&format!("{:?}", facts.gbuffer_formats[1])),
            crate::json_string(&format!("{:?}", facts.gbuffer_formats[2])),
            facts.gbuffer_extents[0],
            facts.gbuffer_extents[1],
            facts.gbuffer_extents[2],
        ),
    )?;
    Ok(())
}

fn asset_json<K: AssetKind>(asset: SceneAssetRef<K>) -> String {
    format!(
        "{{\"slot\":{},\"slotGeneration\":{},\"contentGeneration\":{}}}",
        asset.id().slot(),
        asset.id().slot_generation(),
        asset.generation().get(),
    )
}

fn write_manifest(output: &Path) -> Result<(), Box<dyn Error>> {
    fs::write(
        output.join("s7-deferred-workload.json"),
        concat!(
            "{\"schemaVersion\":1,\"workload\":\"s7-deferred-native-offscreen-probe\",",
            "\"sharedArtifacts\":[\"s7-flat-position.bin\",\"s7-flat-geometry.txt\",",
            "\"s7-tilt-position.bin\",\"s7-tilt-geometry.txt\",\"s7-material.txt\",",
            "\"s7-albedo.rgba\",\"s7-oracle-contract.json\",\"s7-bend-feature.wgsl\",",
            "\"s7-bend-facts.json\",\"s7-pbr-geometry.txt\",\"s7-pbr-material.txt\",",
            "\"s7-pbr-dielectric.rgba\",\"s7-pbr-metal.rgba\",\"s7-pbr-base-color.rgba\",\"s7-pbr-variants.json\",",
            "\"s7-deferred-cache-workload.json\"],",
            "\"cases\":[",
            "{\"name\":\"flat\",\"artifacts\":[\"s7-flat-input.json\",\"s7-flat-expected.rgba\",",
            "\"s7-flat-expected.ppm\",\"s7-flat-stable-mask.bin\",\"s7-flat-graph-owners.txt\",",
            "\"s7-flat-facts.json\",\"s7-flat-deferred.rgba\",\"s7-flat-deferred.ppm\",",
            "\"s7-flat-forward-expected.rgba\",\"s7-flat-forward-expected.ppm\",\"s7-flat-forward.rgba\",\"s7-flat-forward.ppm\"]},",
            "{\"name\":\"tilt\",\"artifacts\":[\"s7-tilt-input.json\",\"s7-tilt-expected.rgba\",",
            "\"s7-tilt-expected.ppm\",\"s7-tilt-stable-mask.bin\",\"s7-tilt-graph-owners.txt\",",
            "\"s7-tilt-facts.json\",\"s7-tilt-deferred.rgba\",\"s7-tilt-deferred.ppm\",",
            "\"s7-tilt-forward-expected.rgba\",\"s7-tilt-forward-expected.ppm\",\"s7-tilt-forward.rgba\",\"s7-tilt-forward.ppm\"]},",
            "{\"name\":\"bend-first\",\"artifacts\":[\"s7-bend-first-input.json\",\"s7-bend-first-expected.rgba\",",
            "\"s7-bend-first-expected.ppm\",\"s7-bend-first-stable-mask.bin\",\"s7-bend-first-graph-owners.txt\",",
            "\"s7-bend-first.rgba\",\"s7-bend-first.ppm\"]},",
            "{\"name\":\"bend-second\",\"artifacts\":[\"s7-bend-second-input.json\",\"s7-bend-second-expected.rgba\",",
            "\"s7-bend-second-expected.ppm\",\"s7-bend-second-stable-mask.bin\",\"s7-bend-second-graph-owners.txt\",",
            "\"s7-bend-second.rgba\",\"s7-bend-second.ppm\"]},",
            "{\"name\":\"code-node-first\",\"artifacts\":[\"s7-code-node-first-material.txt\",",
            "\"s7-code-node-first-input.json\",\"s7-code-node-first-expected.rgba\",\"s7-code-node-first-expected.ppm\",",
            "\"s7-code-node-first-forward-expected.rgba\",\"s7-code-node-first-forward-expected.ppm\",",
            "\"s7-code-node-first-stable-mask.bin\",\"s7-code-node-first-composed.wgsl\",\"s7-code-node-first-source-map.txt\",",
            "\"s7-code-node-first-graph-owners.txt\",\"s7-code-node-first-facts.json\",\"s7-code-node-first-deferred.rgba\",",
            "\"s7-code-node-first-deferred.ppm\",\"s7-code-node-first-forward.rgba\",\"s7-code-node-first-forward.ppm\"]},",
            "{\"name\":\"code-node-second\",\"artifacts\":[\"s7-code-node-second-material.txt\",",
            "\"s7-code-node-second-input.json\",\"s7-code-node-second-expected.rgba\",\"s7-code-node-second-expected.ppm\",",
            "\"s7-code-node-second-forward-expected.rgba\",\"s7-code-node-second-forward-expected.ppm\",",
            "\"s7-code-node-second-stable-mask.bin\",\"s7-code-node-second-composed.wgsl\",\"s7-code-node-second-source-map.txt\",",
            "\"s7-code-node-second-graph-owners.txt\",\"s7-code-node-second-facts.json\",\"s7-code-node-second-deferred.rgba\",",
            "\"s7-code-node-second-deferred.ppm\",\"s7-code-node-second-forward.rgba\",\"s7-code-node-second-forward.ppm\"]},",
            "{\"name\":\"pbr-dielectric\",\"artifacts\":[\"s7-pbr-dielectric-input.json\",\"s7-pbr-dielectric-expected.rgba\",\"s7-pbr-dielectric-expected.ppm\",\"s7-pbr-dielectric-forward-expected.rgba\",\"s7-pbr-dielectric-forward-expected.ppm\",\"s7-pbr-dielectric-stable-mask.bin\",\"s7-pbr-dielectric-composed.wgsl\",\"s7-pbr-dielectric-source-map.txt\",\"s7-pbr-dielectric-graph-owners.txt\",\"s7-pbr-dielectric-facts.json\",\"s7-pbr-dielectric-deferred.rgba\",\"s7-pbr-dielectric-deferred.ppm\",\"s7-pbr-dielectric-forward.rgba\",\"s7-pbr-dielectric-forward.ppm\"]},",
            "{\"name\":\"pbr-metal\",\"artifacts\":[\"s7-pbr-metal-input.json\",\"s7-pbr-metal-expected.rgba\",\"s7-pbr-metal-expected.ppm\",\"s7-pbr-metal-forward-expected.rgba\",\"s7-pbr-metal-forward-expected.ppm\",\"s7-pbr-metal-stable-mask.bin\",\"s7-pbr-metal-composed.wgsl\",\"s7-pbr-metal-source-map.txt\",\"s7-pbr-metal-graph-owners.txt\",\"s7-pbr-metal-facts.json\",\"s7-pbr-metal-deferred.rgba\",\"s7-pbr-metal-deferred.ppm\",\"s7-pbr-metal-forward.rgba\",\"s7-pbr-metal-forward.ppm\"]},",
            "{\"name\":\"pbr-rough\",\"artifacts\":[\"s7-pbr-rough-input.json\",\"s7-pbr-rough-expected.rgba\",\"s7-pbr-rough-expected.ppm\",\"s7-pbr-rough-forward-expected.rgba\",\"s7-pbr-rough-forward-expected.ppm\",\"s7-pbr-rough-stable-mask.bin\",\"s7-pbr-rough-composed.wgsl\",\"s7-pbr-rough-source-map.txt\",\"s7-pbr-rough-graph-owners.txt\",\"s7-pbr-rough-facts.json\",\"s7-pbr-rough-deferred.rgba\",\"s7-pbr-rough-deferred.ppm\",\"s7-pbr-rough-forward.rgba\",\"s7-pbr-rough-forward.ppm\"]},",
            "{\"name\":\"pbr-base-color\",\"artifacts\":[\"s7-pbr-base-color-input.json\",\"s7-pbr-base-color-expected.rgba\",\"s7-pbr-base-color-expected.ppm\",\"s7-pbr-base-color-forward-expected.rgba\",\"s7-pbr-base-color-forward-expected.ppm\",\"s7-pbr-base-color-stable-mask.bin\",\"s7-pbr-base-color-composed.wgsl\",\"s7-pbr-base-color-source-map.txt\",\"s7-pbr-base-color-graph-owners.txt\",\"s7-pbr-base-color-facts.json\",\"s7-pbr-base-color-deferred.rgba\",\"s7-pbr-base-color-deferred.ppm\",\"s7-pbr-base-color-forward.rgba\",\"s7-pbr-base-color-forward.ppm\"]},",
            "{\"name\":\"pbr-normal-transform\",\"artifacts\":[\"s7-pbr-normal-transform-input.json\",\"s7-pbr-normal-transform-expected.rgba\",\"s7-pbr-normal-transform-expected.ppm\",\"s7-pbr-normal-transform-forward-expected.rgba\",\"s7-pbr-normal-transform-forward-expected.ppm\",\"s7-pbr-normal-transform-stable-mask.bin\",\"s7-pbr-normal-transform-composed.wgsl\",\"s7-pbr-normal-transform-source-map.txt\",\"s7-pbr-normal-transform-graph-owners.txt\",\"s7-pbr-normal-transform-facts.json\",\"s7-pbr-normal-transform-deferred.rgba\",\"s7-pbr-normal-transform-deferred.ppm\",\"s7-pbr-normal-transform-forward.rgba\",\"s7-pbr-normal-transform-forward.ppm\"]}],",
            "\"subworkloads\":[\"s7-deferred-cache-workload.json\"],\"status\":\"architecture-probe-not-s7-closure\"}\n"
        ),
    )?;
    Ok(())
}
