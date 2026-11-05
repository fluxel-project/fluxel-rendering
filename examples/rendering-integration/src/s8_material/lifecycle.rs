//! Native, non-timed ownership and device-cache lifecycle evidence for S8.

use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
};

use external_renderer::ExternalOpaquePipeline;
use fluxel_renderer::{
    ForwardCacheStats, ForwardResources, RenderBounds, RenderScene, Renderer, RendererError,
};
use fluxel_rhi::{
    api::{Device, DeviceId},
    backend::{
        ExecutionBackend, MappingBackend, PipelineBackend, RasterBackend, ReadbackBackend,
        ResourceBackend, ShaderBackend,
    },
};

use super::{Config, DRAWS, EXTENT, Fixture, Scenario, TEXTURES, json};

const CASES: [&str; 5] = [
    "device-old-cold",
    "pending-old",
    "pending-replacement",
    "device-new-cold",
    "device-old-return",
];

pub(crate) fn run_lifecycle<B>(
    old_device: Device<B>,
    new_device: Device<B>,
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
    if old_device.id() == new_device.id() {
        return Err("S8 lifecycle requires distinct device identities".into());
    }
    fs::create_dir_all(output)?;
    let inputs = output.join("inputs");
    fs::create_dir_all(&inputs)?;
    let fixture = Fixture::new(DRAWS, TEXTURES)?;
    // All immutable fixture and replacement-generation bytes are persisted before GPU work.
    fixture.write_inputs(&inputs, &Config::new(Scenario::Replacement10Pct, false))?;
    let replacement_assets = fixture.assets_with_textures(fixture.replace_tenth()?);
    fixture.write_replacement_input(&inputs, &replacement_assets.textures)?;
    let mut scene = RenderScene::<ForwardResources>::new()?;
    let receipt = scene.apply(&fixture.initial_changes())?;
    let records = receipt.inserted().to_vec();
    let view = receipt.inserted_views()[0];
    let original = fixture.assets();
    let pipeline = ExternalOpaquePipeline::new();
    let bounds = RenderBounds::new([-1.; 3], [1.; 3])?;
    let original_snapshot = scene.snapshot();
    let mut old = Renderer::new(old_device.clone())?;
    let mut new = Renderer::new(new_device.clone())?;

    let old_cold_case = case_input(
        output,
        "device-old-cold",
        old_device.id(),
        "original",
        original_snapshot.version(),
        old.pending(),
    )?;
    let old_cold_before = old.forward_cache_stats();
    let old_cold_submission = crate::block_on(old.submit_forward(
        &pipeline,
        original_snapshot.clone(),
        view,
        bounds,
        &original,
    ))?;
    let old_cold_after_submit = old.pending();
    finish(
        &mut old,
        &old_cold_submission,
        &old_cold_case,
        old_cold_before,
        0,
        old_cold_after_submit,
    )?;

    let old_case = case_input(
        output,
        "pending-old",
        old_device.id(),
        "original",
        original_snapshot.version(),
        old.pending(),
    )?;
    let old_before = old.forward_cache_stats();
    let old_submission = crate::block_on(old.submit_forward(
        &pipeline,
        original_snapshot.clone(),
        view,
        bounds,
        &original,
    ))?;
    let after_old = old.pending();
    if after_old != 1 {
        return Err("old S8 owner did not remain pending".into());
    }

    scene.apply(&fixture.replacement_changes(&records, &replacement_assets.textures))?;
    let replacement_snapshot = scene.snapshot();
    let replacement_case = case_input(
        output,
        "pending-replacement",
        old_device.id(),
        "first-10pct-replaced",
        replacement_snapshot.version(),
        after_old,
    )?;
    let replacement_before = old.forward_cache_stats();
    let replacement_submission = crate::block_on(old.submit_forward(
        &pipeline,
        replacement_snapshot,
        view,
        bounds,
        &replacement_assets,
    ))?;
    let after_replacement = old.pending();
    if after_replacement != 2 {
        return Err("S8 replacement did not coexist with old owner".into());
    }

    let foreign_rejected = matches!(
        crate::block_on(new.readback(&old_submission)),
        Err(RendererError::ForeignSubmission)
    );
    let after_foreign = old.pending();
    if !foreign_rejected || after_foreign != after_replacement {
        return Err("foreign submission altered a pending old owner".into());
    }
    finish(
        &mut old,
        &old_submission,
        &old_case,
        old_before,
        0,
        after_old,
    )?;
    let after_old_readback = old.pending();
    finish(
        &mut old,
        &replacement_submission,
        &replacement_case,
        replacement_before,
        after_old,
        after_replacement,
    )?;
    let after_replacement_readback = old.pending();
    if after_old_readback != 1 || after_replacement_readback != 0 {
        return Err("S8 pending owner retirement order is incorrect".into());
    }

    let new_case = case_input(
        output,
        "device-new-cold",
        new_device.id(),
        "original",
        original_snapshot.version(),
        new.pending(),
    )?;
    let new_before = new.forward_cache_stats();
    let new_submission = crate::block_on(new.submit_forward(
        &pipeline,
        original_snapshot.clone(),
        view,
        bounds,
        &original,
    ))?;
    let new_after_submit = new.pending();
    finish(
        &mut new,
        &new_submission,
        &new_case,
        new_before,
        0,
        new_after_submit,
    )?;
    let new_after = new.forward_cache_stats();

    let return_case = case_input(
        output,
        "device-old-return",
        old_device.id(),
        "original",
        original_snapshot.version(),
        old.pending(),
    )?;
    let return_before = old.forward_cache_stats();
    let return_submission =
        crate::block_on(old.submit_forward(&pipeline, original_snapshot, view, bounds, &original))?;
    let return_after_submit = old.pending();
    finish(
        &mut old,
        &return_submission,
        &return_case,
        return_before,
        0,
        return_after_submit,
    )?;
    let return_after = old.forward_cache_stats();
    verify_cache(new_before, new_after, return_before, return_after)?;
    write_root(
        output,
        old_device.id(),
        new_device.id(),
        [
            after_old,
            after_replacement,
            after_foreign,
            after_old_readback,
            after_replacement_readback,
        ],
        foreign_rejected,
        new_before,
        new_after,
        return_before,
        return_after,
    )
}

fn case_input(
    output: &Path,
    name: &str,
    device: DeviceId,
    generation: &str,
    scene_version: u64,
    pending_before: usize,
) -> Result<PathBuf, Box<dyn Error>> {
    let path = output.join(name);
    fs::create_dir_all(&path)?;
    let sources =
        "[\"../inputs/s8-material-input.json\",\"../inputs/s8-material-replacement-input.json\"]";
    fs::write(
        path.join("lifecycle-input.json"),
        format!(
            "{{\"schemaVersion\":1,\"case\":{},\"deviceId\":{},\"scenario\":\"replacement-10pct\",\"generationSelection\":{},\"sceneVersion\":{},\"extent\":{:?},\"pendingBefore\":{},\"fixtureInputRoot\":\"../inputs\",\"sourceFiles\":{},\"submitSequence\":{}}}\n",
            json(name),
            json(&format!("{device:?}")),
            json(generation),
            scene_version,
            EXTENT,
            pending_before,
            sources,
            sequence(name)
        ),
    )?;
    Ok(path)
}

fn finish<B>(
    renderer: &mut Renderer<B>,
    submission: &fluxel_renderer::Submission,
    path: &Path,
    before: ForwardCacheStats,
    pending_before: usize,
    pending_after_submit: usize,
) -> Result<(), Box<dyn Error>>
where
    B: ExecutionBackend + ResourceBackend + MappingBackend + ReadbackBackend,
{
    let raw = crate::block_on(renderer.readback(submission))?;
    // Keep raw bytes before the derived cache/pending facts consumed by an oracle.
    fs::write(path.join("raw.rgba"), raw)?;
    let after = renderer.forward_cache_stats();
    fs::write(
        path.join("facts.json"),
        format!(
            "{{\"schemaVersion\":1,\"pendingBefore\":{},\"pendingAfterSubmit\":{},\"pendingAfterReadback\":{},\"cacheBefore\":{},\"cacheAfter\":{}}}\n",
            pending_before,
            pending_after_submit,
            renderer.pending(),
            cache(before),
            cache(after)
        ),
    )?;
    Ok(())
}

fn verify_cache(
    new_before: ForwardCacheStats,
    new_after: ForwardCacheStats,
    old_before: ForwardCacheStats,
    old_after: ForwardCacheStats,
) -> Result<(), Box<dyn Error>> {
    if new_after.geometry_hits != new_before.geometry_hits
        || new_after.texture_hits != new_before.texture_hits
        || new_after.accepted_texture_allocations <= new_before.accepted_texture_allocations
    {
        return Err("new device cold frame reused an old product".into());
    }
    if old_after.geometry_hits <= old_before.geometry_hits
        || old_after.texture_hits <= old_before.texture_hits
    {
        return Err("old device return missed completed cache products".into());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_root(
    output: &Path,
    old: DeviceId,
    new: DeviceId,
    pending: [usize; 5],
    foreign: bool,
    new_before: ForwardCacheStats,
    new_after: ForwardCacheStats,
    return_before: ForwardCacheStats,
    return_after: ForwardCacheStats,
) -> Result<(), Box<dyn Error>> {
    fs::write(
        output.join("pending-lifecycle.json"),
        format!(
            "{{\"schemaVersion\":1,\"pending\":{:?},\"foreignSubmissionRejected\":{},\"oldOwnerRetainedAcrossForeignRejection\":{}}}\n",
            pending,
            foreign,
            pending[2] == pending[1]
        ),
    )?;
    fs::write(
        output.join("device-cache.json"),
        format!(
            "{{\"schemaVersion\":1,\"oldDeviceId\":{},\"newDeviceId\":{},\"newCold\":{{\"before\":{},\"after\":{}}},\"oldReturn\":{{\"before\":{},\"after\":{}}}}}\n",
            json(&format!("{old:?}")),
            json(&format!("{new:?}")),
            cache(new_before),
            cache(new_after),
            cache(return_before),
            cache(return_after)
        ),
    )?;
    let cases = CASES.iter().map(|name| format!("{{\"name\":{},\"artifacts\":[\"{name}/lifecycle-input.json\",\"{name}/raw.rgba\",\"{name}/facts.json\"]}}", json(name))).collect::<Vec<_>>().join(",");
    let shared = [
        "inputs/s8-material-geometry.bin",
        "inputs/s8-material-variant-0.txt",
        "inputs/s8-material-variant-1.txt",
        "inputs/s8-material-variant-2.txt",
        "inputs/s8-material-variant-3.txt",
        "inputs/s8-material-textures-generation-1.rgba",
        "inputs/s8-material-textures-generation-2-replaced.rgba",
        "inputs/s8-material-input.json",
        "inputs/s8-material-replacement-input.json",
    ]
    .into_iter()
    .map(json)
    .collect::<Vec<_>>()
    .join(",");
    fs::write(
        output.join("s8-material-lifecycle-workload.json"),
        format!(
            "{{\"schemaVersion\":1,\"workload\":\"s8-material-lifecycle\",\"timingMetrics\":\"excluded\",\"sharedArtifacts\":[{shared}],\"rootArtifacts\":[\"pending-lifecycle.json\",\"device-cache.json\"],\"cases\":[{cases}]}}\n"
        ),
    )?;
    Ok(())
}

fn cache(s: ForwardCacheStats) -> String {
    format!(
        "{{\"acceptedBufferAllocations\":{},\"acceptedTextureAllocations\":{},\"residentCopyBytes\":{},\"geometryHits\":{},\"textureHits\":{},\"frameReuses\":{},\"opaqueDraws\":{}}}",
        s.accepted_buffer_allocations,
        s.accepted_texture_allocations,
        s.resident_copy_bytes,
        s.geometry_hits,
        s.texture_hits,
        s.frame_reuses,
        s.opaque_draws
    )
}
fn sequence(name: &str) -> u8 {
    match name {
        "device-old-cold" => 0,
        "pending-old" => 1,
        "pending-replacement" => 2,
        "device-new-cold" => 3,
        "device-old-return" => 4,
        _ => unreachable!(),
    }
}
