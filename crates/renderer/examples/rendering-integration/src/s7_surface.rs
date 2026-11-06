//! Native Surface consumer for the bounded Deferred Color product.
//!
//! Presentation is established by a client capture after Renderer completion.
//! The offscreen companion submits the same immutable snapshot and assets but
//! remains a separate RHI readback artifact.

use std::{error::Error, fs, path::Path};

use external_renderer::ExternalDeferredPipeline;
use fluxel_assets::AssetKind;
use fluxel_renderer::{
    ForwardResources, RenderBounds, RenderChange, RenderChangeSet, RenderRecord, RenderScene,
    RenderTransform, RenderView, Renderer,
};
use fluxel_rhi::{
    api::{Device, Surface},
    backend::{
        ExecutionBackend, MappingBackend, PipelineBackend, RasterBackend, ReadbackBackend,
        ResourceBackend, ShaderBackend, SurfaceBackend,
    },
};

mod deformation;
mod fixture;
mod oracle;
mod pbr;

use crate::{block_on, surface as surface_lifecycle};
use fixture::{Assets, COMPANION_EXTENT};

const INITIAL_EXTENT: [u32; 2] = [320, 180];
const RESIZED_EXTENT: [u32; 2] = [256, 144];
// The flat fixture's independent Deferred oracle is constant inside its plane:
// texture [144,80,32] times factor 0.9, then rounded to RGBA8.
const SURFACE_INTERIOR: [u8; 4] = [130, 72, 29, 255];

pub(crate) fn run_frames<B, Resize, Pump, Capture>(
    device: Device<B>,
    mut target: Surface<B>,
    output: &Path,
    resize: Resize,
    pump: Pump,
    capture: Capture,
) -> Result<(), Box<dyn Error>>
where
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + RasterBackend
        + PipelineBackend
        + ShaderBackend
        + SurfaceBackend,
    Resize: Fn([u32; 2], &str) -> Result<surface_lifecycle::ResizeFact, Box<dyn Error>>,
    Pump: Fn() -> Result<(), Box<dyn Error>>,
    Capture: Fn(usize, [u32; 2], [u8; 4]) -> Result<(), Box<dyn Error>>,
{
    fs::create_dir_all(output)?;
    fs::write(output.join("surface-events.jsonl"), b"")?;
    let assets = Assets::new()?;
    assets.write(output)?;
    let forward_assets = assets.forward();
    let record = assets.record();
    let mut scene = RenderScene::<ForwardResources>::new()?;
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::Insert(record.clone()));
    changes.push(RenderChange::InsertView(RenderView::new(
        RenderTransform::identity(),
        INITIAL_EXTENT,
    )?));
    changes.push(RenderChange::InsertView(RenderView::new(
        RenderTransform::identity(),
        COMPANION_EXTENT,
    )?));
    let receipt = scene.apply(&changes)?;
    let surface_view = receipt.inserted_views()[0];
    let companion_view = receipt.inserted_views()[1];
    let pipeline = ExternalDeferredPipeline::new();
    let volume = RenderBounds::new([-1.0; 3], [1.0; 3])?;
    let mut renderer = Renderer::new(device)?;

    let initial = surface_lifecycle::configure(&mut target, INITIAL_EXTENT, output)?;
    append_lifecycle(output, 0, "configure", None, initial.extent)?;
    submit_case(
        &mut renderer,
        &mut target,
        &pipeline,
        scene.snapshot(),
        surface_view,
        companion_view,
        volume,
        &forward_assets,
        &record,
        0,
        initial.extent,
        output,
        &capture,
    )?;
    surface_lifecycle::await_visual_release(output, 0, initial.extent, &pump)?;

    let observed = resize(RESIZED_EXTENT, "s7-deferred-surface-resize")?;
    append_lifecycle(output, 3, "resize", None, observed.extent)?;
    fs::write(
        output.join("s7-deferred-surface-reconfigure-input.json"),
        format!(
            "{{\"schemaVersion\":1,\"previousExtent\":{:?},\"observedClientExtent\":{:?},\"nextOperation\":\"configure\"}}\n",
            initial.extent, observed.extent,
        ),
    )?;
    let resized = surface_lifecycle::configure(&mut target, observed.extent, output)?;
    append_lifecycle(output, 4, "configure", None, resized.extent)?;
    let mut update = RenderChangeSet::new();
    update.push(RenderChange::UpdateView(
        surface_view,
        RenderView::new(RenderTransform::identity(), resized.extent)?,
    ));
    scene.apply(&update)?;
    submit_case(
        &mut renderer,
        &mut target,
        &pipeline,
        scene.snapshot(),
        surface_view,
        companion_view,
        volume,
        &forward_assets,
        &record,
        1,
        resized.extent,
        output,
        &capture,
    )?;
    surface_lifecycle::await_visual_release(output, 1, resized.extent, &pump)?;
    if renderer.pending() != 0 {
        return Err("S7 Deferred Surface retained work after both completions".into());
    }
    pbr::run(
        &mut renderer,
        &mut target,
        resized.extent,
        volume,
        output,
        &capture,
        &pump,
    )?;
    deformation::run(
        &mut renderer,
        &mut target,
        resized.extent,
        volume,
        output,
        &capture,
        &pump,
        4,
    )?;
    write_workload(output)
}

#[allow(clippy::too_many_arguments)]
fn submit_case<B, Capture>(
    renderer: &mut Renderer<B>,
    target: &mut Surface<B>,
    pipeline: &ExternalDeferredPipeline,
    snapshot: fluxel_renderer::FrameSnapshot<ForwardResources>,
    surface_view: fluxel_renderer::RenderViewId,
    companion_view: fluxel_renderer::RenderViewId,
    volume: RenderBounds,
    assets: &fluxel_renderer::ForwardAssets,
    record: &RenderRecord<ForwardResources>,
    frame: usize,
    surface_extent: [u32; 2],
    output: &Path,
    capture: &Capture,
) -> Result<(), Box<dyn Error>>
where
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + RasterBackend
        + PipelineBackend
        + ShaderBackend
        + SurfaceBackend,
    Capture: Fn(usize, [u32; 2], [u8; 4]) -> Result<(), Box<dyn Error>>,
{
    let (expected, mask) = oracle::image();
    write_inputs(
        output,
        frame,
        &snapshot,
        record,
        surface_extent,
        &expected,
        &mask,
    )?;
    let image = block_on(target.acquire())
        .map_err(|error| format!("S7 Deferred Surface frame {frame} acquire: {error:?}"))?;
    if image.extent() != surface_extent {
        return Err(format!(
            "S7 Deferred Surface lease {frame} extent did not match configuration"
        )
        .into());
    }
    append_lifecycle(
        output,
        if frame == 0 { 1 } else { 5 },
        "acquire",
        Some(frame),
        image.extent(),
    )?;
    let surface = block_on(renderer.submit_deferred_surface(
        pipeline,
        snapshot.clone(),
        surface_view,
        volume,
        assets,
        image,
    ))?;
    let facts = renderer
        .deferred_probe_facts()
        .ok_or("S7 Deferred Surface accepted without current Graph facts")?;
    if facts.passes != 3 {
        return Err(
            "S7 Deferred Surface must declare upload, geometry and lighting Graph passes".into(),
        );
    }
    block_on(renderer.complete(&surface))?;
    append_lifecycle(
        output,
        if frame == 0 { 2 } else { 6 },
        "complete",
        Some(frame),
        surface_extent,
    )?;
    if renderer.pending() != 0 {
        return Err(
            format!("S7 Deferred Surface frame {frame} remained pending after complete").into(),
        );
    }
    capture(frame, surface_extent, SURFACE_INTERIOR)?;

    let companion =
        block_on(renderer.submit_deferred(pipeline, snapshot, companion_view, volume, assets))?;
    let raw = block_on(renderer.readback(&companion))?;
    fs::write(
        output.join(format!("s7-deferred-surface-frame-{frame}-readback.rgba")),
        &raw,
    )?;
    fs::write(
        output.join(format!("s7-deferred-surface-frame-{frame}-readback.ppm")),
        oracle::ppm(&raw)?,
    )?;
    oracle::verify(
        &raw,
        &expected,
        &mask,
        &format!("surface companion {frame}"),
    )?;
    fs::write(
        output.join(format!("s7-deferred-surface-frame-{frame}-facts.json")),
        format!(
            "{{\"schemaVersion\":1,\"graphPasses\":{},\"pendingAfterComplete\":0,\"surfaceCapture\":\"surface-client-frame-{}.bmp\",\"companionRaw\":\"s7-deferred-surface-frame-{}-readback.rgba\"}}\n",
            facts.passes, frame, frame,
        ),
    )?;
    Ok(())
}

fn write_inputs(
    output: &Path,
    frame: usize,
    snapshot: &fluxel_renderer::FrameSnapshot<ForwardResources>,
    record: &RenderRecord<ForwardResources>,
    surface_extent: [u32; 2],
    expected: &[u8],
    mask: &[u8],
) -> Result<(), Box<dyn Error>> {
    fs::write(
        output.join(format!("s7-deferred-surface-frame-{frame}-input.json")),
        format!(
            concat!(
                "{{\"schemaVersion\":1,\"sceneVersion\":{},\"surfaceExtent\":{:?},",
                "\"companionExtent\":{:?},\"route\":\"deferred-color\",",
                "\"triangleRgbTolerance\":1,\"backgroundRgbTolerance\":0,",
                "\"sharedInputFiles\":[\"s7-surface-position.bin\",\"s7-surface-geometry.txt\",\"s7-surface-material.txt\",\"s7-surface-albedo.rgba\"],",
                "\"expectedSurfaceInteriorRgba\":{:?},\"record\":{{\"tint\":{:?},\"bounds\":{{\"minimum\":{:?},\"maximum\":{:?}}},",
                "\"transformColumns\":{:?},\"assets\":{{\"geometry\":{},\"material\":{},\"texture\":{}}}}}}}\n"
            ),
            snapshot.version(),
            surface_extent,
            COMPANION_EXTENT,
            SURFACE_INTERIOR,
            record.resources.tint,
            record.bounds.minimum(),
            record.bounds.maximum(),
            record.transform.columns(),
            asset_json(record.resources.geometry),
            asset_json(record.resources.material),
            asset_json(record.resources.texture),
        ),
    )?;
    fs::write(
        output.join(format!("s7-deferred-surface-frame-{frame}-expected.rgba")),
        expected,
    )?;
    fs::write(
        output.join(format!("s7-deferred-surface-frame-{frame}-expected.ppm")),
        oracle::ppm(expected)?,
    )?;
    fs::write(output.join("s7-deferred-surface-stable-mask.bin"), mask)?;
    Ok(())
}

fn asset_json<K: AssetKind>(reference: fluxel_renderer::SceneAssetRef<K>) -> String {
    format!(
        "{{\"slot\":{},\"slotGeneration\":{},\"contentGeneration\":{}}}",
        reference.id().slot(),
        reference.id().slot_generation(),
        reference.generation().get(),
    )
}

fn append_lifecycle(
    output: &Path,
    sequence: u8,
    event: &str,
    frame: Option<usize>,
    extent: [u32; 2],
) -> Result<(), Box<dyn Error>> {
    use std::io::Write;
    let mut log = fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(output.join("s7-deferred-surface-lifecycle.jsonl"))?;
    writeln!(
        log,
        "{{\"sequence\":{sequence},\"event\":\"{event}\",\"frame\":{},\"extent\":{:?}}}",
        frame.map_or_else(|| "null".to_owned(), |value| value.to_string()),
        extent,
    )?;
    Ok(())
}

fn write_workload(output: &Path) -> Result<(), Box<dyn Error>> {
    write_deformation_workload(output)?;
    fs::write(
        output.join("s7-deferred-surface-workload.json"),
        format!(
            concat!(
                "{{\"schemaVersion\":1,\"workload\":\"s7-deferred-color-surface-resize\",",
                "\"initialExtent\":{:?},\"resizedExtent\":{:?},",
                "\"surfaceProof\":\"native-client-capture\",",
                "\"companionProof\":\"same-snapshot-offscreen-readback\",",
                "\"completionBeforeReconfigure\":true,\"currentCallbackDeclaredGraph\":true,",
                "\"sharedArtifacts\":[\"s7-surface-position.bin\",\"s7-surface-geometry.txt\",\"s7-surface-material.txt\",\"s7-surface-albedo.rgba\",\"s7-deferred-surface-stable-mask.bin\",\"s7-deferred-surface-lifecycle.jsonl\",\"surface-events.jsonl\",\"surface-configuration.jsonl\"],",
                "\"subworkloads\":[\"s7-deferred-surface-pbr-workload.json\",\"s7-deferred-surface-deformation-workload.json\"],",
                "\"cases\":[",
                "{{\"name\":\"initial\",\"artifacts\":[\"s7-deferred-surface-frame-0-input.json\",\"s7-deferred-surface-frame-0-expected.rgba\",\"s7-deferred-surface-frame-0-expected.ppm\",\"s7-deferred-surface-frame-0-readback.rgba\",\"s7-deferred-surface-frame-0-readback.ppm\",\"s7-deferred-surface-frame-0-facts.json\",\"surface-client-frame-0.bmp\",\"surface-client-frame-0.capture.json\",\"surface-client-frame-0.verdict.json\"]}},",
                "{{\"name\":\"resized\",\"artifacts\":[\"s7-deferred-surface-reconfigure-input.json\",\"s7-deferred-surface-frame-1-input.json\",\"s7-deferred-surface-frame-1-expected.rgba\",\"s7-deferred-surface-frame-1-expected.ppm\",\"s7-deferred-surface-frame-1-readback.rgba\",\"s7-deferred-surface-frame-1-readback.ppm\",\"s7-deferred-surface-frame-1-facts.json\",\"surface-client-frame-1.bmp\",\"surface-client-frame-1.capture.json\",\"surface-client-frame-1.verdict.json\"]}}]}}\n"
            ),
            INITIAL_EXTENT, RESIZED_EXTENT,
        ),
    )?;
    Ok(())
}

fn write_deformation_workload(output: &Path) -> Result<(), Box<dyn Error>> {
    fs::write(
        output.join("s7-deferred-surface-deformation-workload.json"),
        "{\"schemaVersion\":1,\"workload\":\"s7-deferred-surface-deformation\",\"sharedArtifacts\":[\"s7-deferred-surface-deformation-feature.wgsl\",\"s7-deferred-surface-deformation-positions.bin\",\"s7-deferred-surface-deformation-texture.rgba\"],\"cases\":[{\"name\":\"first\",\"artifacts\":[\"s7-deferred-surface-deformation-first-input.json\",\"s7-deferred-surface-deformation-first-expected.rgba\",\"s7-deferred-surface-deformation-first-expected.ppm\",\"s7-deferred-surface-deformation-first-stable-mask.bin\",\"s7-deferred-surface-deformation-first-composed.wgsl\",\"s7-deferred-surface-deformation-first-source-map.txt\",\"s7-deferred-surface-deformation-first-owners.txt\",\"s7-deferred-surface-deformation-first-readback.rgba\",\"s7-deferred-surface-deformation-first-readback.ppm\",\"s7-deferred-surface-deformation-first-facts.json\",\"surface-client-frame-4.bmp\",\"surface-client-frame-4.capture.json\",\"surface-client-frame-4.verdict.json\"]},{\"name\":\"second\",\"artifacts\":[\"s7-deferred-surface-deformation-second-input.json\",\"s7-deferred-surface-deformation-second-expected.rgba\",\"s7-deferred-surface-deformation-second-expected.ppm\",\"s7-deferred-surface-deformation-second-stable-mask.bin\",\"s7-deferred-surface-deformation-second-composed.wgsl\",\"s7-deferred-surface-deformation-second-source-map.txt\",\"s7-deferred-surface-deformation-second-owners.txt\",\"s7-deferred-surface-deformation-second-readback.rgba\",\"s7-deferred-surface-deformation-second-readback.ppm\",\"s7-deferred-surface-deformation-second-facts.json\",\"surface-client-frame-5.bmp\",\"surface-client-frame-5.capture.json\",\"surface-client-frame-5.verdict.json\"]}],\"surfaceProof\":\"native-client-capture\",\"companionProof\":\"same-snapshot-offscreen-readback\"}\n",
    )?;
    Ok(())
}

#[cfg(all(windows, feature = "dx12"))]
pub(crate) fn run_dx12(
    device: Device<fluxel_rhi::dx12::Dx12>,
    output: &Path,
) -> Result<(), Box<dyn Error>> {
    use fluxel_host::{Window, WindowConfig};
    use raw_window_handle::HasWindowHandle;
    let window = Window::new(
        WindowConfig::new(
            "Fluxel S7 Deferred Surface DX12",
            INITIAL_EXTENT[0],
            INITIAL_EXTENT[1],
        )?
        .with_visibility(fluxel_host::WindowVisibility::Hidden),
    )?;
    let raw = window
        .window_handle()
        .map_err(|error| format!("S7 Deferred Surface DX12 window: {error:?}"))?
        .as_raw();
    // SAFETY: the Host window outlives its Surface and all accepted presentation leases.
    let target = unsafe { fluxel_rhi::dx12::Dx12::surface_from_raw_window_handle(&device, raw) }?;
    run_frames(
        device,
        target,
        output,
        |extent, phase| surface_lifecycle::resize_window(&window, extent, output, phase),
        || {
            window.poll_events()?;
            Ok(())
        },
        |frame, extent, color| {
            window.show_background()?;
            crate::capture::capture_client_with_tolerance(&window, output, frame, extent, color, 1)
        },
    )
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
pub(crate) fn run_wgl(
    device: Device<fluxel_rhi::gl::Gl>,
    window: &fluxel_host::Window,
    output: &Path,
) -> Result<(), Box<dyn Error>> {
    let target = fluxel_rhi::gl::Gl::surface(&device)?;
    // WGL needs a mapped drawable before its first visible presentation.
    // This background transition preserves the current foreground window.
    window.show_background()?;
    run_frames(
        device,
        target,
        output,
        |extent, phase| surface_lifecycle::resize_window(window, extent, output, phase),
        || {
            window.poll_events()?;
            Ok(())
        },
        |frame, extent, color| {
            window.show_background()?;
            crate::capture::capture_client_with_tolerance(window, output, frame, extent, color, 1)
        },
    )
}
