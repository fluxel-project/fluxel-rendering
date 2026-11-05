//! Native Surface presentation proof for the bounded S6 composed Forward path.
//!
//! Each presented frame contains the ordinary mesh and external quad Feature
//! in one `submit_forward_surface_with_feature` call. Surface leases have no
//! readback ticket by contract, so a native client capture proves presentation;
//! a separately labelled offscreen submission of the same immutable snapshot
//! proves the Renderer readback path without being mistaken for Surface data.

use std::{error::Error, fs, path::Path};

use external_renderer::{ExternalComposedPipeline, QuadFeature, QuadInput};
use fluxel_assets::AssetKind;
use fluxel_renderer::{
    CpuPhase, FeatureBinding, ForwardResources, RenderBounds, RenderChange, RenderChangeSet,
    RenderFeature, RenderRecord, RenderScene, RenderTransform, RenderView, Renderer,
};
use fluxel_rhi::{
    api::{Device, Surface},
    backend::{
        ExecutionBackend, MappingBackend, PipelineBackend, RasterBackend, ReadbackBackend,
        ResourceBackend, ShaderBackend, SurfaceBackend,
    },
};

use super::{fixtures, observed::ObservedQuadFeature, oracle};
use crate::{block_on, surface as surface_lifecycle};

const INITIAL_EXTENT: [u32; 2] = [320, 180];
const RESIZED_EXTENT: [u32; 2] = [256, 144];

/// Runs two actual presented S6 composed frames, with a host resize between them.
pub(super) fn run_frames<B, Resize, Pump, Capture>(
    device: Device<B>,
    mut surface_target: Surface<B>,
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

    let assets = fixtures::Assets::new()?;
    assets.write(output)?;
    let pipeline = ExternalComposedPipeline::new();
    let mut renderer = Renderer::new(device)?;
    let mut scene = RenderScene::<ForwardResources>::new()?;
    let issuer = scene.feature_issuer()?;
    let feature = ObservedQuadFeature::new(QuadFeature::new(&issuer)?);
    let mut storage = issuer.storage::<QuadInput>();
    let volume = RenderBounds::new([-1.0; 3], [1.0; 3])?;

    // The companion view has a stable resolution for raw RHI readback. It has
    // the same scene snapshot, assets, Feature input and pipeline as each
    // presented frame; it is explicitly not a substitute for HWND capture.
    let mut changes = RenderChangeSet::new();
    let mesh = assets.record(false);
    changes.push(RenderChange::Insert(mesh.clone()));
    changes.push(RenderChange::InsertView(RenderView::new(
        RenderTransform::identity(),
        INITIAL_EXTENT,
    )?));
    changes.push(RenderChange::InsertView(RenderView::new(
        RenderTransform::identity(),
        oracle::EXTENT,
    )?));
    let receipt = scene.apply(&changes)?;
    let surface_view = receipt.inserted_views()[0];
    let companion_view = receipt.inserted_views()[1];
    let forward_assets = assets.forward();

    let initial_config = surface_lifecycle::configure(&mut surface_target, INITIAL_EXTENT, output)?;
    append_lifecycle(output, 0, "configure", None, initial_config.extent)?;
    submit_case(
        &mut renderer,
        &mut surface_target,
        &pipeline,
        &issuer,
        &feature,
        &mut storage,
        scene.snapshot(),
        surface_view,
        companion_view,
        volume,
        &mesh,
        &forward_assets,
        assets.input(false, -10),
        0,
        initial_config.extent,
        output,
        &capture,
    )?;
    surface_lifecycle::await_visual_release(output, 0, initial_config.extent, &pump)?;

    let resized = resize(RESIZED_EXTENT, "s6-composed-resize")?;
    append_lifecycle(output, 3, "resize", None, resized.extent)?;
    fs::write(
        output.join("s6-surface-reconfigure-input.json"),
        format!(
            concat!(
                "{{\"schemaVersion\":1,\"previousExtent\":[{},{}],",
                "\"observedClientExtent\":[{},{}],\"nextOperation\":\"configure\"}}\n",
            ),
            initial_config.extent[0],
            initial_config.extent[1],
            resized.extent[0],
            resized.extent[1],
        ),
    )?;
    let resized_config = surface_lifecycle::configure(&mut surface_target, resized.extent, output)?;
    append_lifecycle(output, 4, "configure", None, resized_config.extent)?;
    let mut view_change = RenderChangeSet::new();
    view_change.push(RenderChange::UpdateView(
        surface_view,
        RenderView::new(RenderTransform::identity(), resized.extent)?,
    ));
    scene.apply(&view_change)?;
    submit_case(
        &mut renderer,
        &mut surface_target,
        &pipeline,
        &issuer,
        &feature,
        &mut storage,
        scene.snapshot(),
        surface_view,
        companion_view,
        volume,
        &mesh,
        &forward_assets,
        assets.input(false, 10),
        1,
        resized_config.extent,
        output,
        &capture,
    )?;
    surface_lifecycle::await_visual_release(output, 1, resized_config.extent, &pump)?;

    if renderer.pending() != 0 {
        return Err("S6 Surface workload retained a completed frame".into());
    }
    let callbacks = feature.callback_counts();
    if callbacks.prepare != 4 || callbacks.contribute != 4 {
        return Err(format!(
            "S6 Surface and companion submissions expected four callbacks, got {callbacks:?}"
        )
        .into());
    }
    write_workload(output, callbacks)
}

#[allow(clippy::too_many_arguments)]
fn submit_case<B, Capture>(
    renderer: &mut Renderer<B>,
    surface_target: &mut Surface<B>,
    pipeline: &ExternalComposedPipeline,
    issuer: &fluxel_renderer::FeatureIssuer,
    feature: &ObservedQuadFeature,
    storage: &mut fluxel_renderer::TypedFeatureStorage<QuadInput>,
    snapshot: fluxel_renderer::FrameSnapshot<ForwardResources>,
    surface_view: fluxel_renderer::RenderViewId,
    companion_view: fluxel_renderer::RenderViewId,
    volume: RenderBounds,
    mesh: &RenderRecord<ForwardResources>,
    assets: &fluxel_renderer::ForwardAssets,
    input: QuadInput,
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
    let quad_after_mesh = input.markers[0].order > 0;
    let expected = oracle::image(Some(fixtures::MESH), Some(fixtures::QUAD), quad_after_mesh);
    let mask = oracle::mask();
    let context = issuer.context(&snapshot, CpuPhase::Reference, feature.handle())?;
    let typed = storage.publish(&context, input)?;
    let expected_surface_color = if quad_after_mesh {
        fixtures::QUAD
    } else {
        fixtures::MESH
    };
    // Persist independent facts before the first acquired presentation lease.
    fs::write(
        output.join(format!("s6-surface-frame-{frame}-input.json")),
        input_json(
            &snapshot,
            &typed,
            mesh,
            surface_extent,
            expected_surface_color,
        )?,
    )?;
    fs::write(
        output.join(format!("s6-surface-frame-{frame}-expected.rgba")),
        &expected,
    )?;
    fs::write(
        output.join(format!("s6-surface-frame-{frame}-expected.ppm")),
        oracle::ppm(&expected)?,
    )?;
    fs::write(output.join("s6-surface-stable-mask.bin"), &mask)?;

    let image = block_on(surface_target.acquire())
        .map_err(|error| format!("S6 Surface frame {frame} acquire: {error:?}"))?;
    if image.extent() != surface_extent {
        return Err(format!(
            "S6 Surface frame {frame} lease extent {:?} differed from configured {surface_extent:?}",
            image.extent()
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
    let surface_submission = block_on(renderer.submit_forward_surface_with_feature(
        pipeline,
        snapshot.clone(),
        surface_view,
        volume,
        assets,
        FeatureBinding::new(issuer, feature, &typed),
        image,
    ))?;
    block_on(renderer.complete(&surface_submission))?;
    append_lifecycle(
        output,
        if frame == 0 { 2 } else { 6 },
        "complete",
        Some(frame),
        surface_extent,
    )?;
    if renderer.pending() != 0 {
        return Err(format!("S6 Surface frame {frame} remained pending after completion").into());
    }
    capture(frame, surface_extent, expected_surface_color)?;

    // Surface's public contract deliberately lacks a readback ticket. This
    // offscreen use shares its snapshot/input/pipeline, but has a distinct
    // artifact name so it cannot be represented as a Surface readback.
    let companion_submission = block_on(renderer.submit_forward_with_feature(
        pipeline,
        snapshot,
        companion_view,
        volume,
        assets,
        FeatureBinding::new(issuer, feature, &typed),
    ))?;
    let actual = block_on(renderer.readback(&companion_submission))?;
    fs::write(
        output.join(format!("s6-surface-frame-{frame}-readback.rgba")),
        &actual,
    )?;
    fs::write(
        output.join(format!("s6-surface-frame-{frame}-readback.ppm")),
        oracle::ppm(&actual)?,
    )?;
    oracle::verify(
        &actual,
        &expected,
        &mask,
        &format!("surface companion {frame}"),
    )
}

fn write_workload(
    output: &Path,
    callbacks: super::observed::CallbackCounts,
) -> Result<(), Box<dyn Error>> {
    let first_artifacts = [
        "s6-surface-frame-0-input.json",
        "s6-surface-frame-0-expected.rgba",
        "s6-surface-frame-0-expected.ppm",
        "s6-surface-frame-0-readback.rgba",
        "s6-surface-frame-0-readback.ppm",
        "surface-client-frame-0.bmp",
        "surface-client-frame-0.capture.json",
        "surface-client-frame-0.verdict.json",
        "s6-initial-mesh-geometry-stream-0.bin",
        "s6-initial-quad-geometry-stream-0.bin",
        "s6-initial-mesh-geometry.txt",
        "s6-initial-quad-geometry.txt",
        "s6-initial-mesh-material.txt",
        "s6-initial-quad-material.txt",
        "s6-initial-mesh-texture.rgba",
        "s6-initial-quad-texture.rgba",
        "s6-surface-stable-mask.bin",
        "surface-events.jsonl",
        "surface-configuration.jsonl",
        "s6-surface-lifecycle.jsonl",
        "s6-surface-reconfigure-input.json",
    ];
    let second_artifacts = [
        "s6-surface-frame-1-input.json",
        "s6-surface-frame-1-expected.rgba",
        "s6-surface-frame-1-expected.ppm",
        "s6-surface-frame-1-readback.rgba",
        "s6-surface-frame-1-readback.ppm",
        "surface-client-frame-1.bmp",
        "surface-client-frame-1.capture.json",
        "surface-client-frame-1.verdict.json",
    ];
    let json_list = |entries: &[&str]| {
        entries
            .iter()
            .map(|entry| format!("\"{entry}\""))
            .collect::<Vec<_>>()
            .join(",")
    };
    fs::write(
        output.join("s6-surface-workload.json"),
        format!(
            concat!(
                "{{\"schemaVersion\":1,\"workload\":\"s6-composed-surface-resize\",",
                "\"initialExtent\":[{},{}],\"resizedExtent\":[{},{}],",
                "\"cases\":[",
                "{{\"name\":\"mesh-after-quad-initial\",\"artifacts\":[{}]}},",
                "{{\"name\":\"quad-after-mesh-resized\",\"artifacts\":[{}]}}],",
                "\"surfaceProof\":\"native-client-capture\",",
                "\"readbackProof\":\"same-snapshot-offscreen-companion\",",
                "\"completionBeforeReconfigure\":true,",
                "\"hostResizeReconfiguredBeforeAcquire\":true,",
                "\"featureCallbacks\":{{\"prepare\":{},\"contribute\":{}}}}}\n",
            ),
            INITIAL_EXTENT[0],
            INITIAL_EXTENT[1],
            RESIZED_EXTENT[0],
            RESIZED_EXTENT[1],
            json_list(&first_artifacts),
            json_list(&second_artifacts),
            callbacks.prepare,
            callbacks.contribute,
        ),
    )?;
    Ok(())
}

/// Writes facts from the exact record and typed snapshot that will be bound.
/// This deliberately serializes ids as their diagnostic slot identity plus
/// content generation, rather than reconstructing assets from fixture names.
fn input_json(
    snapshot: &fluxel_renderer::FrameSnapshot<ForwardResources>,
    typed: &fluxel_renderer::TypedFeatureSnapshot<QuadInput>,
    mesh: &RenderRecord<ForwardResources>,
    surface_extent: [u32; 2],
    surface_interior_rgba: [u8; 4],
) -> Result<String, Box<dyn Error>> {
    let marker = typed
        .data()
        .markers
        .first()
        .ok_or("S6 Surface typed input contained no marker")?;
    Ok(format!(
        concat!(
            "{{\"schemaVersion\":1,\"sceneVersion\":{},\"featureSnapshotSceneVersion\":{},",
            "\"featureEnabled\":true,\"surfaceExtent\":{:?},\"companionExtent\":{:?},",
            "\"surfaceInteriorRgba\":{:?},\"mesh\":{{\"bounds\":{{\"minimum\":{:?},\"maximum\":{:?}}},",
            "\"transformColumns\":{:?},\"assets\":{{\"geometry\":{},\"material\":{},\"texture\":{}}}}},",
            "\"marker\":{{\"id\":{},\"center\":{:?},\"halfExtent\":{:?},\"tint\":{:?},\"order\":{},",
            "\"assets\":{{\"geometry\":{},\"material\":{},\"texture\":{}}}}}}}\n"
        ),
        snapshot.version(),
        typed.scene_version(),
        surface_extent,
        oracle::EXTENT,
        surface_interior_rgba,
        mesh.bounds.minimum(),
        mesh.bounds.maximum(),
        mesh.transform.columns(),
        asset_json(mesh.resources.geometry),
        asset_json(mesh.resources.material),
        asset_json(mesh.resources.texture),
        marker.id,
        marker.center,
        marker.half_extent,
        marker.tint,
        marker.order,
        asset_json(marker.assets.geometry),
        asset_json(marker.assets.material),
        asset_json(marker.assets.texture),
    ))
}

fn asset_json<K: AssetKind>(reference: fluxel_renderer::SceneAssetRef<K>) -> String {
    format!(
        "{{\"slot\":{},\"slotGeneration\":{},\"contentGeneration\":{}}}",
        reference.id().slot(),
        reference.id().slot_generation(),
        reference.generation().get(),
    )
}

/// Appends observed lifecycle boundaries in the order their operations return.
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
        .open(output.join("s6-surface-lifecycle.jsonl"))?;
    writeln!(
        log,
        "{{\"sequence\":{sequence},\"event\":\"{event}\",\"frame\":{},\"extent\":[{},{}]}}",
        frame.map_or_else(|| "null".to_owned(), |value| value.to_string()),
        extent[0],
        extent[1],
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
            "Fluxel S6 composed Surface DX12",
            INITIAL_EXTENT[0],
            INITIAL_EXTENT[1],
        )?
        .with_visibility(fluxel_host::WindowVisibility::Hidden),
    )?;
    let raw = window
        .window_handle()
        .map_err(|error| format!("S6 DX12 window handle: {error:?}"))?
        .as_raw();
    // SAFETY: the Host window outlives the Surface and every accepted lease.
    let surface_target =
        unsafe { fluxel_rhi::dx12::Dx12::surface_from_raw_window_handle(&device, raw) }?;
    run_frames(
        device,
        surface_target,
        output,
        |extent, phase| surface_lifecycle::resize_window(&window, extent, output, phase),
        || {
            window.poll_events()?;
            Ok(())
        },
        |frame, extent, color| {
            window.show_background()?;
            crate::capture::capture_client(&window, output, frame, extent, color)
        },
    )
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
pub(crate) fn run_wgl(
    device: Device<fluxel_rhi::gl::Gl>,
    window: &fluxel_host::Window,
    output: &Path,
) -> Result<(), Box<dyn Error>> {
    let surface_target = fluxel_rhi::gl::Gl::surface(&device)?;
    // WGL needs a mapped drawable before its first visible presentation.
    // This background transition preserves the current foreground window.
    window.show_background()?;
    run_frames(
        device,
        surface_target,
        output,
        |extent, phase| surface_lifecycle::resize_window(window, extent, output, phase),
        || {
            window.poll_events()?;
            Ok(())
        },
        |frame, extent, color| {
            window.show_background()?;
            crate::capture::capture_client(window, output, frame, extent, color)
        },
    )
}
