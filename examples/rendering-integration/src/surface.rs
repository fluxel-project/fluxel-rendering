//! Native surface lifecycle fixture for the independent S0/S1 consumer.
//!
//! The Renderer owns upload, raster recording, presentation and completion.
//! This module owns only the host-bound surface lifecycle: configuration,
//! image acquisition, and an actual Win32 client resize. The fixture requests
//! that resize through the native HWND, then requires the Host event and the
//! queried client extent to agree before reconfiguring the RHI surface.

use std::{
    error::Error,
    fs,
    fs::OpenOptions,
    io::Write,
    path::Path,
    time::{Duration, Instant},
};

use fluxel_assets::{Acquire, AssetKind, AssetSnapshot, AssetStore, Production, ResidentBytes};
#[cfg(windows)]
use fluxel_host::Window;
#[cfg(all(windows, feature = "dx12"))]
use fluxel_host::WindowConfig;
use fluxel_renderer::Renderer;
use fluxel_rhi::{
    api::{
        ColorSpace, CompositeAlphaMode, Device, PresentMode, Support, Surface, SurfaceCapabilities,
        SurfaceConfig, SurfaceUsage, TextureFormat,
    },
    backend::{
        ExecutionBackend, MappingBackend, PipelineBackend, RasterBackend, ReadbackBackend,
        ResourceBackend, ShaderBackend, SurfaceBackend,
    },
};

use crate::{block_on, raster};

struct SurfaceBytes;
impl AssetKind for SurfaceBytes {}

#[derive(Clone, Copy)]
pub(crate) struct ResizeFact {
    /// The actual client extent after a matching Host lifecycle event.
    pub(crate) extent: [u32; 2],
    /// The Host lifecycle event that confirmed the extent.
    pub(crate) event: &'static str,
}

#[path = "surface_lifecycle.rs"]
mod surface_lifecycle;
pub(crate) use surface_lifecycle::configure;
#[cfg(windows)]
pub(crate) use surface_lifecycle::resize_window;
use surface_lifecycle::{initial, replacement};

fn run_frames<B, Resize, Pump, Capture>(
    device: Device<B>,
    mut surface: Surface<B>,
    output: &Path,
    initial_extent: [u32; 2],
    resize: &Resize,
    pump: &Pump,
    capture: &Capture,
) -> Result<(), Box<dyn Error>>
where
    Resize: Fn([u32; 2], &str) -> Result<ResizeFact, Box<dyn Error>>,
    Pump: Fn() -> Result<(), Box<dyn Error>>,
    Capture: Fn(usize, [u32; 2], [u8; 4]) -> Result<(), Box<dyn Error>>,
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + RasterBackend
        + PipelineBackend
        + ShaderBackend
        + SurfaceBackend,
{
    fs::create_dir_all(output)?;
    // Start this stream before a host transition so a failed resize still
    // retains the events and client-extent observation that led to refusal.
    fs::write(output.join("surface-events.jsonl"), b"")?;
    let store = AssetStore::<SurfaceBytes, Vec<u8>, String>::new();
    let handle = store.create()?;
    let colors = [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]];
    let inputs = colors.map(raster::vertices);
    let snapshots = [
        initial(&store, &handle, inputs[0].clone())?,
        replacement(&store, &handle, inputs[1].clone())?,
        replacement(&store, &handle, inputs[2].clone())?,
    ];
    for index in 0..inputs.len() {
        fs::write(
            output.join(format!("surface-input-{index}.bin")),
            &inputs[index],
        )?;
        let extent = if index == 1 {
            [256, 144]
        } else {
            initial_extent
        };
        fs::write(
            output.join(format!("surface-expected-{index}.json")),
            format!(
                "{{\"rgba\":[{},{},{},{}],\"extent\":[{},{}],\"triangleNdc\":[[-0.75,-0.65],[0.55,-0.50],[-0.20,0.75]],\"backgroundRgb\":[0,0,0],\"stableInteriorSample\":[0.40,0.57]}}\n",
                colors[index][0],
                colors[index][1],
                colors[index][2],
                colors[index][3],
                extent[0],
                extent[1],
            ),
        )?;
    }
    let generations = [
        snapshots[0].generation().get(),
        snapshots[1].generation().get(),
        snapshots[2].generation().get(),
    ];
    // Persist original inputs and expected pixels before the first native
    // oracle can return, including a configure or pipeline failure.
    let initial_config = configure(&mut surface, initial_extent, output)?;
    let pipeline = raster::make_pipeline(&device, initial_config.format)?;
    let mut renderer = Renderer::new(device)?;

    // The covering completion retires the old image lease and all renderer
    // owners before the surface configuration generation changes.
    let first = block_on(renderer.submit_surface_raster(
        snapshots[0].clone(),
        pipeline.clone(),
        block_on(surface.acquire()).map_err(|error| format!("surface first acquire: {error:?}"))?,
        initial_extent,
    ))
    .map_err(|error| format!("surface first submit: {error:?}"))?;
    block_on(renderer.complete(&first))
        .map_err(|error| format!("surface first complete: {error:?}"))?;
    if renderer.pending() != 0 {
        return Err("surface submission survived its completion".into());
    }
    capture(0, initial_extent, colors[0])?;
    await_visual_release(output, 0, initial_extent, pump)?;

    let resized = resize([256, 144], "resize")?;
    let resized_config = configure(&mut surface, resized.extent, output)?;
    if resized_config.format != initial_config.format {
        return Err("surface format changed across fixture resize".into());
    }
    let second = block_on(
        renderer.submit_surface_raster(
            snapshots[1].clone(),
            pipeline.clone(),
            block_on(surface.acquire())
                .map_err(|error| format!("surface second acquire: {error:?}"))?,
            resized.extent,
        ),
    )
    .map_err(|error| format!("surface second submit: {error:?}"))?;
    block_on(renderer.complete(&second))
        .map_err(|error| format!("surface second complete: {error:?}"))?;
    if renderer.pending() != 0 {
        return Err("resized surface submission survived completion".into());
    }
    capture(1, resized.extent, colors[1])?;
    await_visual_release(output, 1, resized.extent, pump)?;

    let restored = resize(initial_extent, "restore")?;
    let restored_config = configure(&mut surface, restored.extent, output)?;
    if restored_config.format != initial_config.format {
        return Err("surface format changed across fixture restore".into());
    }
    let third = block_on(renderer.submit_surface_raster(
        snapshots[2].clone(),
        pipeline,
        block_on(surface.acquire()).map_err(|error| format!("surface third acquire: {error:?}"))?,
        restored.extent,
    ))
    .map_err(|error| format!("surface third submit: {error:?}"))?;
    block_on(renderer.complete(&third))
        .map_err(|error| format!("surface third complete: {error:?}"))?;
    if renderer.pending() != 0 {
        return Err("restored surface submission survived completion".into());
    }
    capture(2, restored.extent, colors[2])?;
    await_visual_release(output, 2, restored.extent, pump)?;

    fs::write(
        output.join("surface-workload.json"),
        format!(
            concat!(
                "{{\n  \"schemaVersion\": 1,\n",
                "  \"workload\": \"s1-renderer-surface-raster-three-generations\",\n",
                "  \"initialExtent\": [{}, {}],\n",
                "  \"hostResizeEventVerified\": true,\n",
                "  \"clientExtentVerified\": true,\n",
                "  \"resizeEvent\": {:?},\n",
                "  \"restoreEvent\": {:?},\n",
                "  \"frames\": 3,\n",
                "  \"assetGenerations\": [{}, {}, {}],\n",
                "  \"completionBeforeReconfigure\": true\n}}\n"
            ),
            initial_extent[0],
            initial_extent[1],
            resized.event,
            restored.event,
            generations[0],
            generations[1],
            generations[2],
        ),
    )?;
    Ok(())
}

/// Optionally keeps a completed presented frame visible for external capture.
///
/// This is intentionally called only after renderer completion; it never
/// supplies GPU completion evidence or retires an acquired surface image.
pub(crate) fn await_visual_release<Pump>(
    output: &Path,
    frame: usize,
    extent: [u32; 2],
    pump: &Pump,
) -> Result<(), Box<dyn Error>>
where
    Pump: Fn() -> Result<(), Box<dyn Error>>,
{
    let Some(value) = std::env::var_os("FLUXEL_INTEGRATION_FRAME_HOLD_MS") else {
        return Ok(());
    };
    let milliseconds = value
        .to_string_lossy()
        .parse::<u64>()
        .map_err(|_| "FLUXEL_INTEGRATION_FRAME_HOLD_MS must be an integer")?;
    if milliseconds == 0 || milliseconds > 120_000 {
        return Err("FLUXEL_INTEGRATION_FRAME_HOLD_MS must be 1 through 120000".into());
    }
    let release = output.join(format!("surface-frame-{frame}.release"));
    let ready = output.join(format!("surface-frame-{frame}.ready.json"));
    fs::write(
        &ready,
        format!(
            concat!(
                "{{\n  \"schemaVersion\": 1,\n",
                "  \"frame\": {},\n",
                "  \"extent\": [{}, {}],\n",
                "  \"releaseFile\": {:?}\n}}\n"
            ),
            frame,
            extent[0],
            extent[1],
            release
                .file_name()
                .ok_or("release file has no name")?
                .to_string_lossy(),
        ),
    )?;
    let deadline = Instant::now() + Duration::from_millis(milliseconds);
    loop {
        pump()?;
        if release.is_file() {
            fs::write(
                output.join(format!("surface-frame-{frame}.released.json")),
                "{\"released\":true}\n",
            )?;
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(
                format!("timed out waiting for visual capture release of frame {frame}").into(),
            );
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[cfg(all(windows, feature = "dx12"))]
pub(crate) fn run_dx12(
    device: Device<fluxel_rhi::dx12::Dx12>,
    output: &Path,
) -> Result<(), Box<dyn Error>> {
    use raw_window_handle::HasWindowHandle;
    let window = Window::new(WindowConfig::new("Fluxel S1 Surface DX12", 320, 180)?)?;
    let raw = window
        .window_handle()
        .map_err(|error| format!("DX12 surface window handle: {error:?}"))?
        .as_raw();
    // SAFETY: `window` outlives the surface and every lease submitted above.
    let surface = unsafe { fluxel_rhi::dx12::Dx12::surface_from_raw_window_handle(&device, raw) }?;
    run_frames(
        device,
        surface,
        output,
        [320, 180],
        &|extent, phase| resize_window(&window, extent, output, phase),
        &|| {
            window.poll_events()?;
            Ok(())
        },
        &|frame, extent, color| {
            crate::capture::capture_client(&window, output, frame, extent, color)
        },
    )
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
pub(crate) fn run_wgl(
    device: Device<fluxel_rhi::gl::Gl>,
    window: &Window,
    output: &Path,
) -> Result<(), Box<dyn Error>> {
    let surface = fluxel_rhi::gl::Gl::surface(&device)?;
    run_frames(
        device,
        surface,
        output,
        [320, 180],
        &|extent, phase| resize_window(window, extent, output, phase),
        &|| {
            window.poll_events()?;
            Ok(())
        },
        &|frame, extent, color| {
            crate::capture::capture_client(window, output, frame, extent, color)
        },
    )
}
