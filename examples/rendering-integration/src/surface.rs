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
struct ResizeFact {
    extent: [u32; 2],
    event: &'static str,
}

fn json_string(value: &str) -> String {
    let mut escaped = String::from("\"");
    for character in value.chars() {
        match character {
            '\"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            value if value.is_control() => escaped.push_str(&format!("\\u{:04x}", value as u32)),
            value => escaped.push(value),
        }
    }
    escaped.push('\"');
    escaped
}

fn append_resize_observation(
    output: &Path,
    phase: &str,
    requested: [u32; 2],
    events: &[String],
    selected_event: Option<&str>,
    client_extent: Option<[u32; 2]>,
) -> Result<(), Box<dyn Error>> {
    let mut log = OpenOptions::new()
        .append(true)
        .create(true)
        .open(output.join("surface-events.jsonl"))?;
    let events = events
        .iter()
        .map(|event| json_string(event))
        .collect::<Vec<_>>()
        .join(", ");
    let selected_event = selected_event.map_or_else(|| "null".to_owned(), json_string);
    let client_extent = client_extent.map_or_else(
        || "null".to_owned(),
        |[width, height]| format!("[{width}, {height}]"),
    );
    writeln!(
        log,
        "{{\"phase\":{},\"requestedExtent\":[{},{}],\"observedEvents\":[{}],\"selectedEvent\":{},\"clientExtent\":{}}}",
        json_string(phase),
        requested[0],
        requested[1],
        events,
        selected_event,
        client_extent,
    )?;
    Ok(())
}

fn initial(
    store: &AssetStore<SurfaceBytes, Vec<u8>, String>,
    handle: &fluxel_assets::AssetHandle<SurfaceBytes>,
    bytes: Vec<u8>,
) -> Result<AssetSnapshot<SurfaceBytes, Vec<u8>>, Box<dyn Error>> {
    let size = u64::try_from(bytes.len())?;
    match store.acquire(handle)? {
        Acquire::Producer(permit) => Ok(permit.commit(bytes, ResidentBytes::new(size))?),
        _ => Err("unexpected non-producer initial surface asset state".into()),
    }
}

fn replacement(
    store: &AssetStore<SurfaceBytes, Vec<u8>, String>,
    handle: &fluxel_assets::AssetHandle<SurfaceBytes>,
    bytes: Vec<u8>,
) -> Result<AssetSnapshot<SurfaceBytes, Vec<u8>>, Box<dyn Error>> {
    let size = u64::try_from(bytes.len())?;
    match store.request_replacement(handle)? {
        Production::Producer(permit) => Ok(permit.commit(bytes, ResidentBytes::new(size))?),
        Production::Waiter(_) => Err("unexpected surface replacement waiter".into()),
    }
}

fn configure<B: SurfaceBackend>(
    surface: &mut Surface<B>,
    extent: [u32; 2],
    output: &Path,
) -> Result<SurfaceConfig, Box<dyn Error>> {
    let caps = surface.capabilities()?;
    let format = if caps.formats.contains(&TextureFormat::Rgba8Unorm) {
        TextureFormat::Rgba8Unorm
    } else {
        caps.formats
            .first()
            .copied()
            .ok_or("surface exposed no format")?
    };
    let present_mode = if caps.present_modes.contains(&PresentMode::Fifo) {
        PresentMode::Fifo
    } else {
        caps.present_modes
            .first()
            .copied()
            .ok_or("surface exposed no present mode")?
    };
    let alpha_mode = if caps.alpha_modes.contains(&CompositeAlphaMode::Opaque) {
        CompositeAlphaMode::Opaque
    } else {
        caps.alpha_modes
            .first()
            .copied()
            .ok_or("surface exposed no composite alpha mode")?
    };
    let color_space = if caps.color_spaces.contains(&ColorSpace::Srgb) {
        ColorSpace::Srgb
    } else {
        caps.color_spaces
            .first()
            .copied()
            .ok_or("surface exposed no color space")?
    };
    if extent[0] < caps.min_extent[0]
        || extent[1] < caps.min_extent[1]
        || extent[0] > caps.max_extent[0]
        || extent[1] > caps.max_extent[1]
    {
        return Err(format!(
            "surface extent {extent:?} is outside its reported range {:?}..={:?}",
            caps.min_extent, caps.max_extent
        )
        .into());
    }
    if caps.usages.0 & SurfaceUsage::RENDER_ATTACHMENT.0 == 0 {
        return Err("surface does not advertise render-attachment usage".into());
    }
    let config = SurfaceConfig {
        extent,
        format,
        present_mode,
        alpha_mode,
        color_space,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 1,
    };
    if let Support::Unsupported(reason) = surface.configuration_support(&config) {
        append_configuration_observation(output, &caps, &config, false, Some(reason))?;
        return Err(
            format!("surface rejected the selected supported configuration: {reason}").into(),
        );
    }
    append_configuration_observation(output, &caps, &config, true, None)?;
    block_on(surface.configure(&config))
        .map_err(|error| format!("surface configure {extent:?}: {error:?}"))?;
    Ok(config)
}

fn append_configuration_observation(
    output: &Path,
    caps: &SurfaceCapabilities,
    config: &SurfaceConfig,
    supported: bool,
    reason: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let SurfaceCapabilities {
        formats,
        present_modes,
        alpha_modes,
        color_spaces,
        usages,
        min_extent,
        max_extent,
    } = caps;
    let SurfaceConfig {
        extent,
        format,
        present_mode,
        alpha_mode,
        color_space,
        usage,
        maximum_frame_latency,
    } = config;
    fn enum_values<T: std::fmt::Debug>(values: &[T]) -> String {
        values
            .iter()
            .map(|value| json_string(&format!("{value:?}")))
            .collect::<Vec<_>>()
            .join(",")
    }
    let mut log = OpenOptions::new()
        .append(true)
        .create(true)
        .open(output.join("surface-configuration.jsonl"))?;
    writeln!(
        log,
        concat!(
            "{{\"extent\":[{},{}],\"requiredUsage\":\"render_attachment\",\"configurationSupport\":{},\"reason\":{},",
            "\"capabilities\":{{\"formats\":[{}],\"presentModes\":[{}],\"alphaModes\":[{}],\"colorSpaces\":[{}],\"usages\":{},\"minExtent\":{:?},\"maxExtent\":{:?}}},",
            "\"configuration\":{{\"extent\":{:?},\"format\":{},\"presentMode\":{},\"alphaMode\":{},\"colorSpace\":{},\"usage\":{},\"maximumFrameLatency\":{}}}}}"
        ),
        extent[0],
        extent[1],
        supported,
        reason.map_or_else(|| "null".to_owned(), json_string),
        enum_values(formats),
        enum_values(present_modes),
        enum_values(alpha_modes),
        enum_values(color_spaces),
        usages.0,
        min_extent,
        max_extent,
        extent,
        json_string(&format!("{format:?}")),
        json_string(&format!("{present_mode:?}")),
        json_string(&format!("{alpha_mode:?}")),
        json_string(&format!("{color_space:?}")),
        usage.0,
        maximum_frame_latency,
    )?;
    Ok(())
}

#[cfg(windows)]
fn resize_window(
    window: &Window,
    target: [u32; 2],
    output: &Path,
    phase: &str,
) -> Result<ResizeFact, Box<dyn Error>> {
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};
    use windows::Win32::{
        Foundation::{HWND, RECT},
        UI::WindowsAndMessaging::{
            GetClientRect, GetWindowRect, SWP_NOACTIVATE, SWP_NOMOVE, SWP_NOZORDER, SetWindowPos,
        },
    };

    let raw = window
        .window_handle()
        .map_err(|error| format!("resize window handle: {error:?}"))?
        .as_raw();
    let RawWindowHandle::Win32(handle) = raw else {
        return Err("surface fixture expected a Win32 window".into());
    };
    let hwnd = HWND(handle.hwnd.get() as *mut _);
    let mut outer = RECT::default();
    let mut client = RECT::default();
    // SAFETY: `hwnd` was borrowed from the live host window on its owner thread;
    // the RECT values are local writable storage.
    unsafe {
        GetWindowRect(hwnd, &mut outer)?;
        GetClientRect(hwnd, &mut client)?;
    }
    let border_width = (outer.right - outer.left) - (client.right - client.left);
    let border_height = (outer.bottom - outer.top) - (client.bottom - client.top);
    let outer_width = i32::try_from(target[0])?
        .checked_add(border_width)
        .ok_or("resize width overflow")?;
    let outer_height = i32::try_from(target[1])?
        .checked_add(border_height)
        .ok_or("resize height overflow")?;
    // SAFETY: this changes only the live fixture HWND and retains its z-order
    // and location. The previous renderer submission was completion-drained.
    unsafe {
        SetWindowPos(
            hwnd,
            None,
            0,
            0,
            outer_width,
            outer_height,
            SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE,
        )?;
    }
    let events = window.poll_events()?;
    let observed_events = events
        .iter()
        .map(|event| format!("{event:?}"))
        .collect::<Vec<_>>();
    let mut observed = RECT::default();
    // Query the actual client extent even when the Host event is absent. The
    // resulting log must remain useful to diagnose an oracle refusal.
    // SAFETY: the same live HWND and local RECT remain valid for this query.
    unsafe {
        GetClientRect(hwnd, &mut observed)?;
    }
    let extent = [
        u32::try_from(observed.right - observed.left)?,
        u32::try_from(observed.bottom - observed.top)?,
    ];
    let event = events.iter().find_map(|event| match event {
        fluxel_host::WindowEvent::Resized { width, height } if [*width, *height] == target => {
            Some("resized")
        }
        fluxel_host::WindowEvent::Restored { width, height } if [*width, *height] == target => {
            Some("restored")
        }
        _ => None,
    });
    let Some(event) = event else {
        append_resize_observation(output, phase, target, &observed_events, None, Some(extent))?;
        return Err(format!("host did not report requested resize {target:?}: {events:?}").into());
    };
    append_resize_observation(
        output,
        phase,
        target,
        &observed_events,
        Some(event),
        Some(extent),
    )?;
    if extent != target {
        return Err(format!("host resize produced {extent:?}, requested {target:?}").into());
    }
    Ok(ResizeFact { extent, event })
}

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
    Capture: Fn(usize, [u32; 2]) -> Result<(), Box<dyn Error>>,
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
                "{{\"rgba\":[{},{},{},{}],\"extent\":[{},{}]}}\n",
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
    capture(0, initial_extent)?;
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
    capture(1, resized.extent)?;
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
    capture(2, restored.extent)?;
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

fn await_visual_release<Pump>(
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
        &|frame, extent| crate::capture::capture_client(&window, output, frame, extent),
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
        &|frame, extent| crate::capture::capture_client(window, output, frame, extent),
    )
}
