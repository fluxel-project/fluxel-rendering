//! Native device opening and one same-Graph visible/capture path for DX12 and WGL.

use crate::Mode;
use fluxel_host::{Window, WindowConfig, WindowVisibility};
use fluxel_renderer::{
    BoundsVisibility, ForwardAssets, ForwardCpuPipeline, FrameSnapshot, GeometryData,
    PreparedCaptureReceiver, RenderBounds, RenderViewId, Renderer,
};
use fluxel_rhi::{
    api::{
        ColorSpace, CompositeAlphaMode, Device, DeviceRequest, PresentMode, Surface, SurfaceConfig,
        SurfaceUsage, TextureFormat,
    },
    backend::{
        ExecutionBackend, MappingBackend, PipelineBackend, RasterBackend, ReadbackBackend,
        ResourceBackend, ShaderBackend, SurfaceBackend,
    },
};
#[cfg(feature = "dx12")]
use raw_window_handle::HasWindowHandle;
use renderer_07_offscreen_material::{evidence, inputs, oracle, runtime, window_oracle};
use std::{error::Error, path::Path, time::Duration};

type Result<T = ()> = std::result::Result<T, Box<dyn Error>>;

pub(super) fn run(
    mode: Mode,
    exit_after_capture: bool,
    dragon: GeometryData,
    plane: GeometryData,
    output: &Path,
) -> Result {
    #[cfg(all(feature = "dx12", feature = "native-gl-wgl"))]
    return Err("select exactly one native backend feature: dx12 or native-gl-wgl".into());
    #[cfg(all(feature = "dx12", not(feature = "native-gl-wgl")))]
    return run_dx12(mode, exit_after_capture, dragon, plane, output);
    #[cfg(all(feature = "native-gl-wgl", not(feature = "dx12")))]
    return run_wgl(mode, exit_after_capture, dragon, plane, output);
    #[allow(unreachable_code)]
    Err("no native backend feature is enabled".into())
}

#[cfg(feature = "dx12")]
fn run_dx12(
    mode: Mode,
    exit_after_capture: bool,
    dragon: GeometryData,
    plane: GeometryData,
    output: &Path,
) -> Result {
    use fluxel_rhi::dx12::Dx12Instance;
    if matches!(mode, Mode::Visible) {
        enable_per_monitor_v2(output)?;
    }
    // WinRT must precede and outlive every native GPU owner.
    let capture = create_capture(mode)?;
    let native = Dx12Instance::new()?;
    let instance = native.instance()?;
    let adapter = instance
        .enumerate()?
        .into_iter()
        .next()
        .ok_or("no DX12 adapter")?;
    let device = runtime::block_on(instance.open(DeviceRequest {
        adapter: adapter.id,
        required_features: Vec::new(),
    }))?;
    crate::diagnostics::record_dx12_device_open(output)?;
    match mode {
        Mode::Offscreen => runtime::run_offscreen(device, dragon, plane, output),
        Mode::Visible => {
            let window = host_window()?;
            let raw = window
                .window_handle()
                .map_err(|e| format!("visible DX12 window handle: {e:?}"))?
                .as_raw();
            // SAFETY: the surface drops before the host window.
            let surface =
                unsafe { fluxel_rhi::dx12::Dx12::surface_from_raw_window_handle(&device, raw) }?;
            run_visible(
                device,
                surface,
                &window,
                dragon,
                plane,
                output,
                capture.as_ref(),
                exit_after_capture,
            )
        }
    }
}

#[cfg(feature = "native-gl-wgl")]
fn run_wgl(
    mode: Mode,
    exit_after_capture: bool,
    dragon: GeometryData,
    plane: GeometryData,
    output: &Path,
) -> Result {
    use fluxel_rhi::{
        api::{AdapterId, Instance, InstanceId},
        gl::{Gl, GlInstance},
    };
    if matches!(mode, Mode::Visible) {
        enable_per_monitor_v2(output)?;
    }
    // Capture precedes HWND/context creation and outlives their GPU owners.
    let capture = create_capture(mode)?;
    let window = host_window()?;
    // SAFETY: this exact host window outlives its adopted context and device.
    let native = unsafe { GlInstance::from_wgl_window(&window, inputs::EXTENT) }?;
    let id = InstanceId::allocate()?;
    // SAFETY: this instance exclusively owns the adopted WGL context.
    let instance = unsafe { Instance::<Gl>::from_native(id, native) };
    let adapter = instance
        .enumerate()?
        .into_iter()
        .next()
        .ok_or("no WGL adapter")?;
    let device = runtime::block_on(instance.open(DeviceRequest {
        adapter: AdapterId {
            instance: id,
            ordinal: adapter.id.ordinal,
        },
        required_features: Vec::new(),
    }))?;
    let observed_device = device.clone();
    let mut observer = crate::diagnostics::WglObserver::install(&observed_device)?;
    let result = match mode {
        Mode::Offscreen => runtime::run_offscreen(device, dragon, plane, output),
        Mode::Visible => {
            Gl::surface(&device)
                .map_err(|error| -> Box<dyn Error> { Box::new(error) })
                .and_then(|surface| {
                    // WGL needs a mapped drawable before presentation; never activate it.
                    window.show_background()?;
                    run_visible(
                        device,
                        surface,
                        &window,
                        dragon,
                        plane,
                        output,
                        capture.as_ref(),
                        exit_after_capture,
                    )
                })
        }
    };
    let diagnostic_result = observer.write(&observed_device, output);
    result.and(diagnostic_result)
}

fn host_window() -> Result<Window> {
    Ok(Window::new(
        WindowConfig::new(
            "Fluxel Renderer 07 capture Material",
            inputs::EXTENT[0],
            inputs::EXTENT[1],
        )?
        .with_visibility(WindowVisibility::Hidden),
    )?)
}

fn create_capture(mode: Mode) -> Result<Option<crate::integration_capture::CaptureRuntime>> {
    match mode {
        Mode::Visible => Ok(Some(crate::integration_capture::CaptureRuntime::new()?)),
        Mode::Offscreen => Ok(None),
    }
}

fn enable_per_monitor_v2(output: &Path) -> Result {
    unsafe {
        use windows::Win32::UI::HiDpi::{
            DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, SetProcessDpiAwarenessContext,
        };
        SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)?;
    }
    std::fs::write(
        output.join("visible-dpi.json"),
        "{\"schemaVersion\":1,\"processDpiAwareness\":\"per-monitor-aware-v2\",\"scope\":\"fixture process only; configured before WinRT/HWND/native GPU objects\"}\n",
    )?;
    Ok(())
}

fn run_visible<B>(
    device: Device<B>,
    mut surface: Surface<B>,
    window: &Window,
    dragon: GeometryData,
    plane: GeometryData,
    output: &Path,
    capture: Option<&crate::integration_capture::CaptureRuntime>,
    exit_after_capture: bool,
) -> Result
where
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + RasterBackend
        + PipelineBackend
        + ShaderBackend
        + SurfaceBackend,
{
    let _capture = capture.ok_or("visible path requires initialized Windows Graphics Capture")?;
    configure_surface(&mut surface)?;
    let expected = oracle::build(&dragon, &plane)?;
    let primary = inputs::build_capture_scene(dragon.clone(), plane.clone())?;
    let control = inputs::build_capture_scene_with_albedo(
        dragon.clone(),
        plane.clone(),
        oracle::CONTROL_ALBEDO,
    )?;
    evidence::write_pre_submission(output, &dragon, &plane, &expected, &primary, &control)?;
    let primary_companion = CompanionInput::from(&primary);
    let control_companion = CompanionInput::from(&control);
    let mut renderer = Renderer::new(device)?;
    submit_visible(&mut surface, &mut renderer, primary)?;
    // DX12 becomes visible here; WGL was mapped before its first present.
    window.show_background()?;
    let primary_rgba = run_companion(&mut renderer, primary_companion, output, "visible-primary")?;
    capture_and_verify(
        window,
        output,
        0,
        &expected,
        oracle::PRIMARY_ALBEDO,
        &primary_rgba,
    )?;
    submit_visible(&mut surface, &mut renderer, control)?;
    let control_rgba = run_companion(&mut renderer, control_companion, output, "visible-control")?;
    capture_and_verify(
        window,
        output,
        1,
        &expected,
        oracle::CONTROL_ALBEDO,
        &control_rgba,
    )?;
    oracle::verify_pair(&primary_rgba, &control_rgba, &expected)?;
    std::fs::write(
        output.join("visible-result.json"),
        "{\"schemaVersion\":1,\"surface\":\"two same-Graph capture receiver control submissions presented through acquired Surface leases\",\"window\":\"shown in background without activation\",\"visualCapture\":\"captured-and-probe-verified\"}\n",
    )?;
    if !exit_after_capture {
        wait_for_close(window)?;
    }
    Ok(())
}

fn configure_surface<B: SurfaceBackend>(surface: &mut Surface<B>) -> Result {
    let capabilities = surface.capabilities()?;
    let config = [TextureFormat::Bgra8Unorm, TextureFormat::Rgba8Unorm]
        .into_iter()
        .filter(|format| capabilities.formats.contains(format))
        .map(|format| SurfaceConfig {
            extent: inputs::EXTENT,
            format,
            present_mode: PresentMode::Fifo,
            alpha_mode: CompositeAlphaMode::Opaque,
            color_space: ColorSpace::Srgb,
            usage: SurfaceUsage::RENDER_ATTACHMENT,
            maximum_frame_latency: 1,
        })
        .find(|config| {
            matches!(
                surface.configuration_support(config),
                fluxel_rhi::api::Support::Supported(())
            )
        })
        .ok_or("Surface has no supported linear RGBA8/BGRA8 opaque FIFO configuration")?;
    runtime::block_on(surface.configure(&config))?;
    Ok(())
}

struct CompanionInput {
    snapshot: FrameSnapshot<fluxel_renderer::ForwardResources>,
    view: RenderViewId,
    visibility: RenderBounds,
    assets: ForwardAssets,
    slot: fluxel_renderer::CaptureTextureSlot,
    receiver: PreparedCaptureReceiver,
}
impl From<&inputs::CaptureSceneInputs> for CompanionInput {
    fn from(input: &inputs::CaptureSceneInputs) -> Self {
        Self {
            snapshot: input.capture_snapshot.clone(),
            view: input.capture_view,
            visibility: input.visibility,
            assets: input.assets.clone(),
            slot: input.slot.clone(),
            receiver: input.receiver.clone(),
        }
    }
}

fn run_companion<B>(
    renderer: &mut Renderer<B>,
    input: CompanionInput,
    output: &Path,
    name: &str,
) -> Result<Vec<u8>>
where
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + RasterBackend
        + PipelineBackend
        + ShaderBackend,
{
    std::fs::write(
        output.join(format!("{name}-companion.json")),
        "{\"schemaVersion\":1,\"role\":\"same-input offscreen numeric companion\",\"surfaceSubstitute\":false}\n",
    )?;
    let submission = runtime::block_on(renderer.submit_forward_capture(
        &ForwardCpuPipeline::new(BoundsVisibility),
        input.snapshot,
        input.view,
        input.visibility,
        &input.assets,
        input.slot,
        input.receiver,
    ))?;
    let rgba = runtime::block_on(renderer.readback(&submission))?;
    std::fs::write(
        output.join(format!("{name}-companion-readback.rgba")),
        &rgba,
    )?;
    if renderer.pending() != 0 {
        return Err(format!("{name} companion readback did not retire its Graph owner").into());
    }
    Ok(rgba)
}

fn submit_visible<B>(
    surface: &mut Surface<B>,
    renderer: &mut Renderer<B>,
    input: inputs::CaptureSceneInputs,
) -> Result
where
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + RasterBackend
        + PipelineBackend
        + ShaderBackend
        + SurfaceBackend,
{
    let image = runtime::block_on(surface.acquire())?;
    let submission = runtime::block_on(renderer.submit_forward_capture_surface(
        &ForwardCpuPipeline::new(BoundsVisibility),
        input.capture_snapshot,
        input.capture_view,
        input.visibility,
        &input.assets,
        input.slot.clone(),
        input.receiver,
        image,
    ))?;
    runtime::block_on(renderer.complete(&submission))?;
    Ok(())
}

fn capture_and_verify(
    window: &Window,
    output: &Path,
    frame: usize,
    expected: &oracle::CaptureOracle,
    color: [u8; 4],
    companion: &[u8],
) -> Result {
    match crate::integration_capture::snapshot_client(window, output, frame, inputs::EXTENT) {
        Ok(()) => verify_window_bmp(output, frame, expected, color, companion),
        Err(error) => {
            std::fs::write(
                output.join("visible-capture-unverified.json"),
                format!(
                    "{{\"schemaVersion\":1,\"method\":\"windows-graphics-capture-window\",\"status\":\"unverified\",\"frame\":{frame},\"reason\":{:?}}}\n",
                    error.to_string()
                ),
            )?;
            Err(
                format!("Windows Graphics Capture unavailable for visible frame {frame}: {error}")
                    .into(),
            )
        }
    }
}

fn verify_window_bmp(
    output: &Path,
    frame: usize,
    expected: &oracle::CaptureOracle,
    color: [u8; 4],
    companion: &[u8],
) -> Result {
    let bytes = std::fs::read(output.join(format!("surface-client-frame-{frame}.bmp")))?;
    if bytes.len() != 54 + (inputs::EXTENT[0] * inputs::EXTENT[1] * 4) as usize
        || &bytes[..2] != b"BM"
    {
        return Err("visible capture BMP shape is invalid".into());
    }
    let sample = |pixel: [u32; 2]| {
        let at = 54 + ((pixel[1] * inputs::EXTENT[0] + pixel[0]) * 4) as usize;
        [bytes[at + 2], bytes[at + 1], bytes[at]]
    };
    for probe in &expected.color_probes {
        if sample(probe.final_pixel) != [color[0], color[1], color[2]] {
            return Err(format!(
                "visible frame {frame} probe {:?} did not retain producer RGB",
                probe.final_pixel
            )
            .into());
        }
    }
    if sample(expected.background_probe.final_pixel) != [0, 0, 0] {
        return Err(format!("visible frame {frame} background probe did not remain black").into());
    }
    let dense = window_oracle::compare_bmp_rgb(&bytes, companion, inputs::EXTENT)?;
    std::fs::write(
        output.join(format!("surface-client-frame-{frame}.dense.json")),
        format!(
            "{{\"schemaVersion\":1,\"companion\":\"same-input offscreen readback\",\"comparison\":\"exact RGB excluding fixed DWM mask\",\"comparedPixels\":{},\"maskedPixels\":{},\"alpha\":\"ignored for opaque Surface\"}}\n",
            dense.compared, dense.masked
        ),
    )?;
    Ok(())
}

fn wait_for_close(window: &Window) -> Result {
    loop {
        if window
            .poll_events()?
            .into_iter()
            .any(|event| matches!(event, fluxel_host::WindowEvent::CloseRequested))
        {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(16));
    }
}
