//! Native device opening plus audited background presentation.

use super::*;
use fluxel_host::{Window, WindowConfig, WindowVisibility};
use fluxel_rhi::{
    api::{
        ColorSpace, CompositeAlphaMode, DeviceRequest, PresentMode, Surface, SurfaceConfig,
        SurfaceUsage, TextureFormat,
    },
    backend::SurfaceBackend,
};
use std::time::Duration;

#[cfg(feature = "dx12")]
pub(super) fn run_dx12(options: Options, texture: TextureInput) -> Result<(), Box<dyn Error>> {
    use fluxel_rhi::dx12::Dx12Instance;
    let capture = visible_capture(&options)?;
    let native = Dx12Instance::new()?;
    let instance = native.instance()?;
    let adapter = instance
        .enumerate()?
        .into_iter()
        .next()
        .ok_or("no DX12 adapter")?;
    let device = block_on(instance.open(DeviceRequest {
        adapter: adapter.id,
        required_features: Vec::new(),
    }))?;
    crate::diagnostics::record_dx12_device_open(&options.output)?;
    match options.mode {
        Mode::Offscreen => {
            run_offscreen(device, &texture, options.scene_options(), &options.output)
        }
        Mode::Visible => {
            use raw_window_handle::HasWindowHandle;
            let window = host_window()?;
            let raw = window
                .window_handle()
                .map_err(|error| format!("visible DX12 window handle: {error:?}"))?
                .as_raw();
            let surface =
                unsafe { fluxel_rhi::dx12::Dx12::surface_from_raw_window_handle(&device, raw) }?;
            present_scene(
                device,
                surface,
                &window,
                &texture,
                options.scene_options(),
                &options.output,
                capture.as_ref(),
                options.exit_after_capture,
            )
        }
    }
}

#[cfg(feature = "native-gl-wgl")]
pub(super) fn run_wgl(options: Options, texture: TextureInput) -> Result<(), Box<dyn Error>> {
    use fluxel_rhi::{
        api::{AdapterId, Instance, InstanceId},
        gl::{Gl, GlInstance},
    };
    let capture = visible_capture(&options)?;
    let window = host_window()?;
    let native = unsafe { GlInstance::from_wgl_window(&window, EXTENT) }?;
    let id = InstanceId::allocate()?;
    let instance = unsafe { Instance::<Gl>::from_native(id, native) };
    let adapter = instance
        .enumerate()?
        .into_iter()
        .next()
        .ok_or("no WGL adapter")?;
    let device = block_on(instance.open(DeviceRequest {
        adapter: AdapterId {
            instance: id,
            ordinal: adapter.id.ordinal,
        },
        required_features: Vec::new(),
    }))?;
    let observed_device = device.clone();
    let mut observer = crate::diagnostics::WglObserver::install(&observed_device)?;
    let result = match options.mode {
        Mode::Offscreen => {
            run_offscreen(device, &texture, options.scene_options(), &options.output)
        }
        Mode::Visible => (|| {
            // This driver needs a mapped drawable before its first presentation; it stays non-activating.
            window.show_background()?;
            let surface = Gl::surface(&device)?;
            present_scene(
                device,
                surface,
                &window,
                &texture,
                options.scene_options(),
                &options.output,
                capture.as_ref(),
                options.exit_after_capture,
            )
        })(),
    };
    let diagnostics = observer.write(&observed_device, &options.output);
    result.and(diagnostics)
}

fn visible_capture(
    options: &Options,
) -> Result<Option<crate::integration_capture::CaptureRuntime>, Box<dyn Error>> {
    if matches!(options.mode, Mode::Offscreen) {
        return Ok(None);
    }
    unsafe {
        use windows::Win32::UI::HiDpi::{
            SetProcessDpiAwarenessContext, DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2,
        };
        SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)?;
    }
    fs::write(
        options.output.join("visible-dpi.json"),
        "{\"schemaVersion\":1,\"processDpiAwareness\":\"per-monitor-aware-v2\",\"scope\":\"fixture process only; configured before WinRT/HWND/native GPU objects\"}\n",
    )?;
    Ok(Some(crate::integration_capture::CaptureRuntime::new()?))
}

fn host_window() -> Result<Window, Box<dyn Error>> {
    Ok(Window::new(
        WindowConfig::new("Fluxel Renderer 02 textured Material", EXTENT[0], EXTENT[1])?
            .with_visibility(WindowVisibility::Hidden),
    )?)
}

fn present_scene<B>(
    device: Device<B>,
    mut surface: Surface<B>,
    window: &Window,
    texture: &TextureInput,
    scene_options: SceneOptions,
    output: &Path,
    capture: Option<&crate::integration_capture::CaptureRuntime>,
    exit_after_capture: bool,
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
{
    let _capture = capture.ok_or("visible mode requires initialized Windows Graphics Capture")?;
    configure_surface(&mut surface)?;
    let scene = scene(texture, scene_options)?;
    let bounds = RenderBounds::new([-1.; 3], [1.; 3])?;
    let expected = expected_probes(texture, 1.0);
    write_expected_probes(output, "forward", texture, &expected, 1)?;
    write_expected_probes(output, "deferred", texture, &expected, 1)?;
    crate::window_oracle::write_contract(output, EXTENT)?;
    fs::write(
        output.join("visible-input-contract.json"),
        "{\"schemaVersion\":1,\"routes\":[\"Forward\",\"Deferred\"],\"companion\":\"same snapshot/assets submitted offscreen after presented route\",\"rawBeforeOracle\":true}\n",
    )?;
    let mut renderer = Renderer::new(device)?;
    let forward = block_on(renderer.submit_forward_surface(
        &ForwardCpuPipeline::new(BoundsVisibility),
        scene.snapshot.clone(),
        scene.view,
        bounds,
        &scene.assets,
        block_on(surface.acquire())?,
    ))?;
    block_on(renderer.complete(&forward))?;
    window.show_background()?;
    let forward_raw = read_forward(&mut renderer, &scene, bounds)?;
    fs::write(
        output.join("visible-forward-companion-readback.rgba"),
        &forward_raw,
    )?;
    verify_window(window, output, 0, &forward_raw, &expected)?;
    let deferred = block_on(renderer.submit_deferred_surface(
        &DeferredCpuPipeline::new(BoundsVisibility),
        scene.snapshot.clone(),
        scene.view,
        bounds,
        &scene.assets,
        block_on(surface.acquire())?,
    ))?;
    block_on(renderer.complete(&deferred))?;
    let deferred_raw = read_deferred(&mut renderer, &scene, bounds)?;
    fs::write(
        output.join("visible-deferred-companion-readback.rgba"),
        &deferred_raw,
    )?;
    verify_window(window, output, 1, &deferred_raw, &expected)?;
    fs::write(
        output.join("visible-result.json"),
        "{\"schemaVersion\":1,\"surface\":\"Forward then Deferred through acquired Renderer Surface leases\",\"window\":\"shown in background without activation\",\"visualCapture\":\"captured-and-dense-RGB-verified\"}\n",
    )?;
    if !exit_after_capture {
        wait_for_close(window)?;
    }
    Ok(())
}

fn configure_surface<B: SurfaceBackend>(surface: &mut Surface<B>) -> Result<(), Box<dyn Error>> {
    let caps = surface.capabilities()?;
    let config = [TextureFormat::Bgra8Unorm, TextureFormat::Rgba8Unorm]
        .into_iter()
        .filter(|format| caps.formats.contains(format))
        .map(|format| SurfaceConfig {
            extent: EXTENT,
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
    block_on(surface.configure(&config))?;
    Ok(())
}

fn read_forward<B>(
    renderer: &mut Renderer<B>,
    scene: &Scene,
    bounds: RenderBounds,
) -> Result<Vec<u8>, Box<dyn Error>>
where
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + RasterBackend
        + PipelineBackend
        + ShaderBackend,
{
    let submission = block_on(renderer.submit_forward(
        &ForwardCpuPipeline::new(BoundsVisibility),
        scene.snapshot.clone(),
        scene.view,
        bounds,
        &scene.assets,
    ))?;
    Ok(block_on(renderer.readback(&submission))?)
}

fn read_deferred<B>(
    renderer: &mut Renderer<B>,
    scene: &Scene,
    bounds: RenderBounds,
) -> Result<Vec<u8>, Box<dyn Error>>
where
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + RasterBackend
        + PipelineBackend
        + ShaderBackend,
{
    let submission = block_on(renderer.submit_deferred(
        &DeferredCpuPipeline::new(BoundsVisibility),
        scene.snapshot.clone(),
        scene.view,
        bounds,
        &scene.assets,
    ))?;
    Ok(block_on(renderer.readback(&submission))?)
}

fn verify_window(
    window: &Window,
    output: &Path,
    frame: usize,
    raw: &[u8],
    expected: &[Probe],
) -> Result<(), Box<dyn Error>> {
    crate::integration_capture::snapshot_client(window, output, frame, EXTENT)?;
    let bmp = fs::read(output.join(format!("surface-client-frame-{frame}.bmp")))?;
    let dense = crate::window_oracle::compare_bmp_rgb(&bmp, raw, EXTENT)?;
    verify_probes(raw, expected, 1)?;
    for probe in expected {
        // The predeclared DWM mask applies to window pixels only. Every
        // numerical KTX probe, including these corners, was checked above
        // against the independently retained RHI companion.
        if crate::window_oracle::masked([probe.x, probe.y], EXTENT) {
            continue;
        }
        let at = ((probe.y * EXTENT[0] + probe.x) * 4) as usize;
        if bmp[54 + at + 2].abs_diff(probe.expected[0]) > 1
            || bmp[54 + at + 1].abs_diff(probe.expected[1]) > 1
            || bmp[54 + at].abs_diff(probe.expected[2]) > 1
        {
            return Err(format!(
                "visible frame {frame} KTX probe ({}, {}) differs from independent expected RGBA",
                probe.x, probe.y
            )
            .into());
        }
    }
    fs::write(
        output.join(format!("surface-client-frame-{frame}.dense.json")),
        format!(
            "{{\"schemaVersion\":1,\"companion\":\"same-input offscreen readback\",\"comparison\":\"exact RGB excluding fixed DWM mask\",\"comparedPixels\":{},\"maskedPixels\":{},\"alpha\":\"ignored for opaque Surface\"}}\n",
            dense.compared, dense.masked
        ),
    )?;
    Ok(())
}

fn wait_for_close(window: &Window) -> Result<(), Box<dyn Error>> {
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
