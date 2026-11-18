//! Native Surface presentation with same-input offscreen companions.

use super::*;
use fluxel_host::{Window, WindowConfig, WindowVisibility};
use fluxel_rhi::{
    api::{
        ColorSpace, CompositeAlphaMode, DeviceRequest, PresentMode, Surface, SurfaceConfig,
        SurfaceUsage, TextureFormat,
    },
    backend::SurfaceBackend,
};

#[cfg(feature = "dx12")]
pub(super) fn run_dx12(options: Options) -> Result<(), Box<dyn Error>> {
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
        Mode::Offscreen => run_offscreen(device, &options),
        Mode::Visible => {
            use raw_window_handle::HasWindowHandle;
            let window = host_window()?;
            let raw = window
                .window_handle()
                .map_err(|error| format!("visible DX12 window handle: {error:?}"))?
                .as_raw();
            let surface =
                unsafe { fluxel_rhi::dx12::Dx12::surface_from_raw_window_handle(&device, raw) }?;
            visible(device, surface, &window, &options, capture.as_ref())
        }
    }
}

#[cfg(feature = "native-gl-wgl")]
pub(super) fn run_wgl(options: Options) -> Result<(), Box<dyn Error>> {
    use fluxel_rhi::{
        api::{AdapterId, Instance, InstanceId},
        gl::{Gl, GlInstance},
    };
    let capture = visible_capture(&options)?;
    let window = host_window()?;
    let native = unsafe { GlInstance::from_wgl_window(&window, scene::EXTENT) }?;
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
    let observed = device.clone();
    let mut observer = crate::diagnostics::WglObserver::install(&observed)?;
    let result = match options.mode {
        Mode::Offscreen => run_offscreen(device, &options),
        Mode::Visible => (|| {
            // This WGL driver requires a mapped drawable before its first present.
            window.show_background()?;
            visible(
                device.clone(),
                Gl::surface(&device)?,
                &window,
                &options,
                capture.as_ref(),
            )
        })(),
    };
    result.and(observer.write(&observed, &options.output))
}

fn visible_capture(
    options: &Options,
) -> Result<Option<crate::integration_capture::CaptureRuntime>, Box<dyn Error>> {
    if matches!(options.mode, Mode::Offscreen) {
        return Ok(None);
    }
    unsafe {
        use windows::Win32::UI::HiDpi::{
            DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, SetProcessDpiAwarenessContext,
        };
        SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)?;
    }
    fs::write(
        options.output.join("visible-dpi.json"),
        "{\"schemaVersion\":1,\"processDpiAwareness\":\"per-monitor-aware-v2\",\"scope\":\"before HWND, WinRT, and native GPU objects\"}\n",
    )?;
    Ok(Some(crate::integration_capture::CaptureRuntime::new()?))
}

fn host_window() -> Result<Window, Box<dyn Error>> {
    Ok(Window::new(
        WindowConfig::new(
            "Fluxel PBR sphere material",
            scene::EXTENT[0],
            scene::EXTENT[1],
        )?
        .with_visibility(WindowVisibility::Hidden),
    )?)
}

fn configure<B: SurfaceBackend>(surface: &mut Surface<B>) -> Result<(), Box<dyn Error>> {
    let caps = surface.capabilities()?;
    let config = [TextureFormat::Bgra8Unorm, TextureFormat::Rgba8Unorm]
        .into_iter()
        .filter(|format| caps.formats.contains(format))
        .map(|format| SurfaceConfig {
            extent: scene::EXTENT,
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
        .ok_or("Surface has no supported opaque FIFO RGBA8 configuration")?;
    block_on(surface.configure(&config))?;
    Ok(())
}

fn visible<B>(
    device: Device<B>,
    mut surface: Surface<B>,
    window: &Window,
    options: &Options,
    capture: Option<&crate::integration_capture::CaptureRuntime>,
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
    configure(&mut surface)?;
    crate::window_oracle::write_contract(&options.output, scene::EXTENT)?;
    write_device_capabilities(&device, &options.output)?;
    let mut renderer = Renderer::new(device)?;
    configure_forward_samples(&mut renderer, options)?;
    let bounds = RenderBounds::new([-5.; 3], [5.; 3])?;
    let mut frame = 0;
    for case in cases() {
        let input = case_input(options, &case)?;
        let probes = oracle::reference(&input)?;
        let output = options.output.join(case.name);
        write_input(&output, &input, &probes)?;
        renderer.set_environment_specular_ibl(Some(input.environment.clone()))?;
        renderer.set_environment_irradiance(Some(input.diffuse.clone()))?;
        let shown = block_on(renderer.submit_forward_surface(
            &ForwardCpuPipeline::new(BoundsVisibility).with_depth32_less_equal(),
            input.snapshot.clone(),
            input.view,
            bounds,
            &input.assets,
            block_on(surface.acquire())?,
        ))?;
        block_on(renderer.complete(&shown))?;
        window.show_background()?;
        let raw = read_offscreen(&mut renderer, &input, Route::Forward)?;
        fs::write(output.join("visible-forward-companion-readback.rgba"), &raw)?;
        verify_readback(&raw, &probes, Route::Forward)?;
        capture_and_compare(window, &output, frame, &raw)?;
        frame += 1;
        let shown = block_on(renderer.submit_deferred_surface(
            &ExternalDeferredPipeline::new(),
            input.snapshot.clone(),
            input.view,
            bounds,
            &input.assets,
            block_on(surface.acquire())?,
        ))?;
        block_on(renderer.complete(&shown))?;
        let raw = read_offscreen(&mut renderer, &input, Route::Deferred)?;
        fs::write(
            output.join("visible-deferred-companion-readback.rgba"),
            &raw,
        )?;
        verify_readback(&raw, &probes, Route::Deferred)?;
        capture_and_compare(window, &output, frame, &raw)?;
        frame += 1;
    }
    fs::write(
        options.output.join("visible-result.json"),
        "{\"schemaVersion\":1,\"surface\":\"Forward and Deferred acquired Renderer Surface leases\",\"companion\":\"same-input offscreen submissions\",\"comparison\":\"dense exact RGB except fixed 512-pixel DWM mask\"}\n",
    )?;
    if !options.exit_after_capture {
        wait_for_close(window)?;
    }
    Ok(())
}

fn capture_and_compare(
    window: &Window,
    output: &std::path::Path,
    frame: usize,
    raw: &[u8],
) -> Result<(), Box<dyn Error>> {
    crate::integration_capture::snapshot_client(window, output, frame, scene::EXTENT)?;
    let bmp = fs::read(output.join(format!("surface-client-frame-{frame}.bmp")))?;
    crate::window_oracle::compare_bmp_rgb(&bmp, raw, scene::EXTENT)
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
        std::thread::sleep(std::time::Duration::from_millis(16));
    }
}
