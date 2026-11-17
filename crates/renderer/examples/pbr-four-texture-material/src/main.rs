//! Real-window and independent readback acceptance for four named textures.
use fluxel_renderer::{
    BoundsVisibility, DeferredCpuPipeline, ForwardCpuPipeline, RenderBounds, Renderer,
};
use fluxel_rhi::{
    api::Device,
    backend::{
        ExecutionBackend, MappingBackend, PipelineBackend, RasterBackend, ReadbackBackend,
        ResourceBackend, ShaderBackend,
    },
};
use renderer_pbr_four_texture_material::{mip_generation, mixed_capture, oracle, scene};
use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
};
#[cfg(all(windows, any(feature = "dx12", feature = "native-gl-wgl")))]
#[path = "../../rendering-integration/src/diagnostics.rs"]
mod diagnostics;
mod evidence;
#[cfg(all(windows, any(feature = "dx12", feature = "native-gl-wgl")))]
#[allow(dead_code)]
#[path = "../../rendering-integration/src/capture.rs"]
mod integration_capture;
#[cfg(all(windows, any(feature = "dx12", feature = "native-gl-wgl")))]
mod native;
#[cfg(all(windows, any(feature = "dx12", feature = "native-gl-wgl")))]
mod window_oracle;
use evidence::write_input;
#[derive(Clone, Copy)]
enum Mode {
    Offscreen,
    Visible,
}
#[derive(Clone, Copy)]
enum Route {
    Forward,
    Deferred,
}
struct Options {
    mode: Mode,
    mixed_capture: bool,
    mip_generation: bool,
    exit_after_capture: bool,
    output: PathBuf,
}
fn main() -> Result<(), Box<dyn Error>> {
    let options = parse()?;
    if options.output.exists() && fs::read_dir(&options.output)?.next().is_some() {
        return Err("output must be a new or empty attempt directory".into());
    }
    fs::create_dir_all(&options.output)?;
    #[cfg(all(windows, feature = "dx12"))]
    return native::run_dx12(options);
    #[cfg(all(windows, not(feature = "dx12"), feature = "native-gl-wgl"))]
    return native::run_wgl(options);
    #[cfg(not(all(windows, any(feature = "dx12", feature = "native-gl-wgl"))))]
    Err("select native dx12 or native-gl-wgl on Windows".into())
}
fn parse() -> Result<Options, Box<dyn Error>> {
    let mut mode = None;
    let mut exit_after_capture = false;
    let mut mixed_capture = false;
    let mut mip_generation = false;
    let mut output = PathBuf::from("target/renderer-four-texture-material");
    let mut args = std::env::args_os().skip(1);
    while let Some(arg) = args.next() {
        match arg.to_string_lossy().as_ref() {
            "--visible" | "--offscreen" => {
                if mode.is_some() {
                    return Err("choose one rendering mode".into());
                }
                mode = Some(if arg == "--visible" {
                    Mode::Visible
                } else {
                    Mode::Offscreen
                });
            }
            "--exit-after-capture" => exit_after_capture = true,
            "--mixed-capture" => mixed_capture = true,
            "--mip-generation" => mip_generation = true,
            "--output" => output = args.next().ok_or("--output requires a directory")?.into(),
            _ => return Err(format!("unsupported argument: {}", arg.to_string_lossy()).into()),
        }
    }
    if mixed_capture && mip_generation {
        return Err("--mixed-capture and --mip-generation are separate acceptance modes".into());
    }
    Ok(Options {
        mode: mode.ok_or("select --offscreen or --visible")?,
        mixed_capture,
        mip_generation,
        exit_after_capture,
        output,
    })
}
fn cases() -> [scene::Case; 8] {
    scene::Case::ALL
}
fn case_input(_options: &Options, case: &scene::Case) -> Result<scene::Input, Box<dyn Error>> {
    scene::input(*case)
}

fn run_offscreen<B>(device: Device<B>, options: &Options) -> Result<(), Box<dyn Error>>
where
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + RasterBackend
        + PipelineBackend
        + ShaderBackend,
{
    write_device_capabilities(&device, &options.output)?;
    let mut renderer = Renderer::new(device)?;
    if options.mixed_capture {
        return run_mixed_offscreen(&mut renderer, options);
    }
    if options.mip_generation {
        return run_mip_offscreen(&mut renderer, options);
    }
    for case in cases() {
        let input = scene::input(case)?;
        let probes = oracle::reference(&input)?;
        let output = options.output.join(case.name());
        write_input(&output, &input, &probes)?;
        renderer.set_environment_irradiance(Some(input.diffuse.clone()))?;
        for (route, name) in [(Route::Forward, "forward"), (Route::Deferred, "deferred")] {
            let raw = read_offscreen(&mut renderer, &input, route)?;
            fs::write(output.join(format!("{name}-readback.rgba")), &raw)?;
            verify_readback(&raw, &probes, route)?;
        }
    }
    fs::write(
        options.output.join("result.json"),
        "{\"scope\":\"two records with four named textures, source controls, alias and independent numerical oracle\",\"result\":\"passed\"}\n",
    )?;
    Ok(())
}

/// The first 09 slice: one actual ColorUv0 consumer declares a three-level
/// Linear chain and selects every level using `textureSampleLevel`.
fn run_mip_offscreen<B>(renderer: &mut Renderer<B>, options: &Options) -> Result<(), Box<dyn Error>>
where
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + RasterBackend
        + PipelineBackend
        + ShaderBackend,
{
    for case in mip_generation::Case::ALL {
        let input = mip_generation::input(case)?;
        let probes = mip_generation::reference(&input)?;
        let output = options.output.join(format!("mip-{}", case.name()));
        mip_generation::write_inputs(&output, &input, &probes)?;
        for (route, name) in [(Route::Forward, "forward"), (Route::Deferred, "deferred")] {
            let raw = read_mip_offscreen(renderer, &input, route)?;
            fs::write(output.join(format!("{name}-readback.rgba")), &raw)?;
            mip_generation::verify(&raw, &probes)?;
        }
    }
    fs::write(
        options.output.join("result.json"),
        "{\"scope\":\"GPU-generated Linear RGBA8 mip chain with explicit LOD Forward/Deferred consumer\",\"result\":\"passed\"}\n",
    )?;
    Ok(())
}

fn read_mip_offscreen<B>(
    renderer: &mut Renderer<B>,
    input: &mip_generation::Input,
    route: Route,
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
    let bounds = RenderBounds::new([-1.; 3], [1.; 3])?;
    let submission = match route {
        Route::Forward => block_on(renderer.submit_forward(
            &ForwardCpuPipeline::new(BoundsVisibility).with_depth32_less_equal(),
            input.snapshot.clone(),
            input.view,
            bounds,
            &input.assets,
        ))?,
        Route::Deferred => block_on(renderer.submit_deferred(
            &DeferredCpuPipeline::new(BoundsVisibility),
            input.snapshot.clone(),
            input.view,
            bounds,
            &input.assets,
        ))?,
    };
    let raw = block_on(renderer.readback(&submission))?;
    if renderer.pending() != 0 {
        return Err("mip completion did not retire submitted owners".into());
    }
    Ok(raw)
}

/// Runs the Forward-only mixed-source product. The saved CPU reference is an
/// oracle artifact only; the receiver's emissive texture is always the Graph
/// producer target submitted below.
fn run_mixed_offscreen<B>(
    renderer: &mut Renderer<B>,
    options: &Options,
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
    let mut previous = None;
    for (label, producer_case, receiver_case) in mixed_capture::CONTROLS {
        let input = mixed_capture::input_with_receiver(producer_case, receiver_case)?;
        let output = options.output.join(format!("mixed-{label}"));
        fs::create_dir_all(&output)?;
        let reference = mixed_capture::cpu_capture_reference(&input)?;
        let probes = mixed_capture::reference(&input)?;
        mixed_capture::write_inputs(&output, &input, &probes)?;
        fs::write(output.join("cpu-producer-reference.rgba"), &reference)?;
        fs::write(
            output.join("mixed-probes.json"),
            format!(
                "{{\"count\":{},\"tolerance\":{}}}\n",
                probes.len(),
                oracle::TOLERANCE
            ),
        )?;
        fs::write(
            output.join("mixed-contract.json"),
            format!(
                "{{\"scope\":\"Forward-only PBR receiver: base/normal/orm assets plus GPU capture emissive\",\"slot\":{},\"producerCase\":{}}}\n",
                json_string(mixed_capture::SLOT),
                json_string(label)
            ),
        )?;
        renderer.set_environment_irradiance(Some(input.producer.diffuse.clone()))?;
        let submission = block_on(renderer.submit_forward_capture(
            &ForwardCpuPipeline::new(BoundsVisibility).with_depth32_less_equal(),
            input.producer.snapshot.clone(),
            input.producer.view,
            input.visibility,
            &input.assets,
            input.slot,
            input.receiver,
        ))?;
        let raw = block_on(renderer.readback(&submission))?;
        fs::write(output.join("forward-mixed-readback.rgba"), &raw)?;
        mixed_capture::verify(&raw, &probes)?;
        if renderer.pending() != 0 {
            return Err("mixed capture completion did not retire owners".into());
        }
        if raw.iter().all(|&value| value == 0) {
            return Err("mixed receiver output is unexpectedly clear".into());
        }
        if previous.as_ref() == Some(&raw) {
            return Err("mixed producer/receiver control did not change output".into());
        }
        previous = Some(raw);
    }
    fs::write(
        options.output.join("result.json"),
        "{\"scope\":\"Forward-only same-Graph mixed PBR capture consumer\",\"result\":\"passed\"}\n",
    )?;
    Ok(())
}
fn read_offscreen<B>(
    renderer: &mut Renderer<B>,
    input: &scene::Input,
    route: Route,
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
    let bounds = RenderBounds::new([-5.; 3], [5.; 3])?;
    let submission = match route {
        Route::Forward => block_on(renderer.submit_forward(
            &ForwardCpuPipeline::new(BoundsVisibility).with_depth32_less_equal(),
            input.snapshot.clone(),
            input.view,
            bounds,
            &input.assets,
        ))?,
        Route::Deferred => block_on(renderer.submit_deferred(
            &DeferredCpuPipeline::new(BoundsVisibility),
            input.snapshot.clone(),
            input.view,
            bounds,
            &input.assets,
        ))?,
    };
    let raw = block_on(renderer.readback(&submission))?;
    if renderer.pending() != 0 {
        return Err("completion did not retire submitted owners".into());
    }
    Ok(raw)
}
fn verify_readback(
    raw: &[u8],
    probes: &[oracle::Probe],
    route: Route,
) -> Result<(), Box<dyn Error>> {
    if raw.len() != scene::EXTENT[0] as usize * scene::EXTENT[1] as usize * 4 {
        return Err("readback byte length mismatch".into());
    }
    for probe in probes {
        let expected = match route {
            Route::Forward => probe.forward,
            Route::Deferred => probe.deferred,
        };
        let offset = ((probe.y * scene::EXTENT[0] + probe.x) * 4) as usize;
        let found = &raw[offset..offset + 4];
        if found[..3]
            .iter()
            .zip(expected)
            .any(|(a, b)| a.abs_diff(b) > oracle::TOLERANCE)
        {
            return Err(format!(
                "probe ({},{}) found {found:?}, expected {expected:?}, tolerance {}",
                probe.x,
                probe.y,
                oracle::TOLERANCE
            )
            .into());
        }
    }
    Ok(())
}
fn write_device_capabilities<B: ResourceBackend>(
    device: &Device<B>,
    output: &Path,
) -> Result<(), Box<dyn Error>> {
    let limits = device.capabilities().limits;
    fs::write(
        output.join("device-limits.json"),
        format!(
            "{{\"maxColorAttachments\":{},\"maxBindGroups\":{},\"maxBindingsPerGroup\":{},\"maxVertexBuffers\":{},\"maxVertexAttributes\":{},\"uniformOffsetAlignment\":{}}}\n",
            limits.max_color_attachments,
            limits.max_bind_groups,
            limits.max_bindings_per_group,
            limits.max_vertex_buffers,
            limits.max_vertex_attributes,
            limits.uniform_offset_alignment
        ),
    )?;
    Ok(())
}
fn json_string(value: &str) -> String {
    format!("{value:?}")
}
fn block_on<F: std::future::Future>(future: F) -> F::Output {
    use std::{
        sync::Arc,
        task::{Context, Poll, Wake, Waker},
    };
    struct ThreadWake(std::thread::Thread);
    impl Wake for ThreadWake {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = Waker::from(Arc::new(ThreadWake(std::thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(x) => return x,
            Poll::Pending => std::thread::park(),
        }
    }
}
