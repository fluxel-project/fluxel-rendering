//! Source-retaining native acceptance for perspective ORM/IBL sphere geometry.

use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
};

use external_renderer::ExternalDeferredPipeline;
use fluxel_renderer::{BoundsVisibility, ForwardCpuPipeline, RenderBounds, Renderer};
use fluxel_rhi::{
    api::Device,
    backend::{
        ExecutionBackend, MappingBackend, PipelineBackend, RasterBackend, ReadbackBackend,
        ResourceBackend, ShaderBackend,
    },
};
use renderer_pbr_sphere_material::{oracle, scene};

#[cfg(all(windows, any(feature = "dx12", feature = "native-gl-wgl")))]
#[path = "../../rendering-integration/src/diagnostics.rs"]
mod diagnostics;
#[cfg(all(windows, any(feature = "dx12", feature = "native-gl-wgl")))]
#[allow(dead_code)]
#[path = "../../rendering-integration/src/capture.rs"]
mod integration_capture;
#[cfg(all(windows, any(feature = "dx12", feature = "native-gl-wgl")))]
mod native;
#[cfg(all(windows, any(feature = "dx12", feature = "native-gl-wgl")))]
mod window_oracle;

#[derive(Clone, Copy)]
pub(crate) enum Mode {
    Offscreen,
    Visible,
}
#[derive(Clone, Copy)]
pub(crate) enum Route {
    Forward,
    Deferred,
}

pub(crate) struct Options {
    pub(crate) mode: Mode,
    pub(crate) exit_after_capture: bool,
    pub(crate) output: PathBuf,
    pub(crate) multi_records: bool,
    /// Requested ordinary-Forward sample count.  The sample count is applied
    /// to the exact Renderer used by both offscreen and Surface Forward work.
    pub(crate) forward_samples: u32,
}
pub(crate) struct Case {
    pub(crate) name: &'static str,
    pub(crate) ao_variant: bool,
    pub(crate) eye: [f32; 3],
}
pub(crate) fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "baseline",
            ao_variant: false,
            eye: scene::EYE,
        },
        Case {
            name: "ao-only",
            ao_variant: true,
            eye: scene::EYE,
        },
        Case {
            name: "perspective-eye",
            ao_variant: false,
            eye: [0.35, 0.35, 8.5],
        },
    ]
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
    Err("select --features dx12 or native-gl-wgl on Windows".into())
}

fn parse() -> Result<Options, Box<dyn Error>> {
    let mut mode = None;
    let mut exit_after_capture = false;
    let mut multi_records = false;
    let mut forward_samples = 1;
    let mut output = PathBuf::from("target/renderer-pbr-sphere-material");
    let mut args = std::env::args_os().skip(1);
    while let Some(arg) = args.next() {
        match arg.to_string_lossy().as_ref() {
            "--offscreen" | "--visible" => {
                if mode.is_some() {
                    return Err("select exactly one rendering mode".into());
                }
                mode = Some(if arg == "--visible" {
                    Mode::Visible
                } else {
                    Mode::Offscreen
                });
            }
            "--exit-after-capture" => exit_after_capture = true,
            "--multi-records" => multi_records = true,
            "--samples" => {
                let value = args
                    .next()
                    .ok_or("--samples requires an unsigned integer")?;
                forward_samples = value
                    .to_string_lossy()
                    .parse()
                    .map_err(|_| "--samples requires an unsigned integer")?;
            }
            "--output" => output = args.next().ok_or("--output requires a directory")?.into(),
            _ => return Err(format!("unsupported argument: {}", arg.to_string_lossy()).into()),
        }
    }
    Ok(Options {
        mode: mode.ok_or("select --offscreen or --visible")?,
        exit_after_capture,
        output,
        multi_records,
        forward_samples,
    })
}

pub(crate) fn case_input(options: &Options, case: &Case) -> Result<scene::Input, Box<dyn Error>> {
    if options.multi_records {
        scene::multi_input(case.ao_variant, case.eye)
    } else {
        scene::input(case.ao_variant, case.eye)
    }
}

pub(crate) fn run_offscreen<B>(device: Device<B>, options: &Options) -> Result<(), Box<dyn Error>>
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
    configure_forward_samples(&mut renderer, options)?;
    for case in cases() {
        let input = case_input(options, &case)?;
        let probes = oracle::reference(&input)?;
        let output = options.output.join(case.name);
        write_input(&output, &input, &probes)?;
        renderer.set_environment_specular_ibl(Some(input.environment.clone()))?;
        renderer.set_environment_irradiance(Some(input.diffuse.clone()))?;
        for (name, route) in [("forward", Route::Forward), ("deferred", Route::Deferred)] {
            let raw = read_offscreen(&mut renderer, &input, route)?;
            fs::write(output.join(format!("{name}-raw.rgba")), &raw)?;
            verify_readback(&raw, &probes, route)?;
        }
    }
    fs::write(
        options.output.join("result.json"),
        "{\"scope\":\"perspective ORM and split-sum IBL sphere grid\",\"numericOracle\":\"independent saved-input triangle raster and PBR\",\"nonGoal\":\"unrestricted mixed Deferred surface families or physical environment prefilter generation\"}\n",
    )?;
    Ok(())
}

pub(crate) fn configure_forward_samples<B>(
    renderer: &mut Renderer<B>,
    options: &Options,
) -> Result<(), Box<dyn Error>>
where
    B: ExecutionBackend + ResourceBackend + MappingBackend + ReadbackBackend,
{
    renderer.set_forward_sample_count(options.forward_samples)?;
    fs::write(
        options.output.join("forward-samples.json"),
        format!(
            "{{\"schemaVersion\":1,\"requestedForwardSamples\":{},\"configuredForwardSamples\":{},\"forwardScope\":\"ordinary Forward offscreen and Surface submissions\",\"deferredBaselineSamples\":1,\"deferredPolicy\":\"the paired Deferred route remains a single-sample baseline\"}}\n",
            options.forward_samples,
            renderer.forward_sample_count(),
        ),
    )?;
    Ok(())
}

pub(crate) fn read_offscreen<B>(
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
            &ExternalDeferredPipeline::new(),
            input.snapshot.clone(),
            input.view,
            bounds,
            &input.assets,
        ))?,
    };
    block_on(renderer.readback(&submission)).map_err(Into::into)
}

pub(crate) fn verify_readback(
    raw: &[u8],
    probes: &[oracle::Probe],
    route: Route,
) -> Result<(), Box<dyn Error>> {
    oracle::verify(raw, probes, matches!(route, Route::Deferred)).map_err(Into::into)
}

pub(crate) fn write_input(
    output: &Path,
    input: &scene::Input,
    probes: &[oracle::Probe],
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(output)?;
    fs::write(output.join("geometry.bin"), &input.geometry_bytes)?;
    let mut sources = Vec::new();
    for (index, source) in input.object_sources.iter().enumerate() {
        let geometry = &input.assets.geometry[source.geometry_index];
        let texture = &input.assets.textures[source.texture_index];
        let material = &input.assets.materials[source.material_index];
        for (stream, bytes) in geometry.value().streams.iter().enumerate() {
            fs::write(
                output.join(format!("object-{index}-stream-{stream}.bin")),
                bytes,
            )?;
        }
        fs::write(
            output.join(format!("object-{index}-texture.rgba8")),
            &texture.value().rgba8,
        )?;
        fs::write(
            output.join(format!("object-{index}-material.txt")),
            format!("{:#?}\n", material.value()),
        )?;
        sources.push(format!(
            "{{\"object\":{},\"geometryIndex\":{},\"materialIndex\":{},\"textureIndex\":{},\"textureExtent\":{:?},\"worldFromLocalColumns\":{:?},\"vertices\":[{},{}],\"assetIdentities\":{}}}",
            index, source.geometry_index, source.material_index, source.texture_index,
            texture.value().extent, source.world_from_local, geometry.value().vertices.start,
            geometry.value().vertices.end, json_string(&format!("geometry {:?}/{}; material {:?}/{}; texture {:?}/{}",
                geometry.id(), geometry.generation().get(), material.id(), material.generation().get(),
                texture.id(), texture.generation().get())),
        ));
    }
    fs::write(
        output.join("object-sources.json"),
        format!(
            "{{\"objects\":[{}],\"canonicalGeometry\":\"world-space reference; actual upload bytes are the individual object streams\"}}\n",
            sources.join(",\n"),
        ),
    )?;
    fs::write(output.join("orm.rgba8"), &input.orm_bytes)?;
    fs::write(
        output.join("diffuse-irradiance.rgba8"),
        &input.diffuse.value().rgba8,
    )?;
    fs::write(
        output.join("material.txt"),
        format!("{:#?}\n", input.material),
    )?;
    fs::write(
        output.join("asset-identities.txt"),
        format!(
            "geometry {:?} generation {}\nmaterial {:?} generation {}\norm {:?} generation {}\nradiance-lut {:?} generation {}\ndiffuse {:?} generation {}\n",
            input.assets.geometry[0].id(),
            input.assets.geometry[0].generation().get(),
            input.assets.materials[0].id(),
            input.assets.materials[0].generation().get(),
            input.assets.textures[0].id(),
            input.assets.textures[0].generation().get(),
            input.environment.id(),
            input.environment.generation().get(),
            input.diffuse.id(),
            input.diffuse.generation().get(),
        ),
    )?;
    for (index, mip) in input.environment.value().radiance_mips.iter().enumerate() {
        fs::write(
            output.join(format!("radiance-mip-{index}.rgba16f")),
            &mip.rgba16f_le,
        )?;
    }
    fs::write(
        output.join("brdf-lut.rgba16f"),
        &input.environment.value().brdf_lut.rgba16f_le,
    )?;
    fs::write(
        output.join("input.json"),
        format!(
            "{{\"schemaVersion\":1,\"extent\":{:?},\"eye\":{:?},\"clipFromWorldColumns\":{:?},\"tint\":{:?},\"vertexStrideBytes\":48,\"vertexFields\":\"position3 normal3 uv2 tangent4\",\"sceneRecordCount\":{},\"ormChannels\":\"R occlusion, G perceptual roughness, B metallic; A ignored\",\"radianceMipExtents\":{:?},\"lutExtent\":{:?},\"lodScale\":{},\"lodOffset\":{},\"forwardDepth\":\"Depth32Float clear1 LessEqual write CullNone\"}}\n",
            scene::EXTENT,
            input.eye,
            input.vp,
            input.tint,
            input.object_sources.len(),
            input
                .environment
                .value()
                .radiance_mips
                .iter()
                .map(|m| m.extent)
                .collect::<Vec<_>>(),
            input.environment.value().brdf_lut.extent,
            input.environment.value().lod.scale,
            input.environment.value().lod.offset,
        ),
    )?;
    let records = probes
        .iter()
        .map(|probe| probe.to_json())
        .collect::<Vec<_>>()
        .join(",\n");
    fs::write(
        output.join("oracle.json"),
        format!(
            "{{\"schemaVersion\":1,\"rgbTolerance\":{},\"probes\":[{}]}}\n",
            oracle::TOLERANCE,
            records,
        ),
    )?;
    Ok(())
}

pub(crate) fn write_device_capabilities<B: ResourceBackend>(
    device: &Device<B>,
    output: &Path,
) -> Result<(), Box<dyn Error>> {
    let limits = device.capabilities().limits;
    fs::write(
        output.join("device-limits.json"),
        format!(
            "{{\"scope\":\"selected consumer numeric limits; concrete format support is queried before allocation\",\"maxColorAttachments\":{},\"maxBindGroups\":{},\"maxBindingsPerGroup\":{},\"maxVertexBuffers\":{},\"maxVertexAttributes\":{},\"uniformOffsetAlignment\":{},\"maxBufferBytes\":{}}}\n",
            limits.max_color_attachments,
            limits.max_bind_groups,
            limits.max_bindings_per_group,
            limits.max_vertex_buffers,
            limits.max_vertex_attributes,
            limits.uniform_offset_alignment,
            limits.max_buffer_bytes,
        ),
    )?;
    Ok(())
}

fn json_string(value: &str) -> String {
    format!("{value:?}")
}

fn block_on<F: std::future::Future>(future: F) -> F::Output {
    use std::{
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
        task::{Context, Poll, Wake, Waker},
    };
    struct Wakeup {
        ready: AtomicBool,
        thread: std::thread::Thread,
    }
    impl Wake for Wakeup {
        fn wake(self: Arc<Self>) {
            self.ready.store(true, Ordering::Release);
            self.thread.unpark();
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.ready.store(true, Ordering::Release);
            self.thread.unpark();
        }
    }
    let wake = Arc::new(Wakeup {
        ready: AtomicBool::new(true),
        thread: std::thread::current(),
    });
    let waker = Waker::from(wake.clone());
    let mut context = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
            return value;
        }
        while !wake.ready.swap(false, Ordering::AcqRel) {
            std::thread::park();
        }
    }
}
