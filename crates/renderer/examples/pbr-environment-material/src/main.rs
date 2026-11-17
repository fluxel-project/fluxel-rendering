//! Native offscreen IBL acceptance entry point.

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
use renderer_pbr_environment_material::{
    oracle::{
        Control, expected_from_input, expected_rgba8, sample_mips_rgba16f, sample_rgba16f,
        verify_rgb,
    },
    scene,
};

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

struct Options {
    pub(crate) mode: Mode,
    pub(crate) exit_after_capture: bool,
    pub(crate) output: PathBuf,
}

fn main() -> Result<(), Box<dyn Error>> {
    let options = parse()?;
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
    let mut output = PathBuf::from("target/renderer-pbr-environment-material");
    let mut args = std::env::args_os().skip(1);
    while let Some(arg) = args.next() {
        match arg.to_string_lossy().as_ref() {
            "--offscreen" => mode = Some(Mode::Offscreen),
            "--visible" => mode = Some(Mode::Visible),
            "--exit-after-capture" => exit_after_capture = true,
            "--output" => output = args.next().ok_or("--output requires a directory")?.into(),
            _ => return Err(format!("unsupported argument: {}", arg.to_string_lossy()).into()),
        }
    }
    Ok(Options {
        mode: mode.ok_or("select exactly one of --offscreen or --visible")?,
        exit_after_capture,
        output,
    })
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
    let controls = controls();
    let mut renderer = Renderer::new(device)?;
    let mut forward = Vec::new();
    let mut deferred = Vec::new();
    for control in controls {
        let input = scene::input(control)?;
        let directory = options.output.join(control.name);
        fs::create_dir_all(&directory)?;
        fs::write(directory.join("geometry.bin"), &input.geometry_bytes)?;
        fs::write(
            directory.join("material.txt"),
            format!("{:#?}\n", input.material),
        )?;
        fs::write(
            directory.join("base-color-metallic.rgba"),
            &input.texture_bytes,
        )?;
        for (index, mip) in input.environment.value().radiance_mips.iter().enumerate() {
            fs::write(
                directory.join(format!("radiance-mip-{index}.rgba16f")),
                &mip.rgba16f_le,
            )?;
        }
        fs::write(
            directory.join("brdf-lut.rgba16f"),
            &input.environment.value().brdf_lut.rgba16f_le,
        )?;
        if let Some(bytes) = &input.diffuse_bytes {
            fs::write(directory.join("diffuse-irradiance.rgba8"), bytes)?;
        }
        let oracle = expected_from_input(control, &input);
        let mips: Vec<_> = input
            .environment
            .value()
            .radiance_mips
            .iter()
            .map(|mip| (mip.rgba16f_le.as_slice(), mip.extent))
            .collect();
        let length = control
            .camera
            .into_iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        let view = control.camera.map(|x| x / length);
        let reflection = [-view[0], -view[1], view[2]];
        let uv = [
            (reflection[2].atan2(reflection[0]) / std::f32::consts::TAU + 0.5).rem_euclid(1.0),
            reflection[1].clamp(-1.0, 1.0).acos() / std::f32::consts::PI,
        ];
        let lod = 2.0 * control.roughness.clamp(0.0, 1.0);
        let radiance = sample_mips_rgba16f(&mips, uv, lod);
        let lut = sample_rgba16f(
            &input.environment.value().brdf_lut.rgba16f_le,
            input.environment.value().brdf_lut.extent,
            [1.5, control.roughness],
            false,
        );
        fs::write(
            directory.join("oracle.json"),
            format!(
                "{{\"schemaVersion\":1,\"case\":\"{}\",\"expectedLinearRgb\":{:?},\"expectedRgba8\":{:?},\"readbackTolerance\":5,\"contract\":\"independent direct GGX plus Babylon height-correlated split-sum; artificial LUT is not a physical BRDF integration\"}}\n",
                control.name,
                oracle,
                expected_rgba8(control, &input)
            ),
        )?;
        fs::write(
            directory.join("oracle-samples.json"),
            format!(
                "{{\"latLongUv\":{:?},\"lod\":{},\"radianceRgba\":{:?},\"lutClampRgba\":{:?}}}\n",
                uv, lod, radiance, lut
            ),
        )?;
        renderer.set_environment_specular_ibl(Some(input.environment.clone()))?;
        renderer.set_environment_irradiance(input.diffuse.clone())?;
        let bounds = RenderBounds::new([-1.; 3], [1.; 3])?;
        let submission = block_on(renderer.submit_forward(
            &ForwardCpuPipeline::new(BoundsVisibility),
            input.snapshot.clone(),
            input.view,
            bounds,
            &input.assets,
        ))?;
        let raw = block_on(renderer.readback(&submission))?;
        fs::write(directory.join("forward-raw.rgba"), &raw)?;
        verify_companion(&raw, expected_rgba8(control, &input))?;
        let submission = block_on(renderer.submit_deferred(
            &ExternalDeferredPipeline::new(),
            input.snapshot.clone(),
            input.view,
            bounds,
            &input.assets,
        ))?;
        let raw_deferred = block_on(renderer.readback(&submission))?;
        fs::write(directory.join("deferred-raw.rgba"), &raw_deferred)?;
        verify_companion(&raw_deferred, expected_rgba8(control, &input))?;
        forward.push(center(&raw)?);
        deferred.push(center(&raw_deferred)?);
    }
    if forward.windows(2).all(|pair| pair[0] == pair[1])
        || deferred.windows(2).all(|pair| pair[0] == pair[1])
    {
        return Err("IBL controls did not alter either actual renderer route".into());
    }
    fs::write(
        options.output.join("result.json"),
        "{\"schemaVersion\":1,\"scope\":\"Forward and Deferred IBL resource and calculation contract\",\"nonGoal\":\"physical LUT generation or environment prefilter quality\"}\n",
    )?;
    Ok(())
}

pub(crate) fn controls() -> [Control; 7] {
    [
        Control {
            name: "dielectric-smooth",
            metallic: 0.,
            roughness: 0.1,
            camera: [0., 0., 1.],
            lowest_mip_only: false,
            lut_variant: false,
            diffuse: false,
        },
        Control {
            name: "metal-smooth",
            metallic: 1.,
            roughness: 0.1,
            camera: [0., 0., 1.],
            lowest_mip_only: false,
            lut_variant: false,
            diffuse: false,
        },
        Control {
            name: "metal-rough",
            metallic: 1.,
            roughness: 0.9,
            camera: [0., 0., 1.],
            lowest_mip_only: false,
            lut_variant: false,
            diffuse: false,
        },
        Control {
            name: "camera-change",
            metallic: 1.,
            roughness: 0.1,
            camera: [1., 0., 1.],
            lowest_mip_only: false,
            lut_variant: false,
            diffuse: false,
        },
        Control {
            name: "lowest-mip",
            metallic: 1.,
            roughness: 0.9,
            camera: [0., 0., 1.],
            lowest_mip_only: true,
            lut_variant: false,
            diffuse: false,
        },
        Control {
            name: "lut-change",
            metallic: 1.,
            roughness: 0.1,
            camera: [0., 0., 1.],
            lowest_mip_only: false,
            lut_variant: true,
            diffuse: false,
        },
        Control {
            name: "diffuse-specular",
            metallic: 0.,
            roughness: 0.4,
            camera: [1., 0., 1.],
            lowest_mip_only: false,
            lut_variant: false,
            diffuse: true,
        },
    ]
}

pub(crate) fn write_visible_input(
    output: &Path,
    control: Control,
    input: &scene::Input,
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(output)?;
    fs::write(output.join("geometry.bin"), &input.geometry_bytes)?;
    fs::write(
        output.join("material.txt"),
        format!("{:#?}\n", input.material),
    )?;
    fs::write(
        output.join("base-color-metallic.rgba"),
        &input.texture_bytes,
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
    if let Some(bytes) = &input.diffuse_bytes {
        fs::write(output.join("diffuse-irradiance.rgba8"), bytes)?;
    }
    fs::write(
        output.join("oracle.json"),
        format!(
            "{{\"schemaVersion\":1,\"case\":\"{}\",\"expectedLinearRgb\":{:?},\"expectedRgba8\":{:?},\"readbackTolerance\":5,\"contract\":\"independent direct GGX plus Babylon height-correlated split-sum\"}}\n",
            control.name,
            expected_from_input(control, input),
            expected_rgba8(control, input)
        ),
    )?;
    Ok(())
}

fn center(raw: &[u8]) -> Result<[u8; 4], Box<dyn Error>> {
    let p = ((scene::EXTENT[1] / 2 * scene::EXTENT[0] + scene::EXTENT[0] / 2) * 4) as usize;
    Ok(raw
        .get(p..p + 4)
        .ok_or("readback has no centre pixel")?
        .try_into()?)
}

pub(crate) fn verify_companion(raw: &[u8], expected: [u8; 4]) -> Result<(), Box<dyn Error>> {
    if raw.len() != (scene::EXTENT[0] * scene::EXTENT[1] * 4) as usize {
        return Err("IBL companion readback extent differs from the saved oracle extent".into());
    }
    for y in [
        scene::EXTENT[1] / 4,
        scene::EXTENT[1] / 2,
        scene::EXTENT[1] * 3 / 4,
    ] {
        for x in [
            scene::EXTENT[0] / 4,
            scene::EXTENT[0] / 2,
            scene::EXTENT[0] * 3 / 4,
        ] {
            let at = ((y * scene::EXTENT[0] + x) * 4) as usize;
            let found: [u8; 4] = raw[at..at + 4].try_into()?;
            verify_rgb(found, expected, 5).map_err(|reason| format!("IBL interior readback at ({x},{y}) differs from independent payload oracle: {reason}"))?;
        }
    }
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
