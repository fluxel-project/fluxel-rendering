//! Native presentation consumer for the S5 Forward path.
//!
//! This workload owns a small public scene and the host-bound surface
//! lifecycle. The Renderer retains every acquired image until `complete`,
//! while this module configures, resizes, and independently captures the
//! native client after each completed presentation.

use std::{collections::BTreeMap, error::Error, fs, path::Path};

use fluxel_assets::{Acquire, AssetSnapshot, AssetStore, ResidentBytes};
use fluxel_material::{
    BindingLocation, MaterialAsset, MaterialDefinition, MaterialFragment, MaterialParameter,
    MaterialResource, MaterialResourceKind, MaterialType, StaticValue, SurfaceOutput,
};
use fluxel_renderer::{
    BoundsVisibility, ColorTextureAsset, ColorTextureData, ForwardAssets, ForwardCpuPipeline,
    ForwardResources, GeometryAsset, GeometryAttribute, GeometryData, GeometryLayout,
    GeometryStream, RenderBounds, RenderChange, RenderChangeSet, RenderRecord, RenderScene,
    RenderTransform, RenderView, Renderer, SceneAssetRef,
};
use fluxel_rhi::{
    api::{Device, Surface, VertexFormat, VertexStepMode},
    backend::{
        ExecutionBackend, MappingBackend, PipelineBackend, RasterBackend, ReadbackBackend,
        ResourceBackend, ShaderBackend, SurfaceBackend,
    },
};

use crate::{block_on, surface};

const INITIAL_EXTENT: [u32; 2] = [320, 180];
const RESIZED_EXTENT: [u32; 2] = [256, 144];
const COLOR: [u8; 4] = [255, 0, 0, 255];

fn geometry(
    store: &AssetStore<GeometryAsset, GeometryData, String>,
) -> Result<AssetSnapshot<GeometryAsset, GeometryData>, Box<dyn Error>> {
    let mut bytes = Vec::new();
    for position in [
        [-0.75_f32, -0.65, 0.0],
        [0.55, -0.50, 0.0],
        [-0.20, 0.75, 0.0],
    ] {
        for value in position {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    let value = GeometryData {
        layout: GeometryLayout {
            streams: vec![GeometryStream {
                stride: 12,
                step_mode: VertexStepMode::Vertex,
                attributes: vec![GeometryAttribute {
                    semantic: "POSITION".into(),
                    offset: 0,
                    format: VertexFormat::Float32(3),
                }],
            }],
        },
        streams: vec![bytes],
        vertices: 0..3,
    };
    let handle = store.create()?;
    match store.acquire(&handle)? {
        Acquire::Producer(permit) => Ok(permit.commit(value, ResidentBytes::new(36))?),
        _ => Err("new Forward surface geometry did not yield a producer".into()),
    }
}

fn material(
    store: &AssetStore<MaterialAsset, MaterialDefinition, String>,
) -> Result<AssetSnapshot<MaterialAsset, MaterialDefinition>, Box<dyn Error>> {
    let value = MaterialDefinition::new(
        vec![MaterialFragment {
            name: "surface_color".into(),
            inputs: Vec::new(),
            output: MaterialType::Vec4,
            calls: Vec::new(),
            wgsl: "fn surface_color() -> vec4<f32> { return textureLoad(material_texture, vec2<i32>(0, 0), 0) * tint; }".into(),
        }],
        "surface_color",
        vec![MaterialParameter {
            name: "tint".into(),
            ty: MaterialType::Vec4,
            resource: "material_tint".into(),
        }],
        vec![
            MaterialResource {
                name: "material_tint".into(),
                kind: MaterialResourceKind::UniformVec4,
                location: BindingLocation { group: 0, binding: 0 },
            },
            MaterialResource {
                name: "material_texture".into(),
                kind: MaterialResourceKind::Texture2dF32,
                location: BindingLocation { group: 0, binding: 1 },
            },
        ],
        BTreeMap::from([("forward_sample_level".into(), StaticValue::U32(0))]),
        SurfaceOutput::Color,
    );
    let handle = store.create()?;
    match store.acquire(&handle)? {
        Acquire::Producer(permit) => Ok(permit.commit(value, ResidentBytes::new(0))?),
        _ => Err("new Forward surface material did not yield a producer".into()),
    }
}

fn texture(
    store: &AssetStore<ColorTextureAsset, ColorTextureData, String>,
) -> Result<AssetSnapshot<ColorTextureAsset, ColorTextureData>, Box<dyn Error>> {
    let handle = store.create()?;
    let value = ColorTextureData {
        extent: [1, 1],
        rgba8: COLOR.to_vec(),
    };
    match store.acquire(&handle)? {
        Acquire::Producer(permit) => Ok(permit.commit(value, ResidentBytes::new(4))?),
        _ => Err("new Forward surface texture did not yield a producer".into()),
    }
}

fn run_frames<B, Resize, Pump, Capture>(
    device: Device<B>,
    mut surface: Surface<B>,
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
    Resize: Fn([u32; 2], &str) -> Result<surface::ResizeFact, Box<dyn Error>>,
    Pump: Fn() -> Result<(), Box<dyn Error>>,
    Capture: Fn(usize, [u32; 2]) -> Result<(), Box<dyn Error>>,
{
    fs::create_dir_all(output)?;
    fs::write(output.join("surface-events.jsonl"), b"")?;
    let geometry_store = AssetStore::<GeometryAsset, GeometryData, String>::new();
    let material_store = AssetStore::<MaterialAsset, MaterialDefinition, String>::new();
    let texture_store = AssetStore::<ColorTextureAsset, ColorTextureData, String>::new();
    let geometry = geometry(&geometry_store)?;
    let material = material(&material_store)?;
    let texture = texture(&texture_store)?;
    fs::write(
        output.join("forward-surface-geometry.bin"),
        &geometry.value().streams[0],
    )?;
    fs::write(
        output.join("forward-surface-texture.rgba"),
        &texture.value().rgba8,
    )?;
    fs::write(
        output.join("forward-surface-expected.json"),
        "{\"triangleRgba\":[255,0,0,255],\"backgroundRgb\":[0,0,0],\"nativeCapture\":true}\n",
    )?;

    let mut scene = RenderScene::new()?;
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::Insert(RenderRecord {
        resources: ForwardResources {
            geometry: SceneAssetRef::from_snapshot(&geometry),
            material: SceneAssetRef::from_snapshot(&material),
            texture: SceneAssetRef::from_snapshot(&texture),
            tint: [1.0; 4],
        },
        transform: RenderTransform::identity(),
        bounds: RenderBounds::new([-1.0; 3], [1.0; 3])?,
    }));
    changes.push(RenderChange::InsertView(RenderView::new(
        RenderTransform::identity(),
        INITIAL_EXTENT,
    )?));
    let receipt = scene.apply(&changes)?;
    let view = receipt.inserted_views()[0];
    let assets = ForwardAssets {
        geometry: vec![geometry],
        materials: vec![material],
        textures: vec![texture],
    };
    let pipeline = ForwardCpuPipeline::new(BoundsVisibility);
    let volume = RenderBounds::new([-1.0; 3], [1.0; 3])?;
    let mut renderer = Renderer::new(device)?;

    let config = surface::configure(&mut surface, INITIAL_EXTENT, output)?;
    let first_image = block_on(surface.acquire())
        .map_err(|error| format!("Forward surface acquire: {error:?}"))?;
    if first_image.format() != config.format || first_image.extent() != config.extent {
        return Err("Forward surface lease facts did not match the configured surface".into());
    }
    let first = block_on(renderer.submit_forward_surface(
        &pipeline,
        scene.snapshot(),
        view,
        volume,
        &assets,
        first_image,
    ))?;
    block_on(renderer.complete(&first))?;
    if renderer.pending() != 0 {
        return Err("Forward surface first frame remained pending".into());
    }
    capture(0, config.extent)?;
    surface::await_visual_release(output, 0, config.extent, &pump)?;

    let resized = resize(RESIZED_EXTENT, "forward-resize")?;
    // A host resize changes the client extent, but it does not by itself define
    // one portable surface-generation transition. Reconfigure before acquiring
    // the next lease so every backend receives the newly observed extent.
    fs::write(
        output.join("forward-surface-reconfigure-input.json"),
        format!(
            "{{\"schemaVersion\":1,\"previousExtent\":[{},{}],\"observedClientExtent\":[{},{}],\"nextOperation\":\"configure\"}}\n",
            config.extent[0], config.extent[1], resized.extent[0], resized.extent[1],
        ),
    )?;
    let config = surface::configure(&mut surface, resized.extent, output)?;
    let mut view_update = RenderChangeSet::new();
    view_update.push(RenderChange::UpdateView(
        view,
        RenderView::new(RenderTransform::identity(), resized.extent)?,
    ));
    scene.apply(&view_update)?;
    let second_image = block_on(surface.acquire())
        .map_err(|error| format!("Forward resized surface acquire: {error:?}"))?;
    if second_image.format() != config.format || second_image.extent() != config.extent {
        return Err(
            "Forward resized surface lease facts did not match the configured surface".into(),
        );
    }
    let second = block_on(renderer.submit_forward_surface(
        &pipeline,
        scene.snapshot(),
        view,
        volume,
        &assets,
        second_image,
    ))?;
    block_on(renderer.complete(&second))?;
    if renderer.pending() != 0 {
        return Err("Forward surface resized frame remained pending".into());
    }
    capture(1, config.extent)?;
    surface::await_visual_release(output, 1, config.extent, &pump)?;

    fs::write(
        output.join("s5-forward-surface-workload.json"),
        format!(
            "{{\"schemaVersion\":1,\"workload\":\"s5-forward-surface-resize\",\"initialExtent\":[{},{}],\"resizedExtent\":[{},{}],\"presentedFrames\":2,\"completionBeforeReconfigure\":true,\"hostResizeReconfiguredBeforeAcquire\":true,\"visualEvidence\":\"native-client-capture\"}}\n",
            INITIAL_EXTENT[0], INITIAL_EXTENT[1], RESIZED_EXTENT[0], RESIZED_EXTENT[1],
        ),
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
            "Fluxel S5 Forward Surface DX12",
            INITIAL_EXTENT[0],
            INITIAL_EXTENT[1],
        )?
        .with_visibility(fluxel_host::WindowVisibility::Hidden),
    )?;
    let raw = window
        .window_handle()
        .map_err(|error| format!("Forward DX12 window handle: {error:?}"))?
        .as_raw();
    // SAFETY: the Host window outlives the surface and every submitted lease.
    let surface = unsafe { fluxel_rhi::dx12::Dx12::surface_from_raw_window_handle(&device, raw) }?;
    run_frames(
        device,
        surface,
        output,
        |extent, phase| surface::resize_window(&window, extent, output, phase),
        || {
            window.poll_events()?;
            Ok(())
        },
        |frame, extent| {
            window.show_background()?;
            crate::capture::capture_client(&window, output, frame, extent, COLOR)
        },
    )
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
pub(crate) fn run_wgl(
    device: Device<fluxel_rhi::gl::Gl>,
    window: &fluxel_host::Window,
    output: &Path,
) -> Result<(), Box<dyn Error>> {
    let surface = fluxel_rhi::gl::Gl::surface(&device)?;
    // WGL needs a mapped drawable before its first visible presentation.
    // This background transition preserves the current foreground window.
    window.show_background()?;
    run_frames(
        device,
        surface,
        output,
        |extent, phase| surface::resize_window(window, extent, output, phase),
        || {
            window.poll_events()?;
            Ok(())
        },
        |frame, extent| {
            window.show_background()?;
            crate::capture::capture_client(window, output, frame, extent, COLOR)
        },
    )
}
