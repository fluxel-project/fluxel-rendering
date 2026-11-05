//! Persistent, user-visible Renderer demonstrations.
//!
//! The native entry point owns window creation and translates Host events into
//! [`DemoEvent`].  This module deliberately owns only the real Renderer Graph
//! path: every visible frame acquires a `Surface` lease and is submitted by
//! `Renderer`, never by a fixture-specific presentation shortcut.

use std::{
    collections::BTreeMap,
    error::Error,
    fs,
    path::Path,
    thread,
    time::{Duration, Instant},
};

use external_renderer::ExternalDeferredPipeline;
use fluxel_assets::{Acquire, AssetSnapshot, AssetStore, ResidentBytes};
use fluxel_material::{
    BindingLocation, MaterialAsset, MaterialDefinition, MaterialFragment, MaterialParameter,
    MaterialResource, MaterialResourceKind, MaterialType, StaticValue, SurfaceOutput,
};
use fluxel_renderer::{
    BoundsVisibility, ColorTextureAsset, ColorTextureData, ForwardAssets, ForwardCpuPipeline,
    ForwardResources, GeometryAsset, GeometryAttribute, GeometryData, GeometryLayout,
    GeometryStream, RenderBounds, RenderChange, RenderChangeSet, RenderId, RenderRecord,
    RenderScene, RenderTransform, RenderView, RenderViewId, Renderer, SceneAssetRef,
};
use fluxel_rhi::{
    api::{Device, Surface, VertexFormat, VertexStepMode},
    backend::{
        ExecutionBackend, MappingBackend, PipelineBackend, RasterBackend, ReadbackBackend,
        ResourceBackend, ShaderBackend, SurfaceBackend,
    },
};

use crate::{block_on, demo_evidence, surface};

/// The selectable presentation route.  Both variants submit to a real Surface.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DemoRoute {
    Forward,
    Deferred,
}

/// Lifecycle facts the native entry point must forward from the Host window.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DemoEvent {
    CloseRequested,
    Resized([u32; 2]),
    Minimized,
    Restored([u32; 2]),
    SurfaceDestroyed,
    SurfaceCreated,
    Suspended,
    Resumed,
    RedrawRequested,
}

pub(crate) fn evidence_enabled() -> bool {
    demo_evidence::enabled()
}

const FRAME_INTERVAL: Duration = Duration::from_millis(16);

fn visibility() -> RenderBounds {
    RenderBounds::new([-1.0; 3], [1.0; 3]).expect("demo uses fixed finite visibility bounds")
}

/// Drives an already-created native surface until the user closes its window.
///
/// `pump` must be non-blocking and return all pending lifecycle events.  The
/// `after_frame` runs only after the completed surface presentation. A zero
/// extent, minimize, suspend, or destroyed drawable idles the renderer and
/// never attempts a zero-sized configuration.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run<B, Pump, AfterFrame>(
    device: Device<B>,
    mut target: Surface<B>,
    route: DemoRoute,
    initial_extent: [u32; 2],
    output: &Path,
    mut pump: Pump,
    mut after_frame: AfterFrame,
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
    Pump: FnMut() -> Result<Vec<DemoEvent>, Box<dyn Error>>,
    AfterFrame: FnMut(usize, [u32; 2]) -> Result<(), Box<dyn Error>>,
{
    if initial_extent.contains(&0) {
        return Err("demo initial client extent must be non-zero".into());
    }
    fs::create_dir_all(output)?;

    let assets = Assets::new()?;
    if demo_evidence::enabled() {
        assets.write_evidence(output)?;
    }
    let mut scene = RenderScene::new()?;
    let (record, view) = install_scene(&mut scene, &assets, initial_extent)?;
    let mut renderer = Renderer::new(device)?;
    let forward = ForwardCpuPipeline::new(BoundsVisibility);
    let deferred = ExternalDeferredPipeline::new();
    let evidence = demo_evidence::enabled();
    let mut state = LoopState::new(initial_extent);
    let started = Instant::now();
    let mut frame = 0usize;

    while !state.closed {
        for event in pump()? {
            state.observe(event);
        }
        if state.closed {
            break;
        }
        if !state.ready() {
            thread::sleep(FRAME_INTERVAL);
            continue;
        }
        if state.reconfigure {
            // Each previous frame is completed below before this next loop
            // iteration, so configuration never invalidates a retained lease.
            surface::configure(&mut target, state.extent, output)?;
            update_view(&mut scene, view, state.extent)?;
            state.reconfigure = false;
        }

        let elapsed = if evidence {
            Duration::from_millis(u64::try_from(frame)? * 100)
        } else {
            started.elapsed()
        };
        let animation = update_record(&mut scene, record, &assets, elapsed)?;
        let lease = block_on(target.acquire())
            .map_err(|error| format!("demo surface acquire: {error:?}"))?;
        if lease.extent() != state.extent {
            return Err(format!(
                "demo lease extent {:?} differs from configured {:?}",
                lease.extent(),
                state.extent
            )
            .into());
        }
        let snapshot = scene.snapshot();
        if evidence {
            demo_evidence::write_input(
                route,
                &snapshot,
                frame,
                state.extent,
                animation.transform.columns(),
                animation.tint,
                output,
            )?;
        }
        let submission = match route {
            DemoRoute::Forward => block_on(renderer.submit_forward_surface(
                &forward,
                snapshot.clone(),
                view,
                visibility(),
                &assets.forward(),
                lease,
            ))?,
            DemoRoute::Deferred => block_on(renderer.submit_deferred_surface(
                &deferred,
                snapshot.clone(),
                view,
                visibility(),
                &assets.forward(),
                lease,
            ))?,
        };
        block_on(renderer.complete(&submission))?;
        if renderer.pending() != 0 {
            return Err("demo completion left a Surface lease pending".into());
        }
        if evidence {
            demo_evidence::readback(
                route,
                &mut renderer,
                &forward,
                &deferred,
                snapshot,
                view,
                visibility(),
                &assets.forward(),
                frame,
                output,
            )?;
        }
        after_frame(frame, state.extent)?;
        if evidence && frame == demo_evidence::LAST_FRAME {
            break;
        }
        frame = frame.checked_add(1).ok_or("demo frame identifier overflow")?;
        thread::sleep(FRAME_INTERVAL);
    }
    Ok(())
}

struct LoopState {
    extent: [u32; 2],
    drawable: bool,
    suspended: bool,
    reconfigure: bool,
    closed: bool,
}

impl LoopState {
    fn new(extent: [u32; 2]) -> Self {
        Self {
            extent,
            drawable: true,
            suspended: false,
            reconfigure: true,
            closed: false,
        }
    }

    fn observe(&mut self, event: DemoEvent) {
        match event {
            DemoEvent::CloseRequested => self.closed = true,
            DemoEvent::Resized(extent) | DemoEvent::Restored(extent) => {
                self.extent = extent;
                self.suspended = extent.contains(&0);
                self.reconfigure = !extent.contains(&0);
            }
            DemoEvent::Minimized | DemoEvent::Suspended => self.suspended = true,
            DemoEvent::Resumed => self.suspended = self.extent.contains(&0),
            DemoEvent::SurfaceDestroyed => self.drawable = false,
            DemoEvent::SurfaceCreated => {
                self.drawable = true;
                self.reconfigure = !self.extent.contains(&0);
            }
            DemoEvent::RedrawRequested => {}
        }
    }

    fn ready(&self) -> bool {
        self.drawable && !self.suspended && !self.extent.contains(&0)
    }
}

fn install_scene(
    scene: &mut RenderScene<ForwardResources>,
    assets: &Assets,
    extent: [u32; 2],
) -> Result<(RenderId, RenderViewId), Box<dyn Error>> {
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::Insert(
        assets.record(RenderTransform::identity(), [1.0; 4]),
    ));
    changes.push(RenderChange::InsertView(RenderView::new(
        RenderTransform::identity(),
        extent,
    )?));
    let receipt = scene.apply(&changes)?;
    Ok((receipt.inserted()[0], receipt.inserted_views()[0]))
}

fn update_view(
    scene: &mut RenderScene<ForwardResources>,
    view: RenderViewId,
    extent: [u32; 2],
) -> Result<(), Box<dyn Error>> {
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::UpdateView(
        view,
        RenderView::new(RenderTransform::identity(), extent)?,
    ));
    scene.apply(&changes)?;
    Ok(())
}

struct Animation {
    transform: RenderTransform,
    tint: [f32; 4],
}

fn update_record(
    scene: &mut RenderScene<ForwardResources>,
    id: RenderId,
    assets: &Assets,
    elapsed: Duration,
) -> Result<Animation, Box<dyn Error>> {
    let seconds = elapsed.as_secs_f32();
    let angle = seconds * 0.8;
    let (sin, cos) = angle.sin_cos();
    let transform = RenderTransform::from_columns([
        [cos * 0.72, sin * 0.72, 0.0, 0.0],
        [-sin * 0.72, cos * 0.72, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])?;
    let tint = [
        0.55 + 0.45 * (seconds * 0.9).sin().abs(),
        0.35 + 0.65 * (seconds * 0.6).cos().abs(),
        0.45 + 0.55 * (seconds * 0.4).sin().abs(),
        1.0,
    ];
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::Update(id, assets.record(transform, tint)));
    scene.apply(&changes)?;
    Ok(Animation { transform, tint })
}

struct Assets {
    geometry: AssetSnapshot<GeometryAsset, GeometryData>,
    material: AssetSnapshot<MaterialAsset, MaterialDefinition>,
    texture: AssetSnapshot<ColorTextureAsset, ColorTextureData>,
}

impl Assets {
    fn new() -> Result<Self, Box<dyn Error>> {
        let geometries = AssetStore::new();
        let materials = AssetStore::new();
        let textures = AssetStore::new();
        Ok(Self {
            geometry: geometry(&geometries)?,
            material: material(&materials)?,
            texture: texture(&textures)?,
        })
    }

    fn forward(&self) -> ForwardAssets {
        ForwardAssets {
            geometry: vec![self.geometry.clone()],
            materials: vec![self.material.clone()],
            textures: vec![self.texture.clone()],
        }
    }

    fn write_evidence(&self, output: &Path) -> Result<(), Box<dyn Error>> {
        fs::write(
            output.join("demo-positions.bin"),
            &self.geometry.value().streams[0],
        )?;
        fs::write(
            output.join("demo-geometry.txt"),
            format!("{:#?}\n", self.geometry.value()),
        )?;
        fs::write(
            output.join("demo-material.txt"),
            format!("{:#?}\n", self.material.value()),
        )?;
        fs::write(
            output.join("demo-texture.rgba"),
            &self.texture.value().rgba8,
        )?;
        Ok(())
    }

    fn record(&self, transform: RenderTransform, tint: [f32; 4]) -> RenderRecord<ForwardResources> {
        RenderRecord {
            resources: ForwardResources {
                geometry: SceneAssetRef::from_snapshot(&self.geometry),
                material: SceneAssetRef::from_snapshot(&self.material),
                texture: SceneAssetRef::from_snapshot(&self.texture),
                tint,
            },
            transform,
            bounds: RenderBounds::new([-0.9; 3], [0.9; 3])
                .expect("demo uses fixed finite triangle bounds"),
        }
    }
}

fn geometry(
    store: &AssetStore<GeometryAsset, GeometryData, String>,
) -> Result<AssetSnapshot<GeometryAsset, GeometryData>, Box<dyn Error>> {
    let mut bytes = Vec::with_capacity(36);
    for position in [
        [-0.72_f32, -0.62, 0.5],
        [0.72, -0.56, 0.5],
        [0.0, 0.78, 0.5],
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
        _ => Err("demo geometry producer missing".into()),
    }
}

fn material(
    store: &AssetStore<MaterialAsset, MaterialDefinition, String>,
) -> Result<AssetSnapshot<MaterialAsset, MaterialDefinition>, Box<dyn Error>> {
    let value = MaterialDefinition::new(
        vec![MaterialFragment {
            name: "demo_color".into(),
            inputs: Vec::new(),
            output: MaterialType::Vec4,
            calls: Vec::new(),
            wgsl: "fn demo_color() -> vec4<f32> { return textureLoad(material_texture, vec2<i32>(0, 0), 0) * tint; }".into(),
        }],
        "demo_color",
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
        BTreeMap::from([("demo_sample_level".into(), StaticValue::U32(0))]),
        SurfaceOutput::Color,
    );
    let handle = store.create()?;
    match store.acquire(&handle)? {
        Acquire::Producer(permit) => Ok(permit.commit(value, ResidentBytes::new(0))?),
        _ => Err("demo material producer missing".into()),
    }
}

fn texture(
    store: &AssetStore<ColorTextureAsset, ColorTextureData, String>,
) -> Result<AssetSnapshot<ColorTextureAsset, ColorTextureData>, Box<dyn Error>> {
    let handle = store.create()?;
    let value = ColorTextureData {
        extent: [1, 1],
        rgba8: vec![232, 106, 44, 255],
    };
    match store.acquire(&handle)? {
        Acquire::Producer(permit) => Ok(permit.commit(value, ResidentBytes::new(4))?),
        _ => Err("demo texture producer missing".into()),
    }
}
