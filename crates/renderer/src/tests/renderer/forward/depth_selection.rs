//! Owner facts for the bounded Forward depth selection.

use super::*;
use super::{
    backend::{Native, Script, WitnessBackend},
    support::{block_on, fixture, fixture_with_caps},
};

use std::{cell::Cell, collections::BTreeMap, sync::atomic::Ordering};

use fluxel_assets::{Acquire, AssetStore, ResidentBytes};
use fluxel_material::{
    BindingLocation, MaterialAsset, MaterialDefinition, MaterialFragment, MaterialParameter,
    MaterialResource, MaterialResourceKind, MaterialType, SurfaceOutput,
};
use fluxel_rhi::api::{AccessMask, StageMask, VertexFormat, VertexStepMode};

use crate::{
    BoundsVisibility, ColorTextureAsset, ColorTextureData, ForwardAssets, ForwardCpuPipeline,
    ForwardDepthMode, ForwardResources, FramePipeline, FrameSnapshot, FrameTopology, GeometryAsset,
    GeometryAttribute, GeometryData, GeometryLayout, GeometryStream, PipelineError, PreparedFrame,
    RenderBounds, RenderChange, RenderChangeSet, RenderRecord, RenderScene, RenderTransform,
    RenderView, Renderer, RendererError, SceneAssetRef,
};

pub(super) fn bounds() -> RenderBounds {
    RenderBounds::new([-1.0; 3], [1.0; 3]).expect("test bounds")
}

pub(super) fn forward_device(script: Script) -> (Device<WitnessBackend>, Arc<Native>) {
    fixture_with_caps(script, |caps| {
        // A material-backed Forward pipeline owns transform and material
        // groups; the generic lifecycle fixture deliberately exposes one.
        caps.limits.max_bind_groups = 2;
        caps.limits.max_bindings_per_group = 3;
        caps.limits.max_vertex_buffers = 1;
        caps.limits.max_vertex_attributes = 1;
    })
}

fn material() -> MaterialDefinition {
    MaterialDefinition::new(
        vec![MaterialFragment {
            name: "tinted_texture".into(),
            inputs: Vec::new(),
            output: MaterialType::Vec4,
            calls: Vec::new(),
            wgsl: "fn tinted_texture() -> vec4<f32> { return textureLoad(material_texture, vec2<i32>(0, 0), 0) * tint; }".into(),
        }],
        "tinted_texture",
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
        BTreeMap::new(),
        SurfaceOutput::Color,
    )
}

pub(super) fn scene_and_assets() -> (
    FrameSnapshot<ForwardResources>,
    crate::RenderViewId,
    ForwardAssets,
) {
    let geometries = AssetStore::<GeometryAsset, GeometryData, ()>::new();
    let geometry_handle = geometries.create().expect("geometry handle");
    let geometry = match geometries
        .acquire(&geometry_handle)
        .expect("geometry producer")
    {
        Acquire::Producer(producer) => producer
            .commit(
                GeometryData {
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
                    streams: vec![vec![0; 36]],
                    vertices: 0..3,
                },
                ResidentBytes::new(36),
            )
            .expect("geometry snapshot"),
        _ => unreachable!(),
    };
    let materials = AssetStore::<MaterialAsset, MaterialDefinition, ()>::new();
    let material_handle = materials.create().expect("material handle");
    let material = match materials
        .acquire(&material_handle)
        .expect("material producer")
    {
        Acquire::Producer(producer) => producer
            .commit(material(), ResidentBytes::new(0))
            .expect("material snapshot"),
        _ => unreachable!(),
    };
    let textures = AssetStore::<ColorTextureAsset, ColorTextureData, ()>::new();
    let texture_handle = textures.create().expect("texture handle");
    let texture = match textures.acquire(&texture_handle).expect("texture producer") {
        Acquire::Producer(producer) => producer
            .commit(
                ColorTextureData {
                    extent: [1, 1],
                    encoding: crate::TextureEncoding::Linear,
                    mip_levels: 1,
                    rgba8: vec![1, 2, 3, 255],
                },
                ResidentBytes::new(4),
            )
            .expect("texture snapshot"),
        _ => unreachable!(),
    };
    let mut scene = RenderScene::<ForwardResources>::new().expect("scene");
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::InsertView(
        RenderView::new(RenderTransform::identity(), [8, 8]).expect("view"),
    ));
    changes.push(RenderChange::Insert(RenderRecord {
        resources: ForwardResources {
            geometry: SceneAssetRef::from_snapshot(&geometry),
            material: SceneAssetRef::from_snapshot(&material),
            textures: vec![crate::ForwardTextureBinding::asset(
                "material_texture",
                SceneAssetRef::from_snapshot(&texture),
            )],
            samplers: Vec::new(),
            tint: [1.0; 4],
        },
        transform: RenderTransform::identity(),
        bounds: bounds(),
    }));
    let receipt = scene.apply(&changes).expect("scene insert");
    (
        scene.snapshot(),
        receipt.inserted_views()[0],
        ForwardAssets {
            geometry: vec![geometry],
            materials: vec![material],
            textures: vec![texture],
        },
    )
}

struct CountingDepthPipeline {
    calls: Cell<usize>,
}

impl FramePipeline<ForwardResources> for CountingDepthPipeline {
    fn forward_depth_mode(&self, _: &PreparedFrame<'_, ForwardResources>) -> ForwardDepthMode {
        self.calls.set(self.calls.get() + 1);
        ForwardDepthMode::Depth32LessEqualWrite
    }

    fn prepare<'a>(
        &self,
        snapshot: &'a FrameSnapshot<ForwardResources>,
        view: crate::RenderViewId,
        visibility: RenderBounds,
    ) -> std::result::Result<PreparedFrame<'a, ForwardResources>, PipelineError> {
        ForwardCpuPipeline::new(BoundsVisibility).prepare(snapshot, view, visibility)
    }
}

struct InvalidClearDepthPipeline;

impl FramePipeline<ForwardResources> for InvalidClearDepthPipeline {
    fn forward_depth_mode(&self, _: &PreparedFrame<'_, ForwardResources>) -> ForwardDepthMode {
        ForwardDepthMode::Depth32LessEqualWrite
    }

    fn prepare<'a>(
        &self,
        snapshot: &'a FrameSnapshot<ForwardResources>,
        view: crate::RenderViewId,
        visibility: RenderBounds,
    ) -> std::result::Result<PreparedFrame<'a, ForwardResources>, PipelineError> {
        ForwardCpuPipeline::new(BoundsVisibility).prepare(snapshot, view, visibility)
    }

    fn topology(
        &self,
        _: &PreparedFrame<'_, ForwardResources>,
    ) -> std::result::Result<FrameTopology, PipelineError> {
        Ok(FrameTopology::ClearOnly)
    }
}

#[test]
fn forward_depth_owner_admits_exact_depth32_attachment_before_allocation() {
    let (device, native) = fixture(Script::AcceptedExact);
    let depth = crate::residency::ForwardDepthAttachment::allocate(&device, [7, 5])
        .expect("witness supports one depth attachment");

    assert_eq!(native.created_textures.load(Ordering::SeqCst), 1);
    assert_eq!(depth.desc.format, TextureFormat::Depth32Float);
    assert_eq!(depth.desc.extent, [7, 5, 1]);
    assert_eq!(depth.desc.samples, 1);
    assert_eq!(depth.desc.usage, TextureUsage::RENDER_ATTACHMENT);
    assert_eq!(depth.device, device.id());
    assert_eq!(depth.layout, ImageLayout::Undefined);
    assert_eq!(depth.incoming.stages, StageMask::EMPTY);
    assert_eq!(depth.incoming.accesses, AccessMask::EMPTY);
}

#[test]
fn forward_depth_owner_refuses_zero_extent_before_backend_allocation() {
    let (device, native) = fixture(Script::AcceptedExact);
    assert!(matches!(
        crate::residency::ForwardDepthAttachment::allocate(&device, [0, 5]),
        Err(Error::InvalidRange)
    ));
    assert_eq!(native.created_textures.load(Ordering::SeqCst), 0);
}

#[test]
fn forward_depth_owner_refuses_unsupported_attachment_before_creation() {
    let (device, native) = fixture(Script::AcceptedExact);
    native.depth_supported.store(false, Ordering::SeqCst);
    assert!(matches!(
        crate::residency::ForwardDepthAttachment::allocate(&device, [7, 5]),
        Err(Error::Unsupported("witness depth attachment"))
    ));
    assert_eq!(native.created_textures.load(Ordering::SeqCst), 0);
}

#[test]
fn selected_depth_is_queried_once_and_records_depth32_less_equal_pipeline() {
    let (device, native) = forward_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).expect("renderer");
    let (snapshot, view, assets) = scene_and_assets();
    let pipeline = CountingDepthPipeline {
        calls: Cell::new(0),
    };

    let submission =
        block_on(renderer.submit_forward(&pipeline, snapshot, view, bounds(), &assets))
            .expect("depth Forward submit");

    assert_eq!(
        pipeline.calls.get(),
        1,
        "depth hook is sampled once per frame"
    );
    assert!(
        native
            .pipeline_depth_states
            .lock()
            .unwrap()
            .contains(&Some((
                TextureFormat::Depth32Float,
                true,
                CompareFunction::LessEqual,
            ))),
        "actual pipeline must carry the selected depth state"
    );
    assert_eq!(renderer.pending(), 1);
    let bytes = block_on(renderer.readback(&submission)).expect("complete depth Forward");
    assert_eq!(bytes.len(), 8 * 8 * 4);
}

#[test]
fn disabled_forward_allocates_no_depth_attachment_or_depth_pipeline_state() {
    let (device, native) = forward_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).expect("renderer");
    let (snapshot, view, assets) = scene_and_assets();
    let submission = block_on(renderer.submit_forward(
        &ForwardCpuPipeline::new(BoundsVisibility),
        snapshot,
        view,
        bounds(),
        &assets,
    ))
    .expect("disabled Forward submit");

    assert!(
        native
            .pipeline_depth_states
            .lock()
            .unwrap()
            .iter()
            .all(Option::is_none)
    );
    let disabled_textures = native.created_textures.load(Ordering::SeqCst);
    block_on(renderer.readback(&submission)).expect("complete disabled Forward");

    let (depth_device, depth_native) = forward_device(Script::AcceptedExact);
    let mut depth_renderer = Renderer::new(depth_device).expect("depth renderer");
    let (depth_snapshot, depth_view, depth_assets) = scene_and_assets();
    let depth_submission = block_on(depth_renderer.submit_forward(
        &ForwardCpuPipeline::new(BoundsVisibility).with_depth32_less_equal(),
        depth_snapshot,
        depth_view,
        bounds(),
        &depth_assets,
    ))
    .expect("depth Forward submit");
    assert_eq!(
        depth_native.created_textures.load(Ordering::SeqCst),
        disabled_textures + 1,
        "only the enabled route owns a Depth32Float attachment"
    );
    block_on(depth_renderer.readback(&depth_submission)).expect("complete depth Forward");
}

#[test]
fn clear_only_depth_is_rejected_before_any_gpu_allocation() {
    let (device, native) = forward_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).expect("renderer");
    let (snapshot, view, assets) = scene_and_assets();

    let error = block_on(renderer.submit_forward(
        &InvalidClearDepthPipeline,
        snapshot,
        view,
        bounds(),
        &assets,
    ))
    .expect_err("ClearOnly cannot request an opaque depth attachment");

    assert!(matches!(error, RendererError::Rhi(Error::InvalidState)));
    assert_eq!(native.created_buffers.load(Ordering::SeqCst), 0);
    assert_eq!(native.created_textures.load(Ordering::SeqCst), 0);
    assert!(native.pipeline_depth_states.lock().unwrap().is_empty());
}

#[test]
fn completion_device_loss_keeps_depth_frame_pending() {
    let (device, native) = forward_device(Script::AcceptedCompletionDeviceLost);
    let mut renderer = Renderer::new(device).expect("renderer");
    let (snapshot, view, assets) = scene_and_assets();
    let submission = block_on(renderer.submit_forward(
        &ForwardCpuPipeline::new(BoundsVisibility).with_depth32_less_equal(),
        snapshot,
        view,
        bounds(),
        &assets,
    ))
    .expect("depth Forward submit");

    assert_eq!(renderer.pending(), 1);
    assert!(matches!(
        block_on(renderer.readback(&submission)),
        Err(RendererError::Rhi(Error::DeviceLost))
    ));
    assert_eq!(
        renderer.pending(),
        1,
        "failed completion retains depth owner"
    );
    assert!(native.created_textures.load(Ordering::SeqCst) > 0);
}
