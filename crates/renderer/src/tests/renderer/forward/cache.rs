//! Forward residency, device-isolation, and frame-pool tests.

use super::*;

use crate::{
    ColorTextureAsset, ColorTextureData, ForwardAssets, GeometryAsset, GeometryData,
    GeometryLayout,
    renderer::forward::ForwardPublication,
    residency::{
        ForwardFrame, ForwardFrameBuffers, ForwardResidentBuffers, ForwardResidentCache,
        ForwardStreamInput, ForwardTextureInput,
    },
};
use fluxel_assets::{Acquire, AssetStore, Production, ResidentBytes};

fn blank_frame(device: &Device<WitnessBackend>) -> ForwardFrame<WitnessBackend> {
    let (resident, resident_upload) =
        ForwardResidentBuffers::allocate(device, Vec::new(), Vec::new()).unwrap();
    let uniform_bytes = vec![0; 16];
    let (frame, uniform_upload) =
        ForwardFrameBuffers::allocate(device, [1, 1], 16, uniform_bytes.clone(), 0..16, 256)
            .unwrap();
    ForwardFrame {
        resident,
        resident_upload,
        frame,
        uniform_upload,
        uniform_bytes,
        draws: Vec::new(),
        draw_batches: Vec::new(),
        extent: [1, 1],
        readback_row_pitch: 256,
    }
}

fn execute(
    renderer: &mut Renderer<WitnessBackend>,
    retained: crate::renderer::PendingInputs,
    frame: ForwardFrame<WitnessBackend>,
) -> std::result::Result<Submission, RendererError> {
    block_on(renderer.execute_forward(retained, frame, ForwardPublication::default()))
}

#[test]
fn completed_forward_cache_keys_exact_assets_and_device_owners() {
    let (device, _) = fixture(Script::AcceptedExact);
    let (other_device, _) = fixture(Script::AcceptedExact);

    let geometry_store = AssetStore::<GeometryAsset, GeometryData, ()>::new();
    let geometry_handle = geometry_store.create().unwrap();
    let geometry = match geometry_store.acquire(&geometry_handle).unwrap() {
        Acquire::Producer(producer) => producer
            .commit(
                GeometryData {
                    layout: GeometryLayout::default(),
                    streams: Vec::new(),
                    vertices: 0..0,
                },
                ResidentBytes::new(0),
            )
            .unwrap(),
        _ => unreachable!("new asset must issue a producer"),
    };
    let geometry_replacement = match geometry_store
        .request_replacement(&geometry_handle)
        .unwrap()
    {
        Production::Producer(producer) => producer
            .commit(
                GeometryData {
                    layout: GeometryLayout::default(),
                    streams: Vec::new(),
                    vertices: 0..0,
                },
                ResidentBytes::new(0),
            )
            .unwrap(),
        Production::Waiter(_) => unreachable!(),
    };

    let texture_store = AssetStore::<ColorTextureAsset, ColorTextureData, ()>::new();
    let texture_handle = texture_store.create().unwrap();
    let texture = match texture_store.acquire(&texture_handle).unwrap() {
        Acquire::Producer(producer) => producer
            .commit(
                ColorTextureData {
                    extent: [1, 1],
                    rgba8: vec![1, 2, 3, 4],
                },
                ResidentBytes::new(4),
            )
            .unwrap(),
        _ => unreachable!("new asset must issue a producer"),
    };
    let texture_replacement = match texture_store.request_replacement(&texture_handle).unwrap() {
        Production::Producer(producer) => producer
            .commit(
                ColorTextureData {
                    extent: [1, 1],
                    rgba8: vec![5, 6, 7, 8],
                },
                ResidentBytes::new(4),
            )
            .unwrap(),
        Production::Waiter(_) => unreachable!(),
    };

    let (resident, _) = ForwardResidentBuffers::allocate(
        &device,
        vec![ForwardStreamInput {
            bytes: vec![0; 4],
            layout: VertexBufferLayout {
                stride: 4,
                step_mode: VertexStepMode::Vertex,
                attributes: Vec::new(),
            },
        }],
        vec![ForwardTextureInput {
            bytes: vec![0; 256],
            extent: [1, 1],
            bytes_per_row: 256,
        }],
    )
    .unwrap();
    let geometry_key =
        crate::residency::ForwardGeometryCacheKey::from_snapshot(&geometry, device.id());
    let texture_key =
        crate::residency::ForwardTextureCacheKey::from_snapshot(&texture, device.id());
    let other_geometry_key =
        crate::residency::ForwardGeometryCacheKey::from_snapshot(&geometry, other_device.id());
    let other_texture_key =
        crate::residency::ForwardTextureCacheKey::from_snapshot(&texture, other_device.id());
    let replacement_geometry_key = crate::residency::ForwardGeometryCacheKey::from_snapshot(
        &geometry_replacement,
        device.id(),
    );
    let replacement_texture_key =
        crate::residency::ForwardTextureCacheKey::from_snapshot(&texture_replacement, device.id());

    let mut cache = ForwardResidentCache::default();
    // Publication is intentionally a cache protocol: this test calls it only
    // after allocation and does not represent it as GPU completion evidence.
    assert!(cache.publish_geometry(geometry_key, &[&resident.streams[0]]));
    assert!(cache.publish_texture(texture_key, &resident.textures[0]));
    let streams = cache.geometry(geometry_key).unwrap();
    let cached_texture = cache.texture(texture_key).unwrap();
    assert_eq!(streams[0].instance, resident.streams[0].instance);
    assert_eq!(streams[0].desc.size, resident.streams[0].desc.size);
    assert_eq!(streams[0].desc.usage, resident.streams[0].desc.usage);
    assert_eq!(streams[0].desc.memory, resident.streams[0].desc.memory);
    assert_eq!(streams[0].desc.label, resident.streams[0].desc.label);
    assert_eq!(
        streams[0].incoming,
        BufferUses {
            stages: StageMask::VERTEX_INPUT,
            accesses: AccessMask::VERTEX_READ,
        }
    );
    assert_eq!(cached_texture.instance, resident.textures[0].instance);
    let actual = &cached_texture.desc;
    let original = &resident.textures[0].desc;
    assert_eq!(
        (
            actual.dimension,
            actual.extent,
            actual.array_layers,
            actual.format,
            actual.mip_levels,
            actual.samples,
            actual.usage,
            &actual.view_formats,
            &actual.label
        ),
        (
            original.dimension,
            original.extent,
            original.array_layers,
            original.format,
            original.mip_levels,
            original.samples,
            original.usage,
            &original.view_formats,
            &original.label
        )
    );
    assert_eq!(cached_texture.layout, ImageLayout::ShaderReadOnly);
    assert_eq!(
        cached_texture.incoming,
        TextureUses {
            stages: StageMask::FRAGMENT,
            accesses: AccessMask::SHADER_READ,
        }
    );

    assert!(cache.geometry(other_geometry_key).is_none());
    assert!(cache.texture(other_texture_key).is_none());
    assert!(!cache.publish_geometry(other_geometry_key, &[&resident.streams[0]]));
    assert!(!cache.publish_texture(other_texture_key, &resident.textures[0]));
    assert!(cache.geometry(geometry_key).is_some());
    assert!(cache.texture(texture_key).is_some());
    assert!(cache.geometry(replacement_geometry_key).is_none());
    assert!(cache.texture(replacement_texture_key).is_none());

    cache.retain_assets(&ForwardAssets {
        geometry: vec![geometry_replacement],
        materials: Vec::new(),
        textures: vec![texture_replacement],
    });
    assert!(cache.geometry(geometry_key).is_none());
    assert!(cache.texture(texture_key).is_none());
}

#[test]
fn coexisting_renderers_never_adopt_completed_forward_products_from_another_device() {
    let (old_device, _) = fixture(Script::AcceptedExact);
    let (new_device, _) = fixture(Script::AcceptedExact);
    let mut old_renderer = Renderer::new(old_device.clone()).unwrap();
    let mut new_renderer = Renderer::new(new_device.clone()).unwrap();

    let geometry_store = AssetStore::<GeometryAsset, GeometryData, ()>::new();
    let geometry_handle = geometry_store.create().unwrap();
    let geometry = match geometry_store.acquire(&geometry_handle).unwrap() {
        Acquire::Producer(producer) => producer
            .commit(
                GeometryData {
                    layout: GeometryLayout::default(),
                    streams: Vec::new(),
                    vertices: 0..0,
                },
                ResidentBytes::new(0),
            )
            .unwrap(),
        _ => unreachable!("new asset must issue a producer"),
    };
    let texture_store = AssetStore::<ColorTextureAsset, ColorTextureData, ()>::new();
    let texture_handle = texture_store.create().unwrap();
    let texture = match texture_store.acquire(&texture_handle).unwrap() {
        Acquire::Producer(producer) => producer
            .commit(
                ColorTextureData {
                    extent: [1, 1],
                    rgba8: vec![1, 2, 3, 4],
                },
                ResidentBytes::new(4),
            )
            .unwrap(),
        _ => unreachable!("new asset must issue a producer"),
    };

    let (old_resident, _) = ForwardResidentBuffers::allocate(
        &old_device,
        vec![ForwardStreamInput {
            bytes: vec![0; 4],
            layout: VertexBufferLayout {
                stride: 4,
                step_mode: VertexStepMode::Vertex,
                attributes: Vec::new(),
            },
        }],
        vec![ForwardTextureInput {
            bytes: vec![0; 256],
            extent: [1, 1],
            bytes_per_row: 256,
        }],
    )
    .unwrap();
    let old_geometry =
        crate::residency::ForwardGeometryCacheKey::from_snapshot(&geometry, old_device.id());
    let old_texture =
        crate::residency::ForwardTextureCacheKey::from_snapshot(&texture, old_device.id());
    let new_geometry =
        crate::residency::ForwardGeometryCacheKey::from_snapshot(&geometry, new_device.id());
    let new_texture =
        crate::residency::ForwardTextureCacheKey::from_snapshot(&texture, new_device.id());

    // Model products that have reached completion on the old device. Keep its
    // renderer alive while the new renderer starts from the same logical assets.
    assert!(
        old_renderer
            .forward_residency
            .publish_geometry(old_geometry, &[&old_resident.streams[0]])
    );
    assert!(
        old_renderer
            .forward_residency
            .publish_texture(old_texture, &old_resident.textures[0])
    );
    assert!(
        old_renderer
            .forward_residency
            .geometry(old_geometry)
            .is_some()
    );
    assert!(
        old_renderer
            .forward_residency
            .texture(old_texture)
            .is_some()
    );

    assert!(
        new_renderer
            .forward_residency
            .geometry(new_geometry)
            .is_none()
    );
    assert!(
        new_renderer
            .forward_residency
            .texture(new_texture)
            .is_none()
    );
    // Even a mistaken cross-renderer publication cannot make old native owners
    // appear valid for the new device provenance.
    assert!(
        !new_renderer
            .forward_residency
            .publish_geometry(new_geometry, &[&old_resident.streams[0]])
    );
    assert!(
        !new_renderer
            .forward_residency
            .publish_texture(new_texture, &old_resident.textures[0])
    );
    assert!(
        old_renderer
            .forward_residency
            .geometry(old_geometry)
            .is_some()
    );
    assert!(
        old_renderer
            .forward_residency
            .texture(old_texture)
            .is_some()
    );

    let (new_resident, _) = ForwardResidentBuffers::allocate(
        &new_device,
        vec![ForwardStreamInput {
            bytes: vec![0; 4],
            layout: VertexBufferLayout {
                stride: 4,
                step_mode: VertexStepMode::Vertex,
                attributes: Vec::new(),
            },
        }],
        vec![ForwardTextureInput {
            bytes: vec![0; 256],
            extent: [1, 1],
            bytes_per_row: 256,
        }],
    )
    .unwrap();
    assert!(
        new_renderer
            .forward_residency
            .publish_geometry(new_geometry, &[&new_resident.streams[0]])
    );
    assert!(
        new_renderer
            .forward_residency
            .publish_texture(new_texture, &new_resident.textures[0])
    );
    assert!(
        new_renderer
            .forward_residency
            .geometry(new_geometry)
            .is_some()
    );
    assert!(
        new_renderer
            .forward_residency
            .texture(new_texture)
            .is_some()
    );
}

#[test]
fn forward_frame_pool_waits_for_readback_then_reuses_same_owner_instances() {
    let (device, _) = fixture(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let frame = blank_frame(&device);
    let expected = (
        frame.frame.upload_instance,
        frame.frame.uniform_instance,
        frame.frame.target_instance,
        frame.frame.readback_instance,
    );
    let submission = execute(&mut renderer, crate::renderer::PendingInputs::None, frame).unwrap();

    assert!(renderer.forward_frames[submission.slot].is_none());
    assert_eq!(renderer.pending(), 1);
    // The witness readback byte shape is intentionally invalid, but its map
    // succeeded after the completion wait, which is enough to recycle owners.
    assert!(block_on(renderer.readback(&submission)).is_err());
    assert_eq!(renderer.pending(), 0);
    let pooled = renderer.forward_frames[submission.slot].as_ref().unwrap();
    assert_eq!(
        (
            pooled.upload_instance,
            pooled.uniform_instance,
            pooled.target_instance,
            pooled.readback_instance,
        ),
        expected
    );
    assert_eq!(
        pooled.upload_incoming,
        BufferUses {
            stages: StageMask::COPY,
            accesses: AccessMask::COPY_READ,
        }
    );
    assert_eq!(
        pooled.uniform_incoming,
        BufferUses {
            stages: StageMask::COPY,
            accesses: AccessMask::COPY_WRITE,
        }
    );
    assert_eq!(pooled.target_layout, ImageLayout::CopySrc);
    assert_eq!(
        pooled.readback_incoming,
        BufferUses {
            stages: StageMask::COPY,
            accesses: AccessMask::COPY_WRITE,
        }
    );
}

#[test]
fn device_loss_clears_completed_frame_pool_before_backend_reentry() {
    let (device, _) = fixture(Script::AcceptedCompletionDeviceLost);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let frame = blank_frame(&device);
    // Seed a previously completed allocation; the failure path must clear it
    // together with every device-local cache before rejecting new work.
    renderer.forward_frames[0] = Some(crate::frame_data::pool::recycle(frame.frame, false));
    let submission = execute(
        &mut renderer,
        crate::renderer::PendingInputs::None,
        blank_frame(&device),
    )
    .unwrap();

    assert!(matches!(
        block_on(renderer.readback(&submission)),
        Err(RendererError::Rhi(Error::DeviceLost))
    ));
    assert!(renderer.forward_frames.iter().all(Option::is_none));
    assert!(matches!(
        execute(
            &mut renderer,
            crate::renderer::PendingInputs::None,
            blank_frame(&device)
        ),
        Err(RendererError::RendererClosed)
    ));
}

#[test]
fn forward_frame_pool_compatibility_uses_actual_owner_facts() {
    let (device, _) = fixture(Script::AcceptedExact);
    let (other_device, _) = fixture(Script::AcceptedExact);
    let bytes = vec![0; 16];
    let (mut buffers, _) =
        ForwardFrameBuffers::allocate(&device, [1, 1], 16, bytes, 0..16, 256).unwrap();

    assert!(crate::frame_data::pool::compatible(
        &buffers,
        device.id(),
        [1, 1],
        16,
        256,
    ));
    assert!(!crate::frame_data::pool::compatible(
        &buffers,
        other_device.id(),
        [1, 1],
        16,
        256,
    ));
    assert!(!crate::frame_data::pool::compatible(
        &buffers,
        device.id(),
        [2, 1],
        16,
        256,
    ));
    assert!(!crate::frame_data::pool::compatible(
        &buffers,
        device.id(),
        [1, 1],
        17,
        256,
    ));
    assert!(!crate::frame_data::pool::compatible(
        &buffers,
        device.id(),
        [1, 1],
        16,
        4
    ));
    let original_uniform = buffers.uniform_desc.clone();
    let mut short_uniform = original_uniform.clone();
    short_uniform.size = 8;
    buffers.uniform = device.create_buffer(&short_uniform).unwrap();
    buffers.uniform_desc = short_uniform;
    buffers.uniform_instance = crate::graph::ResourceInstanceId::fresh().unwrap();
    assert!(!crate::frame_data::pool::compatible(
        &buffers,
        device.id(),
        [1, 1],
        16,
        256,
    ));
    buffers.uniform = device.create_buffer(&original_uniform).unwrap();
    buffers.uniform_desc = original_uniform;
    buffers.uniform_instance = crate::graph::ResourceInstanceId::fresh().unwrap();
    let mut short_readback = buffers.readback_desc.clone();
    short_readback.size = 128;
    buffers.readback = device.create_buffer(&short_readback).unwrap();
    buffers.readback_desc = short_readback;
    buffers.readback_instance = crate::graph::ResourceInstanceId::fresh().unwrap();
    assert!(!crate::frame_data::pool::compatible(
        &buffers,
        device.id(),
        [1, 1],
        16,
        256
    ));
}
