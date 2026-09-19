//! Vertex-recipe rejection witness.

use crate::*;

#[allow(clippy::too_many_arguments, clippy::needless_borrow)]
pub(super) fn run_raster_vertex_recipe_negative_paths(
    device: &Device,
    backend: &mut RasterBackend,
    descriptor: &RasterPassDescriptor<'_, Texture>,
    target: &Texture,
    camera_pipeline: &RasterPipeline,
    camera_bindings: &RasterBindings,
    textured_pipeline: &RasterPipeline,
    textured_bindings: &RasterBindings,
    uniform: &Buffer,
    other_camera_bindings: &RasterBindings,
    sampled_texture: &Texture,
    foreign_device: &Device,
) {
    // rejected, and every rejection remains pre-submit.
    let camera_vertex = device
        .create_buffer(BufferDescriptor {
            buffer: BufferDesc { size: 36 },
            usage: BufferUsage::from_kinds([BufferUsageKind::Vertex]),
            memory: MemoryPolicy::DeviceOnly,
        })
        .unwrap();
    let camera_index = device
        .create_buffer(BufferDescriptor {
            buffer: BufferDesc { size: 12 },
            usage: BufferUsage::from_kinds([BufferUsageKind::Index]),
            memory: MemoryPolicy::DeviceOnly,
        })
        .unwrap();
    let mut encoder = backend.begin_encoder(QueueId::new(0)).unwrap();
    assert!(
        backend
            .set_bindings(&mut encoder, &camera_bindings)
            .is_err()
    );
    backend
        .transition_texture(
            &mut encoder,
            &target,
            TextureRange::Whole,
            ResourceAccessState::Undefined,
            ResourceAccessState::ColorAttachmentReadWrite,
        )
        .unwrap();
    backend.begin_raster(&mut encoder, &descriptor).unwrap();
    backend
        .set_raster_pipeline(&mut encoder, &camera_pipeline)
        .unwrap();
    backend
        .set_vertex_buffer(&mut encoder, 0, &camera_vertex, 0)
        .unwrap();
    backend
        .set_index_buffer(&mut encoder, &camera_index, 0, IndexFormat::Uint32)
        .unwrap();
    assert_eq!(
        backend.draw_indexed(&mut encoder, 0..3, 0, 0..1),
        Err(NativeExecutionError::RasterStateMismatch)
    );
    assert_eq!(
        backend.set_bindings(&mut encoder, &other_camera_bindings),
        Err(NativeExecutionError::RasterStateMismatch)
    );
    backend
        .set_bindings(&mut encoder, &camera_bindings)
        .unwrap();
    backend.draw_indexed(&mut encoder, 0..3, 0, 0..1).unwrap();
    backend.end_raster(&mut encoder).unwrap();
    drop(encoder);

    // The textured recipe has an independent binding class: a draw before
    // that exact binding, or any binding/pipeline cross-over, is rejected
    // before the native draw.  The final valid call proves this reaches
    // the real native bind-group boundary without submitting the encoder.
    let mut encoder = backend.begin_encoder(QueueId::new(0)).unwrap();
    backend
        .transition_texture(
            &mut encoder,
            &target,
            TextureRange::Whole,
            ResourceAccessState::Undefined,
            ResourceAccessState::ColorAttachmentReadWrite,
        )
        .unwrap();
    backend.begin_raster(&mut encoder, &descriptor).unwrap();
    backend
        .set_raster_pipeline(&mut encoder, &textured_pipeline)
        .unwrap();
    backend
        .set_vertex_buffer(&mut encoder, 0, &camera_vertex, 0)
        .unwrap();
    backend
        .set_index_buffer(&mut encoder, &camera_index, 0, IndexFormat::Uint32)
        .unwrap();
    assert_eq!(
        backend.draw_indexed(&mut encoder, 0..3, 0, 0..1),
        Err(NativeExecutionError::RasterStateMismatch)
    );
    assert_eq!(
        backend.set_bindings(&mut encoder, &camera_bindings),
        Err(NativeExecutionError::RasterStateMismatch)
    );
    backend
        .set_bindings(&mut encoder, &textured_bindings)
        .unwrap();
    backend.draw_indexed(&mut encoder, 0..3, 0, 0..1).unwrap();
    backend.end_raster(&mut encoder).unwrap();
    drop(encoder);

    // The UV recipe has two non-interchangeable vertex slots. Every
    // negative result below drops its encoder without finish/submit; a
    // subsequent legal call proves rejected setters did not poison the
    // current epoch's readiness.
    let uv_pipeline = device
        .create_raster_pipeline(
            crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUv,
        )
        .unwrap();
    let positions = device
        .create_buffer(BufferDescriptor {
            buffer: BufferDesc { size: 36 },
            usage: BufferUsage::from_kinds([BufferUsageKind::Vertex]),
            memory: MemoryPolicy::DeviceOnly,
        })
        .unwrap();
    let texture_coordinates = device
        .create_buffer(BufferDescriptor {
            buffer: BufferDesc { size: 24 },
            usage: BufferUsage::from_kinds([BufferUsageKind::Vertex]),
            memory: MemoryPolicy::DeviceOnly,
        })
        .unwrap();
    let uv_index = device
        .create_buffer(BufferDescriptor {
            buffer: BufferDesc { size: 12 },
            usage: BufferUsage::from_kinds([BufferUsageKind::Index]),
            memory: MemoryPolicy::DeviceOnly,
        })
        .unwrap();
    let uv_bindings = RasterBindings::RasterUvTexture(
        device
            .create_raster_uv_texture_bindings(
                &uv_pipeline,
                &uniform,
                &sampled_texture,
                &positions,
                &texture_coordinates,
                3,
            )
            .unwrap(),
    );
    let mut encoder = backend.begin_encoder(QueueId::new(0)).unwrap();
    backend
        .transition_texture(
            &mut encoder,
            &target,
            TextureRange::Whole,
            ResourceAccessState::Undefined,
            ResourceAccessState::ColorAttachmentReadWrite,
        )
        .unwrap();
    backend.begin_raster(&mut encoder, &descriptor).unwrap();
    backend
        .set_raster_pipeline(&mut encoder, &uv_pipeline)
        .unwrap();
    assert_eq!(
        backend.set_vertex_buffer(&mut encoder, 2, &positions, 0),
        Err(NativeExecutionError::RasterVertexSlotOutOfRange { slot: 2 })
    );
    assert_eq!(
        backend.draw_indexed(&mut encoder, 0..3, 0, 0..1),
        Err(NativeExecutionError::RasterStateMismatch)
    );
    backend.set_bindings(&mut encoder, &uv_bindings).unwrap();
    assert_eq!(
        backend.set_bindings(&mut encoder, &uv_bindings),
        Err(NativeExecutionError::RasterBindingsAlreadySet)
    );
    assert_eq!(
        backend.set_vertex_buffer(&mut encoder, 2, &positions, 0),
        Err(NativeExecutionError::RasterVertexSlotOutOfRange { slot: 2 })
    );
    assert_eq!(
        backend.set_vertex_buffer(&mut encoder, 0, &texture_coordinates, 0),
        Err(NativeExecutionError::RasterVertexRoleMismatch { slot: 0 })
    );
    assert_eq!(
        backend.set_vertex_buffer(&mut encoder, 0, &positions, 4),
        Err(NativeExecutionError::RasterVertexRangeMismatch { slot: 0 })
    );
    backend
        .set_vertex_buffer(&mut encoder, 0, &positions, 0)
        .unwrap();
    assert_eq!(
        backend.set_vertex_buffer(&mut encoder, 0, &positions, 0),
        Err(NativeExecutionError::RasterVertexSlotAlreadyBound { slot: 0 })
    );
    backend
        .set_index_buffer(&mut encoder, &uv_index, 0, IndexFormat::Uint32)
        .unwrap();
    assert_eq!(
        backend.draw_indexed(&mut encoder, 0..3, 0, 0..1),
        Err(NativeExecutionError::RasterVertexSlotMissing { slot: 1 })
    );
    backend
        .set_vertex_buffer(&mut encoder, 1, &texture_coordinates, 0)
        .unwrap();
    backend.draw_indexed(&mut encoder, 0..3, 0, 0..1).unwrap();

    // The sampler recipe is a separate closed binding, while both UV
    // kernels share the epoch family. Cross-kernel bindings fail closed,
    // and a successful switch clears every UV readiness bit before the
    // native setter is reached again.
    let linear_pipeline = device
        .create_raster_pipeline(
            crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp,
        )
        .unwrap();
    let linear_bindings = RasterBindings::RasterUvLinearClampTexture(
        device
            .create_raster_uv_linear_clamp_texture_bindings(
                &linear_pipeline,
                &uniform,
                &sampled_texture,
                &positions,
                &texture_coordinates,
                3,
            )
            .unwrap(),
    );
    backend
        .set_raster_pipeline(&mut encoder, &linear_pipeline)
        .unwrap();
    assert_eq!(
        backend.set_bindings(&mut encoder, &uv_bindings),
        Err(NativeExecutionError::RasterStateMismatch)
    );
    assert_eq!(
        backend.draw_indexed(&mut encoder, 0..3, 0, 0..1),
        Err(NativeExecutionError::RasterEpochMismatch)
    );
    backend
        .set_bindings(&mut encoder, &linear_bindings)
        .unwrap();
    backend
        .set_vertex_buffer(&mut encoder, 0, &positions, 0)
        .unwrap();
    backend
        .set_vertex_buffer(&mut encoder, 1, &texture_coordinates, 0)
        .unwrap();
    backend
        .set_index_buffer(&mut encoder, &uv_index, 0, IndexFormat::Uint32)
        .unwrap();
    backend.draw_indexed(&mut encoder, 0..3, 0, 0..1).unwrap();
    backend
        .set_raster_pipeline(&mut encoder, &uv_pipeline)
        .unwrap();
    assert_eq!(
        backend.draw_indexed(&mut encoder, 0..3, 0, 0..1),
        Err(NativeExecutionError::RasterEpochMismatch)
    );
    drop(encoder);

    // The normal-Lambert recipe has its own two-stream epoch family. All
    // rejected calls below happen before draw/submit; the final legal
    // recording proves failed setters did not manufacture readiness.
    let normal_pipeline = device
        .create_raster_pipeline(
            crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialNormalLambert,
        )
        .unwrap();
    let other_normal_pipeline = device
        .create_raster_pipeline(
            crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialNormalLambert,
        )
        .unwrap();
    let normals = device
        .create_buffer(BufferDescriptor {
            buffer: BufferDesc { size: 36 },
            usage: BufferUsage::from_kinds([BufferUsageKind::Vertex]),
            memory: MemoryPolicy::DeviceOnly,
        })
        .unwrap();
    let short_normals = device
        .create_buffer(BufferDescriptor {
            buffer: BufferDesc { size: 24 },
            usage: BufferUsage::from_kinds([BufferUsageKind::Vertex]),
            memory: MemoryPolicy::DeviceOnly,
        })
        .unwrap();
    let non_vertex_normals = device
        .create_buffer(BufferDescriptor {
            buffer: BufferDesc { size: 36 },
            usage: BufferUsage::from_kinds([BufferUsageKind::StorageRead]),
            memory: MemoryPolicy::DeviceOnly,
        })
        .unwrap();
    let normal_index = device
        .create_buffer(BufferDescriptor {
            buffer: BufferDesc { size: 12 },
            usage: BufferUsage::from_kinds([BufferUsageKind::Index]),
            memory: MemoryPolicy::DeviceOnly,
        })
        .unwrap();
    let foreign_normal = foreign_device
        .create_buffer(BufferDescriptor {
            buffer: BufferDesc { size: 36 },
            usage: BufferUsage::from_kinds([BufferUsageKind::Vertex]),
            memory: MemoryPolicy::DeviceOnly,
        })
        .unwrap();
    let normal_bindings = RasterBindings::RasterNormal(
        device
            .create_raster_normal_bindings(&normal_pipeline, &uniform, &positions, &normals, 3)
            .unwrap(),
    );
    let other_normal_bindings = RasterBindings::RasterNormal(
        device
            .create_raster_normal_bindings(
                &other_normal_pipeline,
                &uniform,
                &positions,
                &normals,
                3,
            )
            .unwrap(),
    );
    assert!(matches!(
        device.create_raster_normal_bindings(&normal_pipeline, &uniform, &positions, &normals, 2,),
        Err(crate::RasterCreateError::InvalidPositionStreamRange)
    ));
    assert!(matches!(
        device.create_raster_normal_bindings(
            &normal_pipeline,
            &uniform,
            &positions,
            &short_normals,
            3,
        ),
        Err(crate::RasterCreateError::InvalidNormalStreamRange)
    ));
    assert!(matches!(
        device.create_raster_normal_bindings(&camera_pipeline, &uniform, &positions, &normals, 3,),
        Err(crate::RasterCreateError::BindingRecipeMismatch)
    ));
    assert!(matches!(
        device.create_raster_normal_bindings(
            &normal_pipeline,
            &uniform,
            &positions,
            &non_vertex_normals,
            3,
        ),
        Err(crate::RasterCreateError::VertexUsageRequired)
    ));
    assert!(matches!(
        device.create_raster_normal_bindings(
            &normal_pipeline,
            &uniform,
            &positions,
            &foreign_normal,
            3,
        ),
        Err(crate::RasterCreateError::ForeignDevice)
    ));
    let mut encoder = backend.begin_encoder(QueueId::new(0)).unwrap();
    backend
        .transition_texture(
            &mut encoder,
            &target,
            TextureRange::Whole,
            ResourceAccessState::Undefined,
            ResourceAccessState::ColorAttachmentReadWrite,
        )
        .unwrap();
    backend.begin_raster(&mut encoder, &descriptor).unwrap();
    backend
        .set_raster_pipeline(&mut encoder, &normal_pipeline)
        .unwrap();
    assert_eq!(
        backend.draw_indexed(&mut encoder, 0..3, 0, 0..1),
        Err(NativeExecutionError::RasterStateMismatch)
    );
    assert_eq!(
        backend.set_bindings(&mut encoder, &uv_bindings),
        Err(NativeExecutionError::RasterStateMismatch)
    );
    backend
        .set_bindings(&mut encoder, &normal_bindings)
        .unwrap();
    assert_eq!(
        backend.set_vertex_buffer(&mut encoder, 1, &foreign_normal, 0),
        Err(NativeExecutionError::ForeignResource)
    );
    assert_eq!(
        backend.set_bindings(&mut encoder, &normal_bindings),
        Err(NativeExecutionError::RasterBindingsAlreadySet)
    );
    assert_eq!(
        backend.set_bindings(&mut encoder, &other_normal_bindings),
        Err(NativeExecutionError::RasterStateMismatch)
    );
    assert_eq!(
        backend.set_vertex_buffer(&mut encoder, 2, &positions, 0),
        Err(NativeExecutionError::RasterVertexSlotOutOfRange { slot: 2 })
    );
    assert_eq!(
        backend.set_vertex_buffer(&mut encoder, 0, &normals, 0),
        Err(NativeExecutionError::RasterVertexRoleMismatch { slot: 0 })
    );
    assert_eq!(
        backend.set_vertex_buffer(&mut encoder, 1, &positions, 0),
        Err(NativeExecutionError::RasterVertexRoleMismatch { slot: 1 })
    );
    assert_eq!(
        backend.set_vertex_buffer(&mut encoder, 1, &short_normals, 0),
        Err(NativeExecutionError::RasterVertexRoleMismatch { slot: 1 })
    );
    assert_eq!(
        backend.set_vertex_buffer(&mut encoder, 1, &non_vertex_normals, 0),
        Err(NativeExecutionError::RasterVertexRoleMismatch { slot: 1 })
    );
    assert_eq!(
        backend.set_vertex_buffer(&mut encoder, 0, &positions, 4),
        Err(NativeExecutionError::RasterVertexRangeMismatch { slot: 0 })
    );
    backend
        .set_vertex_buffer(&mut encoder, 0, &positions, 0)
        .unwrap();
    assert_eq!(
        backend.set_vertex_buffer(&mut encoder, 0, &positions, 0),
        Err(NativeExecutionError::RasterVertexSlotAlreadyBound { slot: 0 })
    );
    backend
        .set_index_buffer(&mut encoder, &normal_index, 0, IndexFormat::Uint32)
        .unwrap();
    assert_eq!(
        backend.draw_indexed(&mut encoder, 0..3, 0, 0..1),
        Err(NativeExecutionError::RasterVertexSlotMissing { slot: 1 })
    );
    backend
        .set_vertex_buffer(&mut encoder, 1, &normals, 0)
        .unwrap();
    backend.draw_indexed(&mut encoder, 0..3, 0, 0..1).unwrap();
    // Switching away and back makes every old binding/vertex/index state
    // stale; this is checked before a second native draw.
    backend
        .set_raster_pipeline(&mut encoder, &camera_pipeline)
        .unwrap();
    backend
        .set_raster_pipeline(&mut encoder, &normal_pipeline)
        .unwrap();
    assert_eq!(
        backend.draw_indexed(&mut encoder, 0..3, 0, 0..1),
        Err(NativeExecutionError::RasterEpochMismatch)
    );
    drop(encoder);

    // Vertex color uses the same two-stream shape as UV and normal, but its
    // second stream has a different closed ABI (four packed bytes per
    // vertex). Exercise that identity and epoch proof directly: every
    // rejection below occurs before finish/submit, then one legal recording
    // proves a rejected setter did not manufacture readiness.
    let vertex_color_pipeline = device
        .create_raster_pipeline(
            crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialVertexColor,
        )
        .unwrap();
    let other_vertex_color_pipeline = device
        .create_raster_pipeline(
            crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialVertexColor,
        )
        .unwrap();
    let vertex_colors = device
        .create_buffer(BufferDescriptor {
            buffer: BufferDesc { size: 12 },
            usage: BufferUsage::from_kinds([BufferUsageKind::Vertex]),
            memory: MemoryPolicy::DeviceOnly,
        })
        .unwrap();
    let vertex_color_index = device
        .create_buffer(BufferDescriptor {
            buffer: BufferDesc { size: 12 },
            usage: BufferUsage::from_kinds([BufferUsageKind::Index]),
            memory: MemoryPolicy::DeviceOnly,
        })
        .unwrap();
    let foreign_vertex_colors = foreign_device
        .create_buffer(BufferDescriptor {
            buffer: BufferDesc { size: 12 },
            usage: BufferUsage::from_kinds([BufferUsageKind::Vertex]),
            memory: MemoryPolicy::DeviceOnly,
        })
        .unwrap();
    let vertex_color_bindings = RasterBindings::RasterVertexColor(
        device
            .create_raster_vertex_color_bindings(
                &vertex_color_pipeline,
                &uniform,
                &positions,
                &vertex_colors,
                3,
            )
            .unwrap(),
    );
    let other_vertex_color_bindings = RasterBindings::RasterVertexColor(
        device
            .create_raster_vertex_color_bindings(
                &other_vertex_color_pipeline,
                &uniform,
                &positions,
                &vertex_colors,
                3,
            )
            .unwrap(),
    );
    assert!(matches!(
        device.create_raster_vertex_color_bindings(
            &vertex_color_pipeline,
            &uniform,
            &positions,
            &vertex_colors,
            2,
        ),
        Err(crate::RasterCreateError::InvalidPositionStreamRange)
    ));
    assert!(matches!(
        device.create_raster_vertex_color_bindings(
            &normal_pipeline,
            &uniform,
            &positions,
            &vertex_colors,
            3,
        ),
        Err(crate::RasterCreateError::BindingRecipeMismatch)
    ));
    assert!(matches!(
        device.create_raster_vertex_color_bindings(
            &vertex_color_pipeline,
            &uniform,
            &positions,
            &foreign_vertex_colors,
            3,
        ),
        Err(crate::RasterCreateError::ForeignDevice)
    ));
    let mut encoder = backend.begin_encoder(QueueId::new(0)).unwrap();
    backend
        .transition_texture(
            &mut encoder,
            &target,
            TextureRange::Whole,
            ResourceAccessState::Undefined,
            ResourceAccessState::ColorAttachmentReadWrite,
        )
        .unwrap();
    backend.begin_raster(&mut encoder, &descriptor).unwrap();
    backend
        .set_raster_pipeline(&mut encoder, &vertex_color_pipeline)
        .unwrap();
    assert_eq!(
        backend.draw_indexed(&mut encoder, 0..3, 0, 0..1),
        Err(NativeExecutionError::RasterStateMismatch)
    );
    assert_eq!(
        backend.set_bindings(&mut encoder, &normal_bindings),
        Err(NativeExecutionError::RasterStateMismatch)
    );
    backend
        .set_bindings(&mut encoder, &vertex_color_bindings)
        .unwrap();
    assert_eq!(
        backend.set_vertex_buffer(&mut encoder, 1, &foreign_vertex_colors, 0),
        Err(NativeExecutionError::ForeignResource)
    );
    assert_eq!(
        backend.set_bindings(&mut encoder, &vertex_color_bindings),
        Err(NativeExecutionError::RasterBindingsAlreadySet)
    );
    assert_eq!(
        backend.set_bindings(&mut encoder, &other_vertex_color_bindings),
        Err(NativeExecutionError::RasterStateMismatch)
    );
    assert_eq!(
        backend.set_vertex_buffer(&mut encoder, 2, &positions, 0),
        Err(NativeExecutionError::RasterVertexSlotOutOfRange { slot: 2 })
    );
    assert_eq!(
        backend.set_vertex_buffer(&mut encoder, 0, &vertex_colors, 0),
        Err(NativeExecutionError::RasterVertexRoleMismatch { slot: 0 })
    );
    assert_eq!(
        backend.set_vertex_buffer(&mut encoder, 1, &positions, 0),
        Err(NativeExecutionError::RasterVertexRoleMismatch { slot: 1 })
    );
    assert_eq!(
        backend.set_vertex_buffer(&mut encoder, 1, &vertex_colors, 4),
        Err(NativeExecutionError::RasterVertexRangeMismatch { slot: 1 })
    );
    backend
        .set_vertex_buffer(&mut encoder, 0, &positions, 0)
        .unwrap();
    assert_eq!(
        backend.set_vertex_buffer(&mut encoder, 0, &positions, 0),
        Err(NativeExecutionError::RasterVertexSlotAlreadyBound { slot: 0 })
    );
    backend
        .set_index_buffer(&mut encoder, &vertex_color_index, 0, IndexFormat::Uint32)
        .unwrap();
    assert_eq!(
        backend.draw_indexed(&mut encoder, 0..3, 0, 0..1),
        Err(NativeExecutionError::RasterVertexSlotMissing { slot: 1 })
    );
    backend
        .set_vertex_buffer(&mut encoder, 1, &vertex_colors, 0)
        .unwrap();
    backend.draw_indexed(&mut encoder, 0..3, 0, 0..1).unwrap();
    backend
        .set_raster_pipeline(&mut encoder, &camera_pipeline)
        .unwrap();
    backend
        .set_raster_pipeline(&mut encoder, &vertex_color_pipeline)
        .unwrap();
    assert_eq!(
        backend.draw_indexed(&mut encoder, 0..3, 0, 0..1),
        Err(NativeExecutionError::RasterEpochMismatch)
    );
    drop(encoder);

    // Nothing above finishes/submits; validation must remain silent.
    assert!(crate::imp::validation_diagnostics(&device.inner).is_empty());
}
