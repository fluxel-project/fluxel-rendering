//! Logical backend evidence for the same read-set plan executed by the native consumer.
use super::*;
use crate::{
    graph::{ReadSetRasterFrame, ResourceInstanceId},
    residency::ReadSetRasterBuffers,
};

fn input() -> Vec<u8> {
    let mut bytes = Vec::new();
    for value in [1.0_f32, 0.0, 1.0, 1.0] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes.extend_from_slice(&[17, 129, 241, 255]);
    bytes
}
fn pipeline(device: &Device<WitnessBackend>) -> RasterPipeline<WitnessBackend> {
    // SAFETY: inert logical pipeline belongs to the supplied witness device.
    unsafe { RasterPipeline::from_native(device.id(), ()) }
}
fn layout(device: &Device<WitnessBackend>) -> BindGroupLayout<WitnessBackend> {
    let entries = vec![
        BindGroupLayoutEntry {
            binding: 0,
            visibility: vec![ShaderStage::Fragment],
            ty: BindingType::UniformBuffer {
                dynamic_offset: true,
                min_size: 16,
            },
            count: BindingCount::One,
            partially_bound: false,
            nonuniform_indexing: false,
        },
        BindGroupLayoutEntry {
            binding: 1,
            visibility: vec![ShaderStage::Fragment],
            ty: BindingType::SampledTexture {
                sample: TextureSampleType::Float,
                dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: BindingCount::One,
            partially_bound: false,
            nonuniform_indexing: false,
        },
    ];
    // SAFETY: the no-op native layout has the exact metadata supplied here.
    unsafe { BindGroupLayout::from_native(device.id(), (), Arc::from(entries)) }
}

#[test]
fn read_set_submits_current_ranges_reuses_plan_and_retires_exact_owners() {
    let (device, native) = fixture(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let pipeline = pipeline(&device);
    let layout = layout(&device);
    let mut submissions = Vec::new();
    let mut owners = Vec::new();
    let mut id = None;
    for range in [0..256, 256..512, 512..768] {
        let (source, weak) = snapshot_bytes(input());
        owners.push(weak);
        submissions.push(
            block_on(renderer.submit_read_set_raster(
                source,
                pipeline.clone(),
                layout.clone(),
                range,
            ))
            .unwrap(),
        );
        if let Some(id) = id {
            assert_eq!(renderer.plan_id(), Some(id));
        } else {
            id = renderer.plan_id();
        }
    }
    assert!(id.is_some());
    assert_eq!(renderer.compilations(), 1);
    assert_eq!(renderer.pending(), 3);
    assert_eq!(
        *native.copy_destinations.lock().unwrap(),
        [0..16, 256..272, 512..528]
    );
    assert_eq!(*native.uniform_offsets.lock().unwrap(), [0, 256, 512]);
    assert_eq!(*native.texture_uploads.lock().unwrap(), [(512, 256); 3]);
    for packed in native.mapped_bytes.lock().unwrap().iter() {
        assert_eq!(packed.len(), 768);
        assert_eq!(&packed[..16], &input()[..16]);
        assert_eq!(&packed[512..516], &input()[16..20]);
    }
    for (index, submission) in submissions.iter().enumerate() {
        // Zero pixels are only scripted ticket-size/retirement evidence.
        assert_eq!(block_on(renderer.readback(submission)).unwrap().len(), 256);
        assert!(owners[index].upgrade().is_none());
        assert!(
            owners[index + 1..]
                .iter()
                .all(|owner| owner.upgrade().is_some())
        );
    }
    assert_eq!(*native.waited_completions.lock().unwrap(), [1, 2, 3]);
    assert_eq!(renderer.pending(), 0);
}

#[test]
fn read_set_bad_payload_range_and_foreign_pipeline_never_allocate_or_record() {
    let (device, native) = fixture(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    for (bytes, range) in [
        (vec![0; 19], 0..256),
        (input(), 1..257),
        (input(), 0..255),
        (input(), (1_u64 << 32)..(1_u64 << 32) + 256),
    ] {
        let result = block_on(renderer.submit_read_set_raster(
            snapshot_bytes(bytes).0,
            pipeline(&device),
            layout(&device),
            range,
        ));
        assert!(matches!(
            result,
            Err(RendererError::Rhi(Error::InvalidRange))
        ));
    }
    let (foreign, _) = fixture(Script::AcceptedExact);
    assert!(matches!(
        block_on(renderer.submit_read_set_raster(
            snapshot_bytes(input()).0,
            pipeline(&foreign),
            layout(&device),
            0..256
        )),
        Err(RendererError::Rhi(Error::WrongDevice))
    ));
    assert_eq!(native.created_buffers.load(Ordering::SeqCst), 0);
    assert_eq!(native.command_calls.load(Ordering::SeqCst), 0);
    assert_eq!(renderer.compilations(), 0);
}

#[test]
fn read_set_checks_actual_dynamic_uniform_alignment_before_cache_or_allocation() {
    for case in [
        "zero-alignment",
        "actual-alignment",
        "bind-groups",
        "bindings-per-group",
        "upload-buffer",
        "readback-buffer",
        "texture-dimension",
        "texture-array-layers",
        "color-attachment",
    ] {
        let (device, native) = fixture_with_caps(Script::AcceptedExact, |caps| match case {
            "zero-alignment" => caps.limits.uniform_offset_alignment = 0,
            "actual-alignment" => caps.limits.uniform_offset_alignment = 512,
            "bind-groups" => caps.limits.max_bind_groups = 0,
            "bindings-per-group" => caps.limits.max_bindings_per_group = 1,
            "upload-buffer" => caps.limits.max_buffer_bytes = 767,
            "readback-buffer" => caps.limits.max_buffer_bytes = 2047,
            "texture-dimension" => caps.limits.max_texture_dimension_2d = 7,
            "texture-array-layers" => caps.limits.max_texture_array_layers = 0,
            "color-attachment" => caps.limits.max_color_attachments = 0,
            _ => unreachable!(),
        });
        let mut renderer = Renderer::new(device.clone()).unwrap();
        assert!(
            matches!(
                block_on(renderer.submit_read_set_raster(
                    snapshot_bytes(input()).0,
                    pipeline(&device),
                    layout(&device),
                    256..512,
                )),
                Err(RendererError::Rhi(Error::InvalidRange))
            ),
            "{case}"
        );
        assert_eq!(renderer.compilations(), 0, "{case}");
        assert_eq!(native.created_buffers.load(Ordering::SeqCst), 0, "{case}");
        assert_eq!(native.command_calls.load(Ordering::SeqCst), 0, "{case}");
    }

    let (device, _) = fixture_with_caps(Script::AcceptedExact, |caps| {
        caps.limits.uniform_offset_alignment = 512;
        caps.limits.max_vertex_buffers = 0;
        caps.limits.max_vertex_attributes = 0;
    });
    let mut renderer = Renderer::new(device.clone()).unwrap();

    let submission = block_on(renderer.submit_read_set_raster(
        snapshot_bytes(input()).0,
        pipeline(&device),
        layout(&device),
        512..768,
    ))
    .unwrap();
    block_on(renderer.readback(&submission)).unwrap();
}

#[test]
fn read_set_actual_plan_preflight_refuses_owner_descriptor_import_alias_and_device() {
    let (device, native) = fixture(Script::AcceptedExact);
    let frame = ReadSetRasterFrame::compile(0..256).unwrap();
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let plan = renderer.read_set_raster_plan_for(&frame).unwrap();
    let mut buffers = ReadSetRasterBuffers::allocate(&device, 256, plan.row_pitch()).unwrap();
    frame.validate_bindings(device.id(), &buffers).unwrap();
    let good_desc = buffers.uniform_desc.clone();
    let mut incompatible = good_desc.clone();
    incompatible.memory = MemoryClass::Upload;
    buffers.uniform = device.create_buffer(&incompatible).unwrap();
    buffers.uniform_desc = incompatible;
    buffers.uniform_instance = ResourceInstanceId::fresh().unwrap();
    assert!(frame.validate_bindings(device.id(), &buffers).is_err());
    buffers.uniform = device.create_buffer(&good_desc).unwrap();
    buffers.uniform_desc = good_desc;
    buffers.uniform_instance = ResourceInstanceId::fresh().unwrap();
    buffers.source_layout = ImageLayout::ShaderReadOnly;
    assert!(frame.validate_bindings(device.id(), &buffers).is_err());
    buffers.source_layout = ImageLayout::Undefined;
    // Alias identity is conservatively rejected before transitions or recording.
    buffers.target_instance = buffers.source_instance;
    assert!(frame.validate_bindings(device.id(), &buffers).is_err());
    buffers.target_instance = ResourceInstanceId::fresh().unwrap();
    let (foreign, _) = fixture(Script::AcceptedExact);
    assert!(frame.validate_bindings(foreign.id(), &buffers).is_err());
    assert_eq!(native.command_calls.load(Ordering::SeqCst), 0);
}

#[test]
fn read_set_nonaccepted_and_loss_outcomes_follow_shared_retention_rules() {
    for script in [
        Script::CleanRejected,
        Script::PartialAccepted,
        Script::PartialIssued,
        Script::Indeterminate,
        Script::RecordingDeviceLost,
    ] {
        let (device, native) = fixture(script);
        let mut renderer = Renderer::new(device.clone()).unwrap();
        let (source, owner) = snapshot_bytes(input());
        let result = block_on(renderer.submit_read_set_raster(
            source,
            pipeline(&device),
            layout(&device),
            0..256,
        ));
        assert!(result.is_err());
        assert_eq!(native.finishes.load(Ordering::SeqCst), 1);
        if matches!(script, Script::CleanRejected) {
            assert_eq!(renderer.pending(), 0);
            assert!(owner.upgrade().is_none());
        } else {
            assert_eq!(renderer.pending(), 1);
            assert!(owner.upgrade().is_some());
            if matches!(script, Script::RecordingDeviceLost) {
                assert_eq!(renderer.compilations(), 0);
                assert!(block_on(renderer.shutdown()).is_err());
            } else {
                block_on(renderer.shutdown()).unwrap();
                assert!(owner.upgrade().is_none());
            }
        }
    }
}

#[test]
fn read_set_changed_candidate_topology_cannot_reuse_the_cached_device_plan() {
    let (device, native) = fixture(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    let frame = ReadSetRasterFrame::compile(0..256).unwrap();
    let id = renderer.read_set_raster_plan_for(&frame).unwrap().id();
    let changed = ReadSetRasterFrame::changed_topology_for_test();
    assert!(matches!(
        renderer.read_set_raster_plan_for(&changed),
        Err(RendererError::Rhi(Error::InvalidState))
    ));
    assert_eq!(renderer.plan_id(), Some(id));
    assert_eq!(native.created_buffers.load(Ordering::SeqCst), 0);
    assert_eq!(native.command_calls.load(Ordering::SeqCst), 0);
}
