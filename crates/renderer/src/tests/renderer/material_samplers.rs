//! Per-draw sampler values over shared immutable image ownership.
use super::*;
use crate::{BoundsVisibility, DeferredCpuPipeline, ForwardCpuPipeline, ForwardSamplerBinding};

fn device(script: Script) -> (Device<WitnessBackend>, Arc<Native>) {
    fixture_with_caps(script, |caps| {
        caps.limits.max_color_attachments = 3;
        caps.limits.max_bind_groups = 3;
        caps.limits.max_bindings_per_group = 8;
        caps.limits.max_vertex_attributes = 4;
    })
}

fn snapshot<K: fluxel_assets::AssetKind, T: Send + Sync + 'static>(
    value: T,
) -> fluxel_assets::AssetSnapshot<K, T> {
    let store = AssetStore::<K, T, ()>::new();
    match store.acquire(&store.create().unwrap()).unwrap() {
        Acquire::Producer(producer) => producer.commit(value, ResidentBytes::new(0)).unwrap(),
        _ => unreachable!(),
    }
}

fn assets() -> ForwardAssets {
    use fluxel_material::{
        FragmentInput, MaterialNode, MaterialNodeGraph, MaterialNodeId, MaterialNodeInput,
        MaterialNodeInputSource,
    };
    let mut bytes = Vec::new();
    for position in [[0.0_f32, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]] {
        for value in position
            .into_iter()
            .chain([0.0, 0.0, 1.0])
            .chain([0.0, 0.0])
            .chain([1.0, 0.0, 0.0, 1.0])
        {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    let geometry = GeometryData {
        layout: GeometryLayout {
            streams: vec![GeometryStream {
                stride: 48,
                step_mode: VertexStepMode::Vertex,
                attributes: [
                    ("POSITION", 0, 3),
                    ("NORMAL", 12, 3),
                    ("UV0", 24, 2),
                    ("TANGENT", 32, 4),
                ]
                .map(|(semantic, offset, size)| GeometryAttribute {
                    semantic: semantic.into(),
                    offset,
                    format: VertexFormat::Float32(size),
                })
                .to_vec(),
            }],
        },
        streams: vec![bytes],
        vertices: 0..3,
    };
    let uv = || {
        vec![FragmentInput {
            name: "uv".into(),
            ty: MaterialType::Vec2,
        }]
    };
    let material = MaterialDefinition::new(
        vec![MaterialFragment { name: "sampled_color".into(), inputs: uv(), output: MaterialType::Vec4,
            calls: vec![], wgsl: "fn sampled_color(uv: vec2<f32>) -> vec4<f32> { return textureSample(material_texture, material_sampler, uv) * tint; }".into() },
            MaterialFragment { name: "properties".into(), inputs: uv(), output: MaterialType::Vec2, calls: vec![],
                wgsl: "fn properties(uv: vec2<f32>) -> vec2<f32> { return vec2<f32>(0.5, 0.0); }".into() },
            MaterialFragment { name: "sample_normal".into(), inputs: uv(), output: MaterialType::Vec3, calls: vec![],
                wgsl: "fn sample_normal(uv: vec2<f32>) -> vec3<f32> { return vec3<f32>(0.0, 0.0, 1.0); }".into() }],
        "unused", vec![MaterialParameter { name: "tint".into(), ty: MaterialType::Vec4, resource: "material_tint".into() }],
        vec![MaterialResource { name: "material_tint".into(), kind: MaterialResourceKind::UniformVec4, location: BindingLocation { group: 0, binding: 0 } },
            MaterialResource { name: "material_texture".into(), kind: MaterialResourceKind::Texture2dF32, location: BindingLocation { group: 0, binding: 1 } },
            MaterialResource { name: "material_sampler".into(), kind: MaterialResourceKind::Sampler { comparison: false }, location: BindingLocation { group: 0, binding: 2 } }],
        BTreeMap::new(), SurfaceOutput::MetallicRoughnessUv0Normal { properties: "properties".into(), normal: "sample_normal".into() },
    ).with_node_graph(MaterialNodeGraph { nodes: vec![MaterialNode { id: MaterialNodeId(1), callable: "sampled_color".into(),
        inputs: vec![MaterialNodeInput { input: "uv".into(), source: MaterialNodeInputSource::Uv0 }] }], output: MaterialNodeId(1) });
    ForwardAssets {
        geometry: vec![snapshot::<GeometryAsset, _>(geometry)],
        materials: vec![snapshot::<MaterialAsset, _>(material)],
        textures: vec![snapshot::<ColorTextureAsset, _>(ColorTextureData {
            extent: [1, 1],
            encoding: crate::TextureEncoding::Linear,
            mip_levels: 1,
            rgba8: vec![128, 128, 255, 255],
        })],
    }
}

fn policy(address: AddressMode) -> SamplerDesc {
    SamplerDesc {
        label: Some("per-draw sampler test".into()),
        address: [address; 3],
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mip_filter: FilterMode::Nearest,
        lod: 0.0..0.0,
        anisotropy: 1,
        compare: None,
        border: None,
    }
}

fn inputs(
    assets: &ForwardAssets,
    descriptors: &[SamplerDesc],
) -> (crate::FrameSnapshot<ForwardResources>, crate::RenderViewId) {
    named_inputs(assets, descriptors, "material_sampler")
}

fn named_inputs(
    assets: &ForwardAssets,
    descriptors: &[SamplerDesc],
    name: &str,
) -> (crate::FrameSnapshot<ForwardResources>, crate::RenderViewId) {
    let mut scene = RenderScene::new().unwrap();
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::InsertView(
        RenderView::new(RenderTransform::identity(), [8, 8]).unwrap(),
    ));
    for descriptor in descriptors {
        changes.push(RenderChange::Insert(RenderRecord {
            resources: ForwardResources {
                geometry: SceneAssetRef::from_snapshot(&assets.geometry[0]),
                material: SceneAssetRef::from_snapshot(&assets.materials[0]),
                textures: vec![crate::ForwardTextureBinding::asset(
                    "material_texture",
                    SceneAssetRef::from_snapshot(&assets.textures[0]),
                )],
                sampler: Some(ForwardSamplerBinding {
                    resource: name.into(),
                    descriptor: descriptor.clone(),
                }),
                tint: [1.; 4],
            },
            transform: RenderTransform::identity(),
            bounds: bounds(),
        }));
    }
    let receipt = scene.apply(&changes).unwrap();
    (scene.snapshot(), receipt.inserted_views()[0])
}

fn submit(
    renderer: &mut Renderer<WitnessBackend>,
    scene: crate::FrameSnapshot<ForwardResources>,
    view: crate::RenderViewId,
    assets: &ForwardAssets,
    deferred: bool,
) -> std::result::Result<Submission, RendererError> {
    if deferred {
        block_on(renderer.submit_deferred(
            &DeferredCpuPipeline::new(BoundsVisibility),
            scene,
            view,
            bounds(),
            assets,
        ))
    } else {
        block_on(renderer.submit_forward(
            &ForwardCpuPipeline::new(BoundsVisibility),
            scene,
            view,
            bounds(),
            assets,
        ))
    }
}

fn resident_facts(
    renderer: &Renderer<WitnessBackend>,
    submission: &Submission,
    expected_samplers: usize,
) -> crate::graph::ResourceInstanceId {
    let pending = renderer.slots[submission.slot].as_ref().unwrap();
    let resident = match &pending.workload {
        crate::renderer::PendingWork::Forward { _frame, .. } => {
            assert_eq!(_frame.draws.len(), 2);
            assert!(_frame.draws.iter().all(|draw| draw.textures == [0]));
            &_frame.resident
        }
        crate::renderer::PendingWork::Deferred { _frame, .. } => {
            assert_eq!(_frame.draws.len(), 2);
            assert!(_frame.draws.iter().all(|draw| draw.textures == [0]));
            &_frame.resident
        }
        _ => panic!("ordinary material frame"),
    };
    assert_eq!(
        resident.textures.len(),
        1,
        "sampling does not split image owners"
    );
    assert_eq!(resident.samplers.len(), expected_samplers);
    resident.textures[0].instance
}

#[test]
fn material_samplers_bind_each_draw_policy_to_the_same_image_in_both_pipelines() {
    for deferred in [false, true] {
        let (device, native) = device(Script::AcceptedExact);
        let mut renderer = Renderer::new(device).unwrap();
        let assets = assets();
        let (scene, view) = inputs(
            &assets,
            &[
                policy(AddressMode::ClampToEdge),
                policy(AddressMode::MirrorRepeat),
            ],
        );
        let submitted = submit(&mut renderer, scene, view, &assets, deferred).unwrap();
        resident_facts(&renderer, &submitted, 2);
        let actual = native.bound_sampler_descs.lock().unwrap();
        let material: Vec<_> = actual.iter().filter(|(binding, _)| *binding == 2).collect();
        assert_eq!(material.len(), 2);
        // Observations come from the native sampler handles received by the
        // backend, independently of the renderer's saved descriptor table.
        assert_eq!(material[0].1.address, [AddressMode::ClampToEdge; 3]);
        assert_eq!(material[1].1.address, [AddressMode::MirrorRepeat; 3]);
        drop(actual);
        let owners = 2 + usize::from(deferred); // Deferred also samples scene color.
        assert_eq!(native.created_samplers.load(Ordering::SeqCst), owners);
        assert_eq!(native.dropped_samplers.load(Ordering::SeqCst), 0);
        block_on(renderer.readback(&submitted)).unwrap();
        assert_eq!(native.dropped_samplers.load(Ordering::SeqCst), owners);
    }
}

#[test]
fn material_samplers_reuse_equal_policies_and_ignore_debug_labels() {
    for deferred in [false, true] {
        let (device, native) = device(Script::AcceptedExact);
        let mut renderer = Renderer::new(device).unwrap();
        let assets = assets();
        let first = policy(AddressMode::MirrorRepeat);
        let mut second = first.clone();
        second.label = Some("a different debug label".into());
        let (scene, view) = inputs(&assets, &[first, second]);
        let submitted = submit(&mut renderer, scene, view, &assets, deferred).unwrap();
        resident_facts(&renderer, &submitted, 1);
        assert_eq!(
            native.created_samplers.load(Ordering::SeqCst),
            1 + usize::from(deferred)
        );
        block_on(renderer.readback(&submitted)).unwrap();
    }
}

#[test]
fn material_sampler_changes_reuse_completed_images_and_bind_the_new_policy() {
    for deferred in [false, true] {
        let (device, native) = device(Script::AcceptedExact);
        let mut renderer = Renderer::new(device).unwrap();
        let assets = assets();
        let clamp = policy(AddressMode::ClampToEdge);
        let (scene, view) = inputs(&assets, &[clamp.clone(), clamp]);
        let cold = submit(&mut renderer, scene, view, &assets, deferred).unwrap();
        let image = resident_facts(&renderer, &cold, 1);
        block_on(renderer.readback(&cold)).unwrap();
        let uploaded = native.texture_uploads.lock().unwrap().len();
        native.bound_sampler_descs.lock().unwrap().clear();
        let (scene, view) = inputs(
            &assets,
            &[
                policy(AddressMode::Repeat),
                policy(AddressMode::MirrorRepeat),
            ],
        );
        let warm = submit(&mut renderer, scene, view, &assets, deferred).unwrap();
        assert_eq!(resident_facts(&renderer, &warm, 2), image);
        assert_eq!(native.texture_uploads.lock().unwrap().len(), uploaded);
        let actual = native.bound_sampler_descs.lock().unwrap();
        let material: Vec<_> = actual.iter().filter(|(binding, _)| *binding == 2).collect();
        assert_eq!(material.len(), 2);
        assert_eq!(material[0].1.address, [AddressMode::Repeat; 3]);
        assert_eq!(material[1].1.address, [AddressMode::MirrorRepeat; 3]);
        drop(actual);
        block_on(renderer.readback(&warm)).unwrap();
    }
}

#[test]
fn unsupported_material_sampler_cleans_partial_preparation_without_recording() {
    for deferred in [false, true] {
        let (device, native) = device(Script::AcceptedExact);
        let mut renderer = Renderer::new(device).unwrap();
        let assets = assets();
        native.sampler_anisotropy_limit.store(1, Ordering::SeqCst);
        let first = policy(AddressMode::ClampToEdge);
        let mut unsupported = policy(AddressMode::MirrorRepeat);
        unsupported.anisotropy = 4;
        unsupported.mip_filter = FilterMode::Linear;
        let (scene, view) = inputs(&assets, &[first, unsupported]);
        assert!(submit(&mut renderer, scene, view, &assets, deferred).is_err());
        assert_eq!(renderer.pending(), 0);
        assert_eq!(native.command_calls.load(Ordering::SeqCst), 0);
        assert_eq!(native.created_samplers.load(Ordering::SeqCst), 1);
        assert_eq!(native.dropped_samplers.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn material_sampler_invalid_real_draw_inputs_refuse_before_backend_allocation() {
    for deferred in [false, true] {
        let mut invalid = Vec::new();
        let base = policy(AddressMode::MirrorRepeat);
        let mut changed = base.clone();
        changed.lod.start = f32::NAN;
        invalid.push(changed);
        let mut changed = base.clone();
        changed.lod = 3.0..1.0;
        invalid.push(changed);
        let mut changed = base.clone();
        changed.compare = Some(CompareFunction::LessEqual);
        invalid.push(changed);
        let mut changed = base.clone();
        changed.address[0] = AddressMode::ClampToBorder;
        invalid.push(changed);
        let mut changed = base.clone();
        changed.anisotropy = 0;
        invalid.push(changed);
        let mut changed = base.clone();
        changed.anisotropy = 4;
        invalid.push(changed);
        for descriptor in invalid {
            let (device, native) = device(Script::AcceptedExact);
            let mut renderer = Renderer::new(device).unwrap();
            let assets = assets();
            let (scene, view) = inputs(&assets, &[base.clone(), descriptor]);
            assert!(matches!(
                submit(&mut renderer, scene, view, &assets, deferred),
                Err(RendererError::ForwardPreparation(
                    crate::ForwardPreparationError::InvalidSamplerDescriptor { .. }
                ))
            ));
            assert_eq!(native.created_buffers.load(Ordering::SeqCst), 0);
            assert_eq!(native.created_textures.load(Ordering::SeqCst), 0);
            assert_eq!(native.created_samplers.load(Ordering::SeqCst), 0);
            assert_eq!(native.command_calls.load(Ordering::SeqCst), 0);
        }
    }
}

#[test]
fn material_sampler_real_schema_name_mismatch_and_absent_resource_refuse_early() {
    for deferred in [false, true] {
        let (device, native) = device(Script::AcceptedExact);
        let mut renderer = Renderer::new(device).unwrap();
        let assets = assets();
        let (scene, view) = named_inputs(
            &assets,
            &[policy(AddressMode::MirrorRepeat)],
            "wrong_sampler",
        );
        assert!(matches!(
            submit(&mut renderer, scene, view, &assets, deferred),
            Err(RendererError::ForwardPreparation(
                crate::ForwardPreparationError::SamplerResourceMismatch { .. }
            ))
        ));
        let (_, _, no_sampler) = super::scene_and_assets();
        let (scene, view) = inputs(&no_sampler, &[policy(AddressMode::MirrorRepeat)]);
        assert!(matches!(
            submit(&mut renderer, scene, view, &no_sampler, deferred),
            Err(RendererError::ForwardPreparation(
                crate::ForwardPreparationError::UnexpectedSamplerResource { .. }
            ))
        ));
        assert_eq!(native.created_buffers.load(Ordering::SeqCst), 0);
        assert_eq!(native.created_textures.load(Ordering::SeqCst), 0);
        assert_eq!(native.created_samplers.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn material_samplers_do_not_fold_different_filter_lod_or_anisotropy_values() {
    for deferred in [false, true] {
        let mut base = policy(AddressMode::MirrorRepeat);
        base.mip_filter = FilterMode::Linear;
        base.lod = 0.0..3.0;
        let mut different = Vec::new();
        let mut value = base.clone();
        value.mag_filter = FilterMode::Nearest;
        different.push(value);
        let mut value = base.clone();
        value.min_filter = FilterMode::Nearest;
        different.push(value);
        let mut value = base.clone();
        value.mip_filter = FilterMode::Nearest;
        different.push(value);
        let mut value = base.clone();
        value.lod.start = 1.0;
        different.push(value);
        let mut value = base.clone();
        value.lod.end = 2.0;
        different.push(value);
        let mut value = base.clone();
        value.anisotropy = 2;
        different.push(value);
        for expected in different {
            let (device, native) = device(Script::AcceptedExact);
            let mut renderer = Renderer::new(device).unwrap();
            let assets = assets();
            let (scene, view) = inputs(&assets, &[base.clone(), expected.clone()]);
            let submitted = submit(&mut renderer, scene, view, &assets, deferred).unwrap();
            resident_facts(&renderer, &submitted, 2);
            let actual = native.bound_sampler_descs.lock().unwrap();
            let material: Vec<_> = actual.iter().filter(|(binding, _)| *binding == 2).collect();
            assert_eq!(material.len(), 2);
            let bound = &material[1].1;
            assert_eq!(bound.mag_filter, expected.mag_filter);
            assert_eq!(bound.min_filter, expected.min_filter);
            assert_eq!(bound.mip_filter, expected.mip_filter);
            assert_eq!(bound.lod, expected.lod);
            assert_eq!(bound.anisotropy, expected.anisotropy);
            drop(actual);
            block_on(renderer.readback(&submitted)).unwrap();
        }
    }
}

#[test]
fn material_sampler_capture_receiver_keeps_explicit_named_policy_after_repreparation() {
    let (device, native) = device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    let mut assets = assets();
    let (_, receiver_assets) = capture_forward_tests::receiver();
    let receiver_resources = crate::ForwardCaptureResources {
        geometry: SceneAssetRef::from_snapshot(&receiver_assets.geometry[0]),
        material: SceneAssetRef::from_snapshot(&receiver_assets.materials[0]),
        textures: vec![crate::ForwardTextureBinding::capture(
            "captured",
            "producer_scene",
        )],
        sampler: Some(ForwardSamplerBinding::new(
            "capture_sampler",
            SamplerDesc {
                mag_filter: FilterMode::Nearest,
                min_filter: FilterMode::Nearest,
                ..policy(AddressMode::Repeat)
            },
        )),
        tint: [1.0; 4],
    };
    assets.geometry.extend(receiver_assets.geometry);
    assets.materials.extend(receiver_assets.materials);
    let receiver = crate::prepare_forward_capture_receiver(
        &RenderView::new(RenderTransform::identity(), [8, 8]).unwrap(),
        crate::ForwardDrawSource::Feature {
            feature: crate::FeatureIssuer::new(71, 3)
                .issue("sampler-capture")
                .unwrap(),
            local_id: 1,
        },
        0,
        &receiver_resources,
        RenderTransform::identity(),
        &assets,
        &mut fluxel_shader::ShaderComposer::new(),
    )
    .unwrap();
    let (scene, view) = inputs(
        &assets,
        &[
            policy(AddressMode::ClampToEdge),
            policy(AddressMode::MirrorRepeat),
        ],
    );
    let submitted = block_on(renderer.submit_forward_capture(
        &ForwardCpuPipeline::new(BoundsVisibility),
        scene,
        view,
        bounds(),
        &assets,
        crate::CaptureTextureSlot::new("producer_scene").unwrap(),
        receiver,
    ))
    .unwrap();
    let actual = native.bound_sampler_descs.lock().unwrap();
    let material: Vec<_> = actual.iter().filter(|(binding, _)| *binding == 2).collect();
    assert_eq!(material.len(), 3);
    assert!(
        material
            .iter()
            .any(|(_, sampler)| sampler.address == [AddressMode::Repeat; 3]
                && sampler.mag_filter == FilterMode::Nearest
                && sampler.min_filter == FilterMode::Nearest)
    );
    drop(actual);
    assert_eq!(native.created_samplers.load(Ordering::SeqCst), 3);
    assert_eq!(native.dropped_samplers.load(Ordering::SeqCst), 0);
    block_on(renderer.readback(&submitted)).unwrap();
    assert_eq!(native.dropped_samplers.load(Ordering::SeqCst), 3);
}
