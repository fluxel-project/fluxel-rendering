//! PBR resource and input failure boundaries.

use super::*;

fn effect_material() -> fluxel_material::MaterialDefinition {
    uv0_normal_node_material()
        .with_code_node(fluxel_material::MaterialCodeNode {
            name: "ao".into(),
            inputs: vec![fluxel_material::FragmentInput {
                name: "uv".into(),
                ty: fluxel_material::MaterialType::Vec2,
            }],
            output: fluxel_material::MaterialType::F32,
            calls: Vec::new(),
            source: fluxel_shader::SourceUnit::new(
                "test:ao",
                "fn ao(uv: vec2<f32>) -> f32 { return uv.x * 0.0 + 0.25; }",
            ),
            resources: Vec::new(),
        })
        .with_code_node(fluxel_material::MaterialCodeNode {
            name: "emission".into(),
            inputs: vec![fluxel_material::FragmentInput {
                name: "uv".into(),
                ty: fluxel_material::MaterialType::Vec2,
            }],
            output: fluxel_material::MaterialType::Vec3,
            calls: Vec::new(),
            source: fluxel_shader::SourceUnit::new(
                "test:emission",
                "fn emission(uv: vec2<f32>) -> vec3<f32> { return vec3<f32>(0.1, 0.2, 0.3) + vec3<f32>(uv.x * 0.0); }",
            ),
            resources: Vec::new(),
        })
        .with_surface_extensions(fluxel_material::SurfaceExtensions {
            occlusion: Some("ao".into()),
            emissive: Some("emission".into()),
        })
}

fn effects_device(script: Script) -> (fluxel_rhi::api::Device<WitnessBackend>, Arc<Native>) {
    fixture_with_caps(script, |caps| {
        caps.limits.max_color_attachments = 3;
        caps.limits.max_bind_groups = 3;
        caps.limits.max_bindings_per_group = 4;
        caps.limits.max_vertex_buffers = 2;
        caps.limits.max_vertex_attributes = 4;
    })
}

#[test]
fn deferred_surface_effects_submit_offscreen_with_and_without_environment() {
    for environment in [false, true] {
        let (device, _) = effects_device(Script::AcceptedExact);
        let mut renderer = Renderer::new(device).expect("renderer");
        if environment {
            renderer
                .set_environment_irradiance(Some(irradiance_snapshot([1, 2, 3, 255, 4, 5, 6, 255])))
                .expect("environment");
        }
        let (snapshot, view, assets) = scene_and_assets_with_material(
            uv0_tangent_geometry([1.0, 0.0, 0.0, 1.0]),
            RenderTransform::identity(),
            effect_material(),
        );
        let submission = block_on(renderer.submit_deferred(
            &DeferredCpuPipeline::new(BoundsVisibility),
            snapshot,
            view,
            bounds(),
            &assets,
        ))
        .expect("effect Deferred submission");
        let pending = renderer.pending_for(&submission).expect("pending frame");
        let crate::renderer::PendingWork::Deferred { _frame, .. } = &pending.workload else {
            panic!("must retain Deferred owners");
        };
        assert_eq!(
            _frame
                .effects
                .as_ref()
                .expect("effect attachment")
                .desc
                .format,
            fluxel_rhi::api::TextureFormat::Rgba16Float
        );
        block_on(renderer.readback(&submission)).expect("readback retires effect owner");
        assert_eq!(renderer.pending(), 0);
    }
}

#[test]
fn deferred_surface_effects_present_and_three_color_cap_are_real_boundaries() {
    let (device, _) = effects_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).expect("renderer");
    let (snapshot, view, assets) = scene_and_assets_with_material(
        uv0_tangent_geometry([1.0, 0.0, 0.0, 1.0]),
        RenderTransform::identity(),
        effect_material(),
    );
    let mut surface = super::surface_tests::surface(&device, [8, 8]);
    let submission = block_on(renderer.submit_deferred_surface(
        &DeferredCpuPipeline::new(BoundsVisibility),
        snapshot,
        view,
        bounds(),
        &assets,
        block_on(surface.acquire()).expect("image"),
    ))
    .expect("effect Deferred Surface submission");
    let pending = renderer.pending_for(&submission).expect("pending frame");
    let crate::renderer::PendingWork::DeferredSurface { _frame, .. } = &pending.workload else {
        panic!("must retain Deferred Surface owners");
    };
    assert_eq!(
        _frame
            .effects
            .as_ref()
            .expect("effect attachment")
            .desc
            .format,
        fluxel_rhi::api::TextureFormat::Rgba16Float
    );
    block_on(renderer.complete(&submission)).expect("present completion retires effect owner");

    let (device, native) = pbr_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).expect("renderer");
    let (snapshot, view, assets) = scene_and_assets_with_material(
        uv0_tangent_geometry([1.0, 0.0, 0.0, 1.0]),
        RenderTransform::identity(),
        effect_material(),
    );
    let before = counts(&native);
    assert!(matches!(
        block_on(renderer.submit_deferred(
            &DeferredCpuPipeline::new(BoundsVisibility),
            snapshot,
            view,
            bounds(),
            &assets,
        )),
        Err(RendererError::Rhi(Error::Unsupported(_)))
    ));
    assert_eq!(
        counts(&native),
        before,
        "three-color rejection precedes allocation"
    );
}

#[test]
fn forward_surface_effects_submit_offscreen_with_and_without_environment() {
    for environment in [false, true] {
        let (device, _) = effects_device(Script::AcceptedExact);
        let mut renderer = Renderer::new(device).unwrap();
        if environment {
            renderer
                .set_environment_irradiance(Some(irradiance_snapshot([1, 2, 3, 255, 4, 5, 6, 255])))
                .unwrap();
        }
        let (snapshot, view, assets) = scene_and_assets_with_material(
            uv0_tangent_geometry([1.0, 0.0, 0.0, 1.0]),
            RenderTransform::identity(),
            effect_material(),
        );
        let submission = block_on(renderer.submit_forward(
            &ForwardCpuPipeline::new(BoundsVisibility),
            snapshot,
            view,
            bounds(),
            &assets,
        ))
        .expect("effects submit");
        assert_eq!(renderer.pending(), 1);
        drop(submission);
    }
}

#[test]
fn forward_surface_effects_submit_to_a_surface_with_environment() {
    let (device, _) = effects_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).expect("renderer");
    renderer
        .set_environment_irradiance(Some(irradiance_snapshot([1, 2, 3, 255, 4, 5, 6, 255])))
        .expect("environment");
    let mut surface = super::surface_tests::surface(&device, [8, 8]);
    let (snapshot, view, assets) = scene_and_assets_with_material(
        uv0_tangent_geometry([1.0, 0.0, 0.0, 1.0]),
        RenderTransform::identity(),
        effect_material(),
    );
    let submission = block_on(renderer.submit_forward_surface(
        &ForwardCpuPipeline::new(BoundsVisibility),
        snapshot,
        view,
        bounds(),
        &assets,
        block_on(surface.acquire()).expect("lease"),
    ))
    .expect("effect Surface submit");
    assert_eq!(renderer.pending(), 1);
    drop(submission);
}

#[test]
fn forward_surface_effects_reject_missing_callable_before_backend_work() {
    let (device, native) = pbr_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    let (snapshot, view, assets) = scene_and_assets_with_material(
        uv0_tangent_geometry([1.0, 0.0, 0.0, 1.0]),
        RenderTransform::identity(),
        uv0_normal_node_material().with_surface_extensions(fluxel_material::SurfaceExtensions {
            occlusion: Some("missing".into()),
            emissive: None,
        }),
    );
    let before = counts(&native);
    assert!(matches!(
        block_on(renderer.submit_forward(
            &ForwardCpuPipeline::new(BoundsVisibility),
            snapshot,
            view,
            bounds(),
            &assets
        )),
        Err(RendererError::ForwardPreparation(_))
    ));
    assert_eq!(counts(&native), before);
}

#[test]
fn pbr_compiled_plan_refuses_short_or_retyped_transform_footprints_before_work() {
    let (device, native) = pbr_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let (snapshot, view, assets) =
        scene_and_assets(geometry(true, [0.0, 0.0, 1.0]), RenderTransform::identity());
    let mut frame = pbr_frame(&mut renderer, &snapshot, view, &assets);
    assert_eq!(frame.draws[0].transform_bytes, crate::frame_data::forward::PBR_TRANSFORM_BYTES);
    assert_eq!(
        frame.draws[0].transform_range.end - frame.draws[0].transform_range.start,
        crate::frame_data::forward::PBR_TRANSFORM_BYTES
    );
    let plan = crate::graph::DeferredPlan::compile(
        &device,
        PlanId {
            device: device.id(),
            ordinal: 72_001,
        },
        &frame,
    )
    .unwrap();
    let before = counts(&native);
    frame.draws[0].transform_range.end = frame.draws[0].transform_range.start + 64;
    assert!(matches!(
        plan.validate_frame(&frame),
        Err(Error::InvalidRange)
    ));
    assert_eq!(counts(&native), before);

    frame.draws[0].transform_bytes = 64;
    assert!(matches!(
        plan.validate_frame(&frame),
        Err(Error::InvalidState)
    ));
    assert_eq!(counts(&native), before);
}

#[test]
fn pbr_surface_plan_refuses_short_and_retagged_actual_transform_ranges() {
    let (device, native) = pbr_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).expect("renderer");
    let (snapshot, view, assets) =
        scene_and_assets(geometry(true, [0.0, 0.0, 1.0]), RenderTransform::identity());
    let pipeline = DeferredCpuPipeline::new(BoundsVisibility);
    let selected = pipeline
        .prepare(&snapshot, view, bounds())
        .expect("selected");
    let inputs = prepare_forward(
        &selected,
        &assets,
        &mut fluxel_shader::ShaderComposer::new(),
    )
    .expect("actual PBR input");
    let mut frame = block_on(renderer.prepare_deferred_surface_test_frame(&inputs[0]))
        .expect("actual PBR Surface owners");
    let declaration =
        crate::graph::DeferredPlan::declare_surface_with(&device, &frame, |context| {
            pipeline.contribute_deferred(&selected, context)
        })
        .expect("declared Surface");
    let plan = crate::graph::DeferredPlan::compile_declared(
        &device,
        PlanId {
            device: device.id(),
            ordinal: 81_900,
        },
        declaration,
    )
    .expect("Surface plan");
    plan.validate_surface_frame(&frame)
        .expect("legal frame before mutation");
    let before = counts(&native);
    frame.draws[0].transform_range.end = frame.draws[0].transform_range.start + 64;
    assert!(plan.validate_surface_frame(&frame).is_err());
    assert_eq!(counts(&native), before);
    frame.draws[0].transform_bytes = 64;
    assert!(plan.validate_surface_frame(&frame).is_err());
    assert_eq!(counts(&native), before);
}

#[test]
fn pbr_missing_invalid_or_singular_normals_refuse_before_backend_allocation() {
    let cases = [
        (
            geometry(false, [0.0, 0.0, 1.0]),
            RenderTransform::identity(),
        ),
        (
            geometry(true, [f32::NAN, 0.0, 1.0]),
            RenderTransform::identity(),
        ),
        (geometry(true, [0.0; 3]), RenderTransform::identity()),
        (
            geometry(true, [f32::INFINITY, 0.0, 1.0]),
            RenderTransform::identity(),
        ),
        (
            geometry(true, [0.0, 0.0, 1.0]),
            RenderTransform::from_columns([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ])
            .unwrap(),
        ),
    ];
    for (geometry, transform) in cases {
        let (device, native) = pbr_device(Script::AcceptedExact);
        let mut renderer = Renderer::new(device).unwrap();
        let (snapshot, view, assets) = scene_and_assets(geometry, transform);
        let before = counts(&native);
        assert!(matches!(
            block_on(renderer.submit_forward(
                &ForwardCpuPipeline::new(BoundsVisibility),
                snapshot.clone(),
                view,
                bounds(),
                &assets,
            )),
            Err(RendererError::ForwardPreparation(_))
        ));
        assert_eq!(counts(&native), before);
        assert!(matches!(
            block_on(renderer.submit_deferred(
                &DeferredCpuPipeline::new(BoundsVisibility),
                snapshot,
                view,
                bounds(),
                &assets,
            )),
            Err(RendererError::ForwardPreparation(_))
        ));
        assert_eq!(counts(&native), before);
    }
}

#[test]
fn normal_map_tangent_payloads_refuse_before_backend_work_in_both_consumers() {
    let overflowing_world = RenderTransform::from_columns([
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    .expect("finite affine world transform");
    let cases = [
        ([f32::NAN, 0.0, 0.0, 1.0], RenderTransform::identity()),
        ([0.0, 0.0, 0.0, 1.0], RenderTransform::identity()),
        ([1.0, 0.0, 0.0, 0.0], RenderTransform::identity()),
        ([0.0, 0.0, 1.0, 1.0], RenderTransform::identity()),
        ([f32::MAX, 0.0, 0.0, 1.0], overflowing_world),
    ];
    for (tangent, transform) in cases {
        let (device, native) = pbr_device(Script::AcceptedExact);
        let mut renderer = Renderer::new(device).expect("renderer");
        let (snapshot, view, assets) = scene_and_assets_with_material(
            uv0_tangent_geometry(tangent),
            transform,
            uv0_normal_node_material(),
        );
        let before = counts(&native);
        for deferred in [false, true] {
            let result = if deferred {
                block_on(renderer.submit_deferred(
                    &DeferredCpuPipeline::new(BoundsVisibility),
                    snapshot.clone(),
                    view,
                    bounds(),
                    &assets,
                ))
            } else {
                block_on(renderer.submit_forward(
                    &ForwardCpuPipeline::new(BoundsVisibility),
                    snapshot.clone(),
                    view,
                    bounds(),
                    &assets,
                ))
            };
            assert!(matches!(
                result,
                Err(RendererError::ForwardPreparation(
                    crate::ForwardPreparationError::InvalidTangent { .. }
                ))
            ));
            assert_eq!(counts(&native), before);
        }
    }
}
