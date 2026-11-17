//! Deferred PBR transform, normal and tangent input validation coverage.
use super::*;

#[test]
fn pbr_compiled_plan_refuses_short_or_retyped_transform_footprints_before_work() {
    let (device, native) = pbr_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let (snapshot, view, assets) =
        scene_and_assets(geometry(true, [0.0, 0.0, 1.0]), RenderTransform::identity());
    let mut frame = pbr_frame(&mut renderer, &snapshot, view, &assets);
    assert_eq!(frame.draw.transform_bytes, 192);
    assert_eq!(
        frame.draw.transform_range.end - frame.draw.transform_range.start,
        192
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
    frame.draw.transform_range.end = frame.draw.transform_range.start + 64;
    assert!(matches!(
        plan.validate_frame(&frame),
        Err(Error::InvalidRange)
    ));
    assert_eq!(counts(&native), before);

    frame.draw.transform_bytes = 64;
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
    frame.draw.transform_range.end = frame.draw.transform_range.start + 64;
    assert!(plan.validate_surface_frame(&frame).is_err());
    assert_eq!(counts(&native), before);
    frame.draw.transform_bytes = 64;
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
