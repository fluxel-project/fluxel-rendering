//! Surface compute-product replacement guards use actual allocated owners.
use super::*;
use crate::{FramePipeline, prepare_forward};
use fluxel_rhi::api::PlanId;

#[test]
fn surface_producer_rejects_replaced_same_device_and_foreign_products_before_work() {
    let (device, native) = deformation_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let (scene, view, assets) = deformation_scene();
    let snapshot = scene.snapshot();
    let pipeline = DeferredCpuPipeline::new(BoundsVisibility);
    let selected = pipeline.prepare(&snapshot, view, bounds()).unwrap();
    let inputs = prepare_forward(
        &selected,
        &assets,
        &mut fluxel_shader::ShaderComposer::new(),
    )
    .unwrap();
    let request = PositionDeformation {
        source: fluxel_shader::SourceUnit::new("owner-deformation", SOURCE),
        parameters: [0.0; 4],
    };
    let (_, feature, _) = bind(&scene, &snapshot, SOURCE, [0.0; 4]);
    let mut original =
        block_on(renderer.prepare_deformed_surface_test_frame(&inputs[0], &request)).unwrap();
    let declaration =
        crate::graph::DeferredPlan::declare_surface_with(&device, &original, |context| {
            pipeline.contribute_deformed(&selected, &feature, context)
        })
        .unwrap();
    let plan = crate::graph::DeferredPlan::compile_declared(
        &device,
        PlanId {
            device: device.id(),
            ordinal: 82_001,
        },
        declaration,
    )
    .unwrap();
    plan.validate_surface_frame(&original)
        .expect("positive control before replacement");

    let replacement =
        block_on(renderer.prepare_deformed_surface_test_frame(&inputs[0], &request)).unwrap();
    let before = counts(&native);
    original.deformation.as_mut().unwrap().product = replacement.deformation.unwrap().product;
    assert!(matches!(
        plan.validate_surface_frame(&original),
        Err(Error::InvalidRange)
    ));
    assert_eq!(counts(&native), before);

    let mut original =
        block_on(renderer.prepare_deformed_surface_test_frame(&inputs[0], &request)).unwrap();
    plan.validate_surface_frame(&original)
        .expect("fresh legal owner before foreign replacement");
    let (foreign_device, _) = deformation_device(Script::AcceptedExact);
    let mut foreign = Renderer::new(foreign_device).unwrap();
    let foreign_frame =
        block_on(foreign.prepare_deformed_surface_test_frame(&inputs[0], &request)).unwrap();
    let before = counts(&native);
    original.deformation.as_mut().unwrap().product = foreign_frame.deformation.unwrap().product;
    assert!(matches!(
        plan.validate_surface_frame(&original),
        Err(Error::WrongDevice)
    ));
    assert_eq!(counts(&native), before);
}
