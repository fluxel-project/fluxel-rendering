//! Actual-owner admission tests for the provisional Deferred graph plan.

use super::*;

use fluxel_rhi::api::{Device, Error, PlanId};

use crate::{
    BoundsVisibility, ForwardCpuPipeline, ForwardDrawInput, FramePipeline, PreparedFrame,
    prepare_forward,
};

fn prepared_input<'a>(
    snapshot: &'a crate::FrameSnapshot<ForwardResources>,
    view: crate::RenderViewId,
    assets: &'a ForwardAssets,
) -> (PreparedFrame<'a, ForwardResources>, Vec<ForwardDrawInput>) {
    let pipeline = ForwardCpuPipeline::new(BoundsVisibility);
    let selected = pipeline
        .prepare(snapshot, view, bounds())
        .expect("selected frame");
    let inputs = prepare_forward(&selected, assets, &mut fluxel_shader::ShaderComposer::new())
        .expect("prepared Forward input");
    (selected, inputs)
}

fn deferred_frame(
    renderer: &mut Renderer<WitnessBackend>,
    snapshot: &crate::FrameSnapshot<ForwardResources>,
    view: crate::RenderViewId,
    assets: &ForwardAssets,
) -> crate::residency::DeferredFrame<WitnessBackend> {
    let (_selected, inputs) = prepared_input(snapshot, view, assets);
    let [input] = inputs.as_slice() else {
        panic!("fixture has one draw");
    };
    block_on(renderer.prepare_deferred_test_frame(input)).expect("prepared Deferred frame")
}

fn plan(
    device: &Device<WitnessBackend>,
    ordinal: u64,
    frame: &crate::residency::DeferredFrame<WitnessBackend>,
) -> crate::graph::DeferredPlan<WitnessBackend> {
    crate::graph::DeferredPlan::compile(
        device,
        PlanId {
            device: device.id(),
            ordinal,
        },
        frame,
    )
    .expect("compile Deferred plan")
}

fn execution_counts(native: &Native) -> (usize, usize) {
    (
        native.map_calls.load(Ordering::SeqCst),
        native.command_calls.load(Ordering::SeqCst),
    )
}

#[test]
fn deferred_plan_rejects_foreign_actual_gbuffer_owner_before_mapping_or_recording() {
    let (device, native) = deferred_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).expect("renderer");
    let (snapshot, view, assets) = scene_and_assets();
    let mut frame = deferred_frame(&mut renderer, &snapshot, view, &assets);
    let plan = plan(&device, 71_001, &frame);
    let (other_device, _) = deferred_device(Script::AcceptedExact);
    frame.gbuffer =
        crate::residency::plan_deferred_gbuffer_with_effects(&other_device, frame.extent, false)
            .expect("other-device G-buffer plan")
            .allocate_with_effects(&other_device)
            .expect("other-device G-buffer allocation")
            .0;

    let before = execution_counts(&native);
    assert!(matches!(
        plan.validate_frame(&frame),
        Err(Error::WrongDevice)
    ));
    assert_eq!(execution_counts(&native), before);
}

#[test]
fn deferred_plan_rejects_foreign_scene_color_before_composite_recording() {
    let (device, native) = deferred_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).expect("renderer");
    let (snapshot, view, assets) = scene_and_assets();
    let mut frame = deferred_frame(&mut renderer, &snapshot, view, &assets);
    let plan = plan(&device, 71_006, &frame);
    let (other_device, _) = deferred_device(Script::AcceptedExact);
    frame.scene_color =
        crate::residency::allocate_deferred_scene_color(&other_device, frame.extent)
            .expect("other-device scene-color allocation");

    let before = execution_counts(&native);
    assert!(matches!(
        plan.validate_frame(&frame),
        Err(Error::WrongDevice)
    ));
    assert_eq!(execution_counts(&native), before);
}

#[test]
fn deferred_plan_rejects_unfilterable_actual_scene_color_before_composite_recording() {
    let (device, native) = deferred_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).expect("renderer");
    let (snapshot, view, assets) = scene_and_assets();
    let mut frame = deferred_frame(&mut renderer, &snapshot, view, &assets);
    let plan = plan(&device, 71_007, &frame);
    frame.scene_color.filterable = false;

    let before = execution_counts(&native);
    assert!(matches!(
        plan.validate_frame(&frame),
        Err(Error::Unsupported("filterable Deferred scene-color"))
    ));
    assert_eq!(execution_counts(&native), before);
}

#[test]
fn deferred_plan_rejects_actual_gbuffer_format_and_extent_owners_before_execution() {
    let (device, native) = deferred_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).expect("renderer");
    let (snapshot, view, assets) = scene_and_assets();

    let mut wrong_format = deferred_frame(&mut renderer, &snapshot, view, &assets);
    let format_plan = plan(&device, 71_002, &wrong_format);
    let ([_, _, depth], _) =
        crate::residency::plan_deferred_gbuffer_with_effects(&device, wrong_format.extent, false)
            .expect("G-buffer plan")
            .allocate_with_effects(&device)
            .expect("G-buffer allocation");
    wrong_format.gbuffer[0] = depth;
    let before = execution_counts(&native);
    assert!(matches!(
        format_plan.validate_frame(&wrong_format),
        Err(Error::InvalidState)
    ));
    assert_eq!(execution_counts(&native), before);

    let mut wrong_extent = deferred_frame(&mut renderer, &snapshot, view, &assets);
    let extent_plan = plan(&device, 71_003, &wrong_extent);
    let ([albedo, _, _], _) =
        crate::residency::plan_deferred_gbuffer_with_effects(&device, [2, 1], false)
            .expect("different-extent G-buffer plan")
            .allocate_with_effects(&device)
            .expect("different-extent G-buffer allocation");
    wrong_extent.gbuffer[0] = albedo;
    let before = execution_counts(&native);
    assert!(matches!(
        extent_plan.validate_frame(&wrong_extent),
        Err(Error::InvalidState)
    ));
    assert_eq!(execution_counts(&native), before);
}

#[test]
fn deferred_plan_rejects_short_or_misaligned_dynamic_uniform_ranges_before_execution() {
    let (device, native) = deferred_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).expect("renderer");
    let (snapshot, view, assets) = scene_and_assets();

    let mut short_tint = deferred_frame(&mut renderer, &snapshot, view, &assets);
    let short_tint_plan = plan(&device, 71_004, &short_tint);
    short_tint.draws[0].tint_range = 0..15;
    let before = execution_counts(&native);
    assert!(matches!(
        short_tint_plan.validate_frame(&short_tint),
        Err(Error::InvalidRange) | Err(Error::InvalidState)
    ));
    assert_eq!(execution_counts(&native), before);

    let mut short_transform = deferred_frame(&mut renderer, &snapshot, view, &assets);
    let short_transform_plan = plan(&device, 71_005, &short_transform);
    short_transform.draws[0].transform_range = 0..63;
    let before = execution_counts(&native);
    assert!(matches!(
        short_transform_plan.validate_frame(&short_transform),
        Err(Error::InvalidRange) | Err(Error::InvalidState)
    ));
    assert_eq!(execution_counts(&native), before);

    let mut misaligned_tint = deferred_frame(&mut renderer, &snapshot, view, &assets);
    let misaligned_tint_plan = plan(&device, 71_006, &misaligned_tint);
    misaligned_tint.draws[0].tint_range = 4..20;
    let before = execution_counts(&native);
    assert!(matches!(
        misaligned_tint_plan.validate_frame(&misaligned_tint),
        Err(Error::InvalidRange) | Err(Error::InvalidState)
    ));
    assert_eq!(execution_counts(&native), before);

    let mut misaligned_transform = deferred_frame(&mut renderer, &snapshot, view, &assets);
    let misaligned_transform_plan = plan(&device, 71_007, &misaligned_transform);
    misaligned_transform.draws[0].transform_range = 4..68;
    let before = execution_counts(&native);
    assert!(matches!(
        misaligned_transform_plan.validate_frame(&misaligned_transform),
        Err(Error::InvalidRange) | Err(Error::InvalidState)
    ));
    assert_eq!(execution_counts(&native), before);
}
