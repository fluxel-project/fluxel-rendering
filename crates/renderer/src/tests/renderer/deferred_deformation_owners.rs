//! Actual compute-product owner replacement tests for Deferred deformation.

use super::*;

use fluxel_rhi::api::{Device, Error, PlanId};

use crate::{BoundsVisibility, ForwardCpuPipeline, FramePipeline, prepare_forward};

fn input<'a>(
    snapshot: &'a crate::FrameSnapshot<ForwardResources>,
    view: crate::RenderViewId,
    assets: &'a ForwardAssets,
) -> crate::ForwardDrawInput {
    let pipeline = ForwardCpuPipeline::new(BoundsVisibility);
    let frame = pipeline.prepare(snapshot, view, bounds()).unwrap();
    let inputs =
        prepare_forward(&frame, assets, &mut fluxel_shader::ShaderComposer::new()).unwrap();
    inputs.into_iter().next().expect("one Deferred draw")
}

fn frame(
    renderer: &mut Renderer<WitnessBackend>,
    snapshot: &crate::FrameSnapshot<ForwardResources>,
    view: crate::RenderViewId,
    assets: &ForwardAssets,
) -> crate::residency::DeferredFrame<WitnessBackend> {
    let input = input(snapshot, view, assets);
    let deformation = PositionDeformation {
        source: fluxel_shader::SourceUnit::new("owner-deformation", SOURCE),
        parameters: [0.0; 4],
    };
    block_on(renderer.prepare_deformed_test_frame(&input, &deformation)).unwrap()
}

fn plan(
    device: &Device<WitnessBackend>,
    ordinal: u64,
    frame: &crate::residency::DeferredFrame<WitnessBackend>,
) -> crate::graph::DeferredPlan<WitnessBackend> {
    crate::graph::DeferredPlan::compile_with(
        device,
        PlanId {
            device: device.id(),
            ordinal,
        },
        frame,
        |context| {
            context.deformation(|pass| {
                pass.read_source()?;
                pass.read_parameters()?;
                pass.write_vertices()
            })?;
            context.geometry(|pass| {
                pass.depends_on_uploads()?;
                pass.read_vertex_streams()?;
                pass.read_frame_uniforms()?;
                pass.read_sampled_textures()?;
                pass.write_gbuffer()
            })?;
            context.lighting(|pass| {
                pass.read_gbuffer()?;
                pass.write_main_color()
            })
        },
    )
    .unwrap()
}

fn execution_counts(native: &Native) -> (usize, usize) {
    (
        native.map_calls.load(Ordering::SeqCst),
        native.command_calls.load(Ordering::SeqCst),
    )
}

#[test]
fn deformation_plan_rejects_replaced_actual_same_device_product_before_execution() {
    let (device, native) = deformation_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let (scene, view, assets) = deformation_scene();
    let snapshot = scene.snapshot();
    let mut actual = frame(&mut renderer, &snapshot, view, &assets);
    let compiled = plan(&device, 72_001, &actual);
    let mut replacement = frame(&mut renderer, &snapshot, view, &assets);
    let product = replacement.deformation.take().unwrap().product;
    actual.deformation.as_mut().unwrap().product = product;

    let before = execution_counts(&native);
    assert!(matches!(
        compiled.validate_frame(&actual),
        Err(Error::InvalidRange)
    ));
    assert_eq!(execution_counts(&native), before);
}

#[test]
fn deformation_plan_rejects_foreign_actual_product_before_execution() {
    let (device, native) = deformation_device(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let (scene, view, assets) = deformation_scene();
    let snapshot = scene.snapshot();
    let mut actual = frame(&mut renderer, &snapshot, view, &assets);
    let compiled = plan(&device, 72_002, &actual);
    let (foreign_device, _) = deformation_device(Script::AcceptedExact);
    let mut foreign_renderer = Renderer::new(foreign_device).unwrap();
    let mut foreign = frame(&mut foreign_renderer, &snapshot, view, &assets);
    let product = foreign.deformation.take().unwrap().product;
    actual.deformation.as_mut().unwrap().product = product;

    let before = execution_counts(&native);
    assert!(matches!(
        compiled.validate_frame(&actual),
        Err(Error::WrongDevice)
    ));
    assert_eq!(execution_counts(&native), before);
}
