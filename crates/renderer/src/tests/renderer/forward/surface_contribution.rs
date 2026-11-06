//! Surface Forward contribution refusal coverage on fresh and cached plans.

use super::*;

#[derive(Clone, Copy, Debug)]
enum SurfaceRefusal {
    MissingPhase,
    WrongPhase,
    MissingUpload,
    MissingColor,
    RepeatedAccess,
}

fn assert_surface_refusal(case: SurfaceRefusal) {
    let (device, native) = fixture(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let mut surface = witness_surface(&device);
    let stats = crate::preparation::ForwardBatchStats {
        candidates: 0,
        batches: 0,
    };
    let refuse = |context: &mut crate::ForwardPhaseContext<'_, '_, '_, WitnessBackend>| match case {
        SurfaceRefusal::MissingPhase => Ok(()),
        SurfaceRefusal::WrongPhase => context.opaque(|pass| {
            pass.depends_on_uploads()?;
            pass.read_vertex_streams()?;
            pass.read_frame_uniforms()?;
            pass.read_sampled_textures()?;
            pass.write_main_color()
        }),
        SurfaceRefusal::MissingUpload => context.clear(|pass| pass.write_main_color()),
        SurfaceRefusal::MissingColor => context.clear(|pass| pass.depends_on_uploads()),
        SurfaceRefusal::RepeatedAccess => context.clear(|pass| {
            pass.depends_on_uploads()?;
            pass.depends_on_uploads()?;
            pass.write_main_color()
        }),
    };
    let maps = native.map_calls.load(Ordering::SeqCst);
    let commands = native.command_calls.load(Ordering::SeqCst);
    let finishes = native.finishes.load(Ordering::SeqCst);
    assert!(matches!(
        block_on(renderer.execute_forward_surface(
            crate::renderer::PendingInputs::None,
            blank_surface_frame(&device),
            crate::renderer::forward::ForwardPublication::default(),
            crate::FrameTopology::ClearOnly,
            stats,
            block_on(surface.acquire()).unwrap(),
            refuse,
        )),
        Err(RendererError::ForwardPipeline(
            crate::PipelineError::InvalidPhaseContribution
        ))
    ));
    assert_eq!(
        native.map_calls.load(Ordering::SeqCst),
        maps,
        "first {case:?}"
    );
    assert_eq!(
        native.command_calls.load(Ordering::SeqCst),
        commands,
        "first {case:?}"
    );
    assert_eq!(
        native.finishes.load(Ordering::SeqCst),
        finishes,
        "first {case:?}"
    );
    assert_eq!(renderer.pending(), 0, "first {case:?}");
    assert_eq!(renderer.pending(), 0, "first {case:?}");

    let accepted = block_on(renderer.execute_forward_surface(
        crate::renderer::PendingInputs::None,
        blank_surface_frame(&device),
        crate::renderer::forward::ForwardPublication::default(),
        crate::FrameTopology::ClearOnly,
        stats,
        block_on(surface.acquire()).unwrap(),
        |context| context.clear_only(),
    ))
    .unwrap();
    block_on(renderer.complete(&accepted)).unwrap();
    let maps = native.map_calls.load(Ordering::SeqCst);
    let commands = native.command_calls.load(Ordering::SeqCst);
    let finishes = native.finishes.load(Ordering::SeqCst);
    assert!(matches!(
        block_on(renderer.execute_forward_surface(
            crate::renderer::PendingInputs::None,
            blank_surface_frame(&device),
            crate::renderer::forward::ForwardPublication::default(),
            crate::FrameTopology::ClearOnly,
            stats,
            block_on(surface.acquire()).unwrap(),
            refuse,
        )),
        Err(RendererError::ForwardPipeline(
            crate::PipelineError::InvalidPhaseContribution
        ))
    ));
    assert_eq!(
        native.map_calls.load(Ordering::SeqCst),
        maps,
        "cached {case:?}"
    );
    assert_eq!(
        native.command_calls.load(Ordering::SeqCst),
        commands,
        "cached {case:?}"
    );
    assert_eq!(
        native.finishes.load(Ordering::SeqCst),
        finishes,
        "cached {case:?}"
    );
    assert_eq!(renderer.pending(), 0, "cached {case:?}");
}

#[test]
fn surface_contribution_refusals_hold_for_first_and_cached_plan_paths() {
    for case in [
        SurfaceRefusal::MissingPhase,
        SurfaceRefusal::WrongPhase,
        SurfaceRefusal::MissingUpload,
        SurfaceRefusal::MissingColor,
        SurfaceRefusal::RepeatedAccess,
    ] {
        assert_surface_refusal(case);
    }
}

#[derive(Clone, Copy, Debug)]
enum OpaqueSurfaceRefusal {
    MissingVertices,
    MissingUniforms,
    MissingTextures,
    RepeatedAccess,
}

fn assert_surface_opaque_refusal(case: OpaqueSurfaceRefusal) {
    let (device, native) = fixture(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let mut surface = witness_surface(&device);
    let stats = crate::preparation::ForwardBatchStats {
        candidates: 0,
        batches: 0,
    };
    let refuse = |context: &mut crate::ForwardPhaseContext<'_, '_, '_, WitnessBackend>| {
        context.opaque(|pass| match case {
            OpaqueSurfaceRefusal::MissingVertices => {
                pass.depends_on_uploads()?;
                pass.read_frame_uniforms()?;
                pass.read_sampled_textures()?;
                pass.write_main_color()
            }
            OpaqueSurfaceRefusal::MissingUniforms => {
                pass.depends_on_uploads()?;
                pass.read_vertex_streams()?;
                pass.read_sampled_textures()?;
                pass.write_main_color()
            }
            OpaqueSurfaceRefusal::MissingTextures => {
                pass.depends_on_uploads()?;
                pass.read_vertex_streams()?;
                pass.read_frame_uniforms()?;
                pass.write_main_color()
            }
            OpaqueSurfaceRefusal::RepeatedAccess => {
                pass.depends_on_uploads()?;
                pass.read_vertex_streams()?;
                pass.read_frame_uniforms()?;
                pass.read_frame_uniforms()?;
                pass.read_sampled_textures()?;
                pass.write_main_color()
            }
        })
    };
    let maps = native.map_calls.load(Ordering::SeqCst);
    let commands = native.command_calls.load(Ordering::SeqCst);
    let finishes = native.finishes.load(Ordering::SeqCst);
    assert!(matches!(
        block_on(renderer.execute_forward_surface(
            crate::renderer::PendingInputs::None,
            blank_surface_frame(&device),
            crate::renderer::forward::ForwardPublication::default(),
            crate::FrameTopology::OpaqueForward,
            stats,
            block_on(surface.acquire()).unwrap(),
            refuse,
        )),
        Err(RendererError::ForwardPipeline(
            crate::PipelineError::InvalidPhaseContribution
        ))
    ));
    assert_eq!(
        native.map_calls.load(Ordering::SeqCst),
        maps,
        "first {case:?}"
    );
    assert_eq!(
        native.command_calls.load(Ordering::SeqCst),
        commands,
        "first {case:?}"
    );
    assert_eq!(
        native.finishes.load(Ordering::SeqCst),
        finishes,
        "first {case:?}"
    );

    let accepted = block_on(renderer.execute_forward_surface(
        crate::renderer::PendingInputs::None,
        blank_surface_frame(&device),
        crate::renderer::forward::ForwardPublication::default(),
        crate::FrameTopology::OpaqueForward,
        stats,
        block_on(surface.acquire()).unwrap(),
        |context| context.opaque_forward(),
    ))
    .unwrap();
    block_on(renderer.complete(&accepted)).unwrap();
    let maps = native.map_calls.load(Ordering::SeqCst);
    let commands = native.command_calls.load(Ordering::SeqCst);
    let finishes = native.finishes.load(Ordering::SeqCst);
    assert!(matches!(
        block_on(renderer.execute_forward_surface(
            crate::renderer::PendingInputs::None,
            blank_surface_frame(&device),
            crate::renderer::forward::ForwardPublication::default(),
            crate::FrameTopology::OpaqueForward,
            stats,
            block_on(surface.acquire()).unwrap(),
            refuse,
        )),
        Err(RendererError::ForwardPipeline(
            crate::PipelineError::InvalidPhaseContribution
        ))
    ));
    assert_eq!(
        native.map_calls.load(Ordering::SeqCst),
        maps,
        "cached {case:?}"
    );
    assert_eq!(
        native.command_calls.load(Ordering::SeqCst),
        commands,
        "cached {case:?}"
    );
    assert_eq!(
        native.finishes.load(Ordering::SeqCst),
        finishes,
        "cached {case:?}"
    );
    assert_eq!(renderer.pending(), 0, "cached {case:?}");
}

#[test]
fn opaque_surface_access_refusals_hold_for_first_and_cached_plan_paths() {
    for case in [
        OpaqueSurfaceRefusal::MissingVertices,
        OpaqueSurfaceRefusal::MissingUniforms,
        OpaqueSurfaceRefusal::MissingTextures,
        OpaqueSurfaceRefusal::RepeatedAccess,
    ] {
        assert_surface_opaque_refusal(case);
    }
}

#[test]
fn legal_surface_contribution_runs_once_per_frame_while_reusing_the_compiled_plan() {
    let (device, _) = fixture(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let mut surface = witness_surface(&device);
    let calls = std::cell::Cell::new(0);
    let initial = renderer.compilations();
    let stats = crate::preparation::ForwardBatchStats {
        candidates: 0,
        batches: 0,
    };
    for _ in 0..2 {
        let submission = block_on(renderer.execute_forward_surface(
            crate::renderer::PendingInputs::None,
            blank_surface_frame(&device),
            crate::renderer::forward::ForwardPublication::default(),
            crate::FrameTopology::OpaqueForward,
            stats,
            block_on(surface.acquire()).unwrap(),
            |context| {
                calls.set(calls.get() + 1);
                context.opaque_forward()
            },
        ))
        .unwrap();
        block_on(renderer.complete(&submission)).unwrap();
    }
    assert_eq!(calls.get(), 2);
    assert_eq!(renderer.compilations(), initial + 1);
}
