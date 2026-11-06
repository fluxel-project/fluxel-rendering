//! Forward contribution tests against a real Graph transaction.

use super::*;

use crate::{
    BoundsVisibility, ForwardCpuPipeline, ForwardPhaseContext, ForwardResources, FramePipeline,
    FrameTopology, PipelineError, RenderBounds, RenderChange, RenderChangeSet, RenderScene,
    RenderTransform, RenderView,
    renderer::forward::ForwardPublication,
    residency::{ForwardFrame, ForwardFrameBuffers, ForwardResidentBuffers},
};
use std::cell::Cell;

struct MismatchedPhasePipeline;

impl FramePipeline<ForwardResources> for MismatchedPhasePipeline {
    fn prepare<'a>(
        &self,
        snapshot: &'a crate::FrameSnapshot<ForwardResources>,
        view: crate::RenderViewId,
        visibility: RenderBounds,
    ) -> std::result::Result<crate::PreparedFrame<'a, ForwardResources>, PipelineError> {
        ForwardCpuPipeline::new(BoundsVisibility).prepare(snapshot, view, visibility)
    }

    fn topology(
        &self,
        _: &crate::PreparedFrame<'_, ForwardResources>,
    ) -> std::result::Result<FrameTopology, PipelineError> {
        Ok(FrameTopology::OpaqueForward)
    }

    fn contribute<B: fluxel_rhi::backend::ResourceBackend>(
        &self,
        _: &crate::PreparedFrame<'_, ForwardResources>,
        context: &mut ForwardPhaseContext<'_, '_, '_, B>,
    ) -> std::result::Result<(), PipelineError> {
        context.clear(|pass| {
            pass.depends_on_uploads()?;
            pass.write_main_color()
        })
    }
}

struct ReplayedPhasePipeline;

impl FramePipeline<ForwardResources> for ReplayedPhasePipeline {
    fn prepare<'a>(
        &self,
        snapshot: &'a crate::FrameSnapshot<ForwardResources>,
        view: crate::RenderViewId,
        visibility: RenderBounds,
    ) -> std::result::Result<crate::PreparedFrame<'a, ForwardResources>, PipelineError> {
        ForwardCpuPipeline::new(BoundsVisibility).prepare(snapshot, view, visibility)
    }

    fn contribute<B: fluxel_rhi::backend::ResourceBackend>(
        &self,
        _: &crate::PreparedFrame<'_, ForwardResources>,
        context: &mut ForwardPhaseContext<'_, '_, '_, B>,
    ) -> std::result::Result<(), PipelineError> {
        context.opaque(|pass| {
            pass.depends_on_uploads()?;
            pass.read_vertex_streams()?;
            pass.read_frame_uniforms()?;
            pass.read_sampled_textures()?;
            pass.write_main_color()
        })?;
        context.opaque(|pass| {
            pass.depends_on_uploads()?;
            pass.read_vertex_streams()?;
            pass.read_frame_uniforms()?;
            pass.read_sampled_textures()?;
            pass.write_main_color()
        })
    }
}

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

#[test]
fn invalid_phase_contribution_is_rejected_by_the_real_graph_before_mapping_or_execution() {
    let (device, native) = fixture(Script::AcceptedExact);
    let mut scene = RenderScene::<ForwardResources>::new().unwrap();
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::InsertView(
        RenderView::new(RenderTransform::identity(), [1, 1]).unwrap(),
    ));
    let view = scene.apply(&changes).unwrap().inserted_views()[0];
    let snapshot = scene.snapshot();
    let pipeline = MismatchedPhasePipeline;
    let frame = pipeline
        .prepare(
            &snapshot,
            view,
            RenderBounds::new([-1.0; 3], [1.0; 3]).unwrap(),
        )
        .unwrap();
    let forward = blank_frame(&device);
    let maps = native.map_calls.load(Ordering::SeqCst);
    let commands = native.command_calls.load(Ordering::SeqCst);
    let finishes = native.finishes.load(Ordering::SeqCst);
    assert!(matches!(
        crate::graph::ForwardPlan::compile_with(
            &device,
            PlanId {
                device: device.id(),
                ordinal: 81_001
            },
            &forward,
            FrameTopology::OpaqueForward,
            |context| pipeline.contribute(&frame, context),
        ),
        Err(crate::graph::ForwardPlanBuildError::Pipeline(
            PipelineError::InvalidPhaseContribution
        ))
    ));
    assert_eq!(native.map_calls.load(Ordering::SeqCst), maps);
    assert_eq!(native.command_calls.load(Ordering::SeqCst), commands);
    assert_eq!(native.finishes.load(Ordering::SeqCst), finishes);
}

#[test]
fn repeated_phase_contribution_is_rejected_by_the_real_graph_before_mapping_or_execution() {
    let (device, native) = fixture(Script::AcceptedExact);
    let mut scene = RenderScene::<ForwardResources>::new().unwrap();
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::InsertView(
        RenderView::new(RenderTransform::identity(), [1, 1]).unwrap(),
    ));
    let view = scene.apply(&changes).unwrap().inserted_views()[0];
    let snapshot = scene.snapshot();
    let pipeline = ReplayedPhasePipeline;
    let frame = pipeline
        .prepare(
            &snapshot,
            view,
            RenderBounds::new([-1.0; 3], [1.0; 3]).unwrap(),
        )
        .unwrap();
    let forward = blank_frame(&device);
    let maps = native.map_calls.load(Ordering::SeqCst);
    let commands = native.command_calls.load(Ordering::SeqCst);
    let finishes = native.finishes.load(Ordering::SeqCst);
    assert!(matches!(
        crate::graph::ForwardPlan::compile_with(
            &device,
            PlanId {
                device: device.id(),
                ordinal: 81_002
            },
            &forward,
            FrameTopology::OpaqueForward,
            |context| pipeline.contribute(&frame, context),
        ),
        Err(crate::graph::ForwardPlanBuildError::Pipeline(
            PipelineError::InvalidPhaseContribution
        ))
    ));
    assert_eq!(native.map_calls.load(Ordering::SeqCst), maps);
    assert_eq!(native.command_calls.load(Ordering::SeqCst), commands);
    assert_eq!(native.finishes.load(Ordering::SeqCst), finishes);
}

#[test]
fn cached_forward_plan_declares_the_current_contribution_once_before_reuse() {
    let (device, _) = fixture(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let calls = Cell::new(0);
    let stats = crate::preparation::ForwardBatchStats {
        candidates: 0,
        batches: 0,
    };
    let initial = renderer.compilations();

    let first = block_on(renderer.execute_forward_with_contribution(
        crate::renderer::PendingInputs::None,
        blank_frame(&device),
        ForwardPublication::default(),
        FrameTopology::OpaqueForward,
        stats,
        |context| {
            calls.set(calls.get() + 1);
            context.opaque_forward()
        },
    ))
    .unwrap();
    assert!(block_on(renderer.readback(&first)).is_err());
    assert_eq!(calls.get(), 1);
    assert_eq!(renderer.compilations(), initial + 1);

    let second = block_on(renderer.execute_forward_with_contribution(
        crate::renderer::PendingInputs::None,
        blank_frame(&device),
        ForwardPublication::default(),
        FrameTopology::OpaqueForward,
        stats,
        |context| {
            calls.set(calls.get() + 1);
            context.opaque_forward()
        },
    ))
    .unwrap();
    assert!(block_on(renderer.readback(&second)).is_err());
    assert_eq!(calls.get(), 2);
    assert_eq!(renderer.compilations(), initial + 1);
}

#[test]
fn cached_forward_plan_rejects_the_current_illegal_contribution_before_mapping_or_execution() {
    let (device, native) = fixture(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let stats = crate::preparation::ForwardBatchStats {
        candidates: 0,
        batches: 0,
    };
    let first = block_on(renderer.execute_forward_with_contribution(
        crate::renderer::PendingInputs::None,
        blank_frame(&device),
        ForwardPublication::default(),
        FrameTopology::OpaqueForward,
        stats,
        |context| context.opaque_forward(),
    ))
    .unwrap();
    assert!(block_on(renderer.readback(&first)).is_err());
    let commands = native.command_calls.load(Ordering::SeqCst);
    let finishes = native.finishes.load(Ordering::SeqCst);
    let maps = native.map_calls.load(Ordering::SeqCst);

    assert!(matches!(
        block_on(renderer.execute_forward_with_contribution(
            crate::renderer::PendingInputs::None,
            blank_frame(&device),
            ForwardPublication::default(),
            FrameTopology::OpaqueForward,
            stats,
            |context| context.clear(|pass| {
                pass.depends_on_uploads()?;
                pass.write_main_color()
            }),
        )),
        Err(RendererError::ForwardPipeline(
            PipelineError::InvalidPhaseContribution
        ))
    ));
    assert_eq!(native.map_calls.load(Ordering::SeqCst), maps);
    assert_eq!(native.command_calls.load(Ordering::SeqCst), commands);
    assert_eq!(native.finishes.load(Ordering::SeqCst), finishes);
    assert_eq!(renderer.pending(), 0);
}

#[derive(Clone, Copy, Debug)]
enum OpaqueRefusal {
    MissingPhase,
    WrongPhase,
    MissingUpload,
    MissingVertices,
    MissingUniforms,
    MissingTextures,
    MissingColor,
    RepeatedAccess,
}

fn assert_opaque_refusal(case: OpaqueRefusal) {
    let (device, native) = fixture(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let stats = crate::preparation::ForwardBatchStats {
        candidates: 0,
        batches: 0,
    };
    let refuse = |context: &mut ForwardPhaseContext<'_, '_, '_, WitnessBackend>| match case {
        OpaqueRefusal::MissingPhase => Ok(()),
        OpaqueRefusal::WrongPhase => context.clear(|pass| {
            pass.depends_on_uploads()?;
            pass.write_main_color()
        }),
        OpaqueRefusal::MissingUpload => context.opaque(|pass| {
            pass.read_vertex_streams()?;
            pass.read_frame_uniforms()?;
            pass.read_sampled_textures()?;
            pass.write_main_color()
        }),
        OpaqueRefusal::MissingVertices => context.opaque(|pass| {
            pass.depends_on_uploads()?;
            pass.read_frame_uniforms()?;
            pass.read_sampled_textures()?;
            pass.write_main_color()
        }),
        OpaqueRefusal::MissingUniforms => context.opaque(|pass| {
            pass.depends_on_uploads()?;
            pass.read_vertex_streams()?;
            pass.read_sampled_textures()?;
            pass.write_main_color()
        }),
        OpaqueRefusal::MissingTextures => context.opaque(|pass| {
            pass.depends_on_uploads()?;
            pass.read_vertex_streams()?;
            pass.read_frame_uniforms()?;
            pass.write_main_color()
        }),
        OpaqueRefusal::MissingColor => context.opaque(|pass| {
            pass.depends_on_uploads()?;
            pass.read_vertex_streams()?;
            pass.read_frame_uniforms()?;
            pass.read_sampled_textures()
        }),
        OpaqueRefusal::RepeatedAccess => context.opaque(|pass| {
            pass.depends_on_uploads()?;
            pass.read_vertex_streams()?;
            pass.read_frame_uniforms()?;
            pass.read_frame_uniforms()?;
            pass.read_sampled_textures()?;
            pass.write_main_color()
        }),
    };
    let maps = native.map_calls.load(Ordering::SeqCst);
    let commands = native.command_calls.load(Ordering::SeqCst);
    let finishes = native.finishes.load(Ordering::SeqCst);
    assert!(matches!(
        block_on(renderer.execute_forward_with_contribution(
            crate::renderer::PendingInputs::None,
            blank_frame(&device),
            ForwardPublication::default(),
            FrameTopology::OpaqueForward,
            stats,
            refuse,
        )),
        Err(RendererError::ForwardPipeline(
            PipelineError::InvalidPhaseContribution
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

    let accepted = block_on(renderer.execute_forward_with_contribution(
        crate::renderer::PendingInputs::None,
        blank_frame(&device),
        ForwardPublication::default(),
        FrameTopology::OpaqueForward,
        stats,
        |context| context.opaque_forward(),
    ))
    .unwrap();
    assert!(block_on(renderer.readback(&accepted)).is_err());
    let maps = native.map_calls.load(Ordering::SeqCst);
    let commands = native.command_calls.load(Ordering::SeqCst);
    let finishes = native.finishes.load(Ordering::SeqCst);
    assert!(matches!(
        block_on(renderer.execute_forward_with_contribution(
            crate::renderer::PendingInputs::None,
            blank_frame(&device),
            ForwardPublication::default(),
            FrameTopology::OpaqueForward,
            stats,
            refuse,
        )),
        Err(RendererError::ForwardPipeline(
            PipelineError::InvalidPhaseContribution
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
fn opaque_contribution_refusals_hold_for_first_and_cached_plan_paths() {
    for case in [
        OpaqueRefusal::MissingPhase,
        OpaqueRefusal::WrongPhase,
        OpaqueRefusal::MissingUpload,
        OpaqueRefusal::MissingVertices,
        OpaqueRefusal::MissingUniforms,
        OpaqueRefusal::MissingTextures,
        OpaqueRefusal::MissingColor,
        OpaqueRefusal::RepeatedAccess,
    ] {
        assert_opaque_refusal(case);
    }
}
