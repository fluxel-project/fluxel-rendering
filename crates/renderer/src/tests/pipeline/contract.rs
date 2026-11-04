use super::{
    BoundsVisibility, ForwardCpuPipeline, FramePipeline, FrameTopology, PipelineError,
    VisibilityProvider,
};
use crate::{
    FrameSnapshot, RenderBounds, RenderChange, RenderChangeSet, RenderId, RenderRecord,
    RenderScene, RenderTransform, RenderView,
};

type Scene = RenderScene<u8>;

fn record(minimum: [f32; 3], maximum: [f32; 3]) -> RenderRecord<u8> {
    RenderRecord {
        resources: 0,
        transform: RenderTransform::identity(),
        bounds: RenderBounds::new(minimum, maximum).unwrap(),
    }
}

fn populated_scene() -> (Scene, Vec<RenderId>, crate::RenderViewId) {
    let mut scene = Scene::new().unwrap();
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::Insert(record([4.0; 3], [5.0; 3])));
    changes.push(RenderChange::Insert(record([-1.0; 3], [1.0; 3])));
    changes.push(RenderChange::Insert(record([0.5; 3], [2.0; 3])));
    changes.push(RenderChange::InsertView(
        RenderView::new(RenderTransform::identity(), [64, 64]).unwrap(),
    ));
    let receipt = scene.apply(&changes).unwrap();
    (
        scene,
        receipt.inserted().to_vec(),
        receipt.inserted_views()[0],
    )
}

#[test]
fn bounds_provider_selects_intersections_in_deterministic_identity_order() {
    let (scene, ids, view) = populated_scene();
    let snapshot = scene.snapshot();
    let pipeline = ForwardCpuPipeline::new(BoundsVisibility);
    let frame = pipeline
        .prepare(
            &snapshot,
            view,
            RenderBounds::new([-2.0; 3], [1.0; 3]).unwrap(),
        )
        .unwrap();

    assert_eq!(frame.candidates(), &[ids[1], ids[2]]);
    assert_eq!(frame.scene_version(), snapshot.version());
    assert_eq!(frame.view(), view);
    assert_eq!(frame.record(ids[1]).unwrap().bounds.minimum(), [-1.0; 3]);
    assert!(matches!(
        frame.record(ids[0]),
        Err(PipelineError::CandidateNotPrepared { .. })
    ));
}

struct Reverse;

impl VisibilityProvider<u8> for Reverse {
    fn visible(
        &self,
        _: &FrameSnapshot<u8>,
        _: RenderBounds,
        candidates: &[RenderId],
    ) -> Result<Vec<RenderId>, PipelineError> {
        Ok(candidates.iter().copied().rev().collect())
    }
}

#[test]
fn pipeline_normalizes_provider_order() {
    let (scene, ids, view) = populated_scene();
    let snapshot = scene.snapshot();
    let frame = ForwardCpuPipeline::new(Reverse)
        .prepare(
            &snapshot,
            view,
            RenderBounds::new([-10.0; 3], [10.0; 3]).unwrap(),
        )
        .unwrap();
    assert_eq!(frame.candidates(), ids);
}

struct Duplicate;

impl VisibilityProvider<u8> for Duplicate {
    fn visible(
        &self,
        _: &FrameSnapshot<u8>,
        _: RenderBounds,
        candidates: &[RenderId],
    ) -> Result<Vec<RenderId>, PipelineError> {
        Ok(vec![candidates[0], candidates[0]])
    }
}

struct ReturnId(RenderId);

impl VisibilityProvider<u8> for ReturnId {
    fn visible(
        &self,
        _: &FrameSnapshot<u8>,
        _: RenderBounds,
        _: &[RenderId],
    ) -> Result<Vec<RenderId>, PipelineError> {
        Ok(vec![self.0])
    }
}

#[test]
fn pipeline_rejects_duplicate_provider_output() {
    let (scene, _, view) = populated_scene();
    let snapshot = scene.snapshot();
    assert!(matches!(
        ForwardCpuPipeline::new(Duplicate).prepare(
            &snapshot,
            view,
            RenderBounds::new([-10.0; 3], [10.0; 3]).unwrap(),
        ),
        Err(PipelineError::DuplicateCandidate { .. })
    ));
}

#[test]
fn pipeline_rejects_foreign_and_stale_provider_records() {
    let (scene, ids, view) = populated_scene();
    let snapshot = scene.snapshot();
    let foreign = populated_scene().1[0];
    let volume = RenderBounds::new([-10.0; 3], [10.0; 3]).unwrap();
    assert!(matches!(
        ForwardCpuPipeline::new(ReturnId(foreign)).prepare(&snapshot, view, volume),
        Err(PipelineError::Scene(crate::SceneError::ForeignId {
            operation: 0
        }))
    ));

    let mut stale_scene = scene;
    let mut remove = RenderChangeSet::new();
    remove.push(RenderChange::Remove(ids[0]));
    stale_scene.apply(&remove).unwrap();
    let stale_snapshot = stale_scene.snapshot();
    assert!(matches!(
        ForwardCpuPipeline::new(ReturnId(ids[0])).prepare(&stale_snapshot, view, volume),
        Err(PipelineError::Scene(crate::SceneError::StaleId {
            operation: 0
        }))
    ));
}

#[test]
fn empty_scene_and_multiple_views_preserve_explicit_preparation_scope() {
    let mut empty = Scene::new().unwrap();
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::InsertView(
        RenderView::new(RenderTransform::identity(), [32, 32]).unwrap(),
    ));
    changes.push(RenderChange::InsertView(
        RenderView::new(RenderTransform::identity(), [96, 48]).unwrap(),
    ));
    let views = empty.apply(&changes).unwrap().inserted_views().to_vec();
    let empty_snapshot = empty.snapshot();
    let pipeline = ForwardCpuPipeline::new(BoundsVisibility);
    let volume = RenderBounds::new([-1.0; 3], [1.0; 3]).unwrap();
    for view in views {
        let frame = pipeline.prepare(&empty_snapshot, view, volume).unwrap();
        assert_eq!(frame.view(), view);
        assert!(frame.candidates().is_empty());
    }

    let (scene, ids, first_view) = populated_scene();
    let mut second_scene = scene;
    let mut add_view = RenderChangeSet::new();
    add_view.push(RenderChange::InsertView(
        RenderView::new(RenderTransform::identity(), [128, 72]).unwrap(),
    ));
    let second_view = second_scene.apply(&add_view).unwrap().inserted_views()[0];
    let snapshot = second_scene.snapshot();
    let wide = RenderBounds::new([-10.0; 3], [10.0; 3]).unwrap();
    let pipeline = ForwardCpuPipeline::new(BoundsVisibility);
    assert_eq!(
        pipeline
            .prepare(&snapshot, first_view, wide)
            .unwrap()
            .candidates(),
        ids
    );
    assert_eq!(
        pipeline
            .prepare(&snapshot, second_view, wide)
            .unwrap()
            .candidates(),
        ids
    );
}

#[test]
fn prepared_frame_retains_its_snapshot_and_refuses_unprepared_foreign_records() {
    let (mut scene, ids, view) = populated_scene();
    let initial = scene.snapshot();
    let volume = RenderBounds::new([-2.0; 3], [1.0; 3]).unwrap();
    let pipeline = ForwardCpuPipeline::new(BoundsVisibility);
    let frame = pipeline.prepare(&initial, view, volume).unwrap();
    let before = frame.record(ids[1]).unwrap().bounds;

    let foreign = populated_scene().1[0];
    assert!(matches!(
        frame.record(foreign),
        Err(PipelineError::CandidateNotPrepared { .. })
    ));

    let mut update = RenderChangeSet::new();
    update.push(RenderChange::Update(ids[1], record([8.0; 3], [9.0; 3])));
    scene.apply(&update).unwrap();
    let current = scene.snapshot();
    assert_eq!(frame.scene_version(), initial.version());
    assert_eq!(frame.record(ids[1]).unwrap().bounds, before);
    assert_eq!(current.record(ids[1]).unwrap().bounds.minimum(), [8.0; 3]);
}

struct CountCalls(std::cell::Cell<u32>);

impl VisibilityProvider<u8> for &CountCalls {
    fn visible(
        &self,
        _: &FrameSnapshot<u8>,
        _: RenderBounds,
        _: &[RenderId],
    ) -> Result<Vec<RenderId>, PipelineError> {
        self.0.set(self.0.get() + 1);
        Ok(Vec::new())
    }
}

#[test]
fn foreign_view_is_rejected_before_the_provider_runs() {
    let (scene, _, _) = populated_scene();
    let other = populated_scene().2;
    let snapshot = scene.snapshot();
    let provider = CountCalls(std::cell::Cell::new(0));
    assert!(matches!(
        ForwardCpuPipeline::new(&provider).prepare(
            &snapshot,
            other,
            RenderBounds::new([-10.0; 3], [10.0; 3]).unwrap(),
        ),
        Err(PipelineError::Scene(_))
    ));
    assert_eq!(provider.0.get(), 0);
}

#[test]
fn execution_boundary_refuses_a_different_pipeline_scene_version_or_view() {
    let (mut scene, _, view) = populated_scene();
    let original = scene.snapshot();
    let pipeline = ForwardCpuPipeline::new(BoundsVisibility);
    let volume = RenderBounds::new([-10.0; 3], [10.0; 3]).unwrap();
    let prepared = pipeline.prepare(&original, view, volume).unwrap();
    prepared.validate_for(&original, view).unwrap();
    let (foreign, _, foreign_view) = populated_scene();
    assert!(matches!(
        prepared.validate_for(&foreign.snapshot(), foreign_view),
        Err(PipelineError::PreparedFrameMismatch)
    ));
    let mut change = RenderChangeSet::new();
    change.push(RenderChange::InsertView(
        RenderView::new(RenderTransform::identity(), [1, 1]).unwrap(),
    ));
    let other_view = scene.apply(&change).unwrap().inserted_views()[0];
    assert!(matches!(
        prepared.validate_for(&original, other_view),
        Err(PipelineError::PreparedFrameMismatch)
    ));
    assert!(matches!(
        prepared.validate_for(&scene.snapshot(), view),
        Err(PipelineError::PreparedFrameMismatch)
    ));
}

struct ClearPipeline(ForwardCpuPipeline<BoundsVisibility>);

impl FramePipeline<u8> for ClearPipeline {
    fn prepare<'a>(
        &self,
        snapshot: &'a FrameSnapshot<u8>,
        view: crate::RenderViewId,
        visibility: RenderBounds,
    ) -> Result<super::PreparedFrame<'a, u8>, PipelineError> {
        self.0.prepare(snapshot, view, visibility)
    }

    fn topology(&self, _: &super::PreparedFrame<'_, u8>) -> Result<FrameTopology, PipelineError> {
        Ok(FrameTopology::ClearOnly)
    }

    fn contribute<B: fluxel_rhi::backend::ResourceBackend>(
        &self,
        _: &super::PreparedFrame<'_, u8>,
        context: &mut super::ForwardPhaseContext<'_, '_, '_, B>,
    ) -> Result<(), PipelineError> {
        context.clear(|pass| {
            pass.depends_on_uploads()?;
            pass.write_main_color()
        })
    }
}

#[test]
fn frame_pipeline_contributes_only_a_fixed_renderer_product() {
    let (scene, _, view) = populated_scene();
    let snapshot = scene.snapshot();
    let volume = RenderBounds::new([-10.0; 3], [10.0; 3]).unwrap();
    let opaque = ForwardCpuPipeline::new(BoundsVisibility)
        .prepare(&snapshot, view, volume)
        .unwrap();
    assert_eq!(
        ForwardCpuPipeline::new(BoundsVisibility)
            .topology(&opaque)
            .unwrap(),
        FrameTopology::OpaqueForward
    );

    let clear_pipeline = ClearPipeline(ForwardCpuPipeline::new(BoundsVisibility));
    let clear = clear_pipeline.prepare(&snapshot, view, volume).unwrap();
    assert_eq!(
        clear_pipeline.topology(&clear).unwrap(),
        FrameTopology::ClearOnly
    );
}
