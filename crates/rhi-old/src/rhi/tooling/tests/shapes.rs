//! No native shapes (rhi-design sections 54, 56, and 58.8).
//!
//! What these tests can check structurally is the part that is checkable from
//! the portable side: every identity an event, a definition, or a command
//! carries is an `ObjectId`, an `AcquiredFrameId`, or a runtime id the artifact
//! layer remaps — never a native handle, pointer, or GPU address — and the
//! relations a capture has to rebuild are present rather than summarized into
//! counts that cannot be replayed.

use std::sync::Arc;

use super::{FakeDevice, Observed, Recorder, definition_id, presentation_target, subscribe, test_identity};
use crate::rhi::binding::{BindGroupLayoutDescriptor, BindingSlotId};
use crate::rhi::command::{ClearValueClass, ColorClearValue, LoadOp, StoreOp};
use crate::rhi::format::{SubmissionLaneId, TextureFormat};
use crate::rhi::platform::{Label, next_object_id};
use crate::rhi::presentation::{AcquiredFrameId, PresentationConfiguration};
use crate::rhi::resource::{
    BufferRange, Extent3d, TextureAspects, TextureViewDescriptor, TextureViewDimension,
};
use crate::rhi::submission::{
    CompletionPoint, PlanPoint, SubmissionBatchId, SubmissionPlanId, SubmissionPoint,
};
use crate::rhi::tooling::{
    CapturedBindGroupDefinition, CapturedBindGroupEntry, CapturedBindingResource,
    CapturedColorAttachment, CapturedColorAttachmentView, CapturedDependencySource,
    CapturedObjectDefinition, CapturedPlanDependency, CapturedPresentPlan, CapturedRasterScope,
    CapturedSubmissionBatch, CapturedSubmissionPlan, CapturedSubmissionReceipt, SemanticEvent,
    SemanticObserver,
};

/// Every `ObjectId` reachable from a definition.
///
/// A definition's dependency graph is walkable through these and nothing else,
/// which is what makes it reconstructible without a live handle.
fn definition_ids(definition: &CapturedObjectDefinition) -> Vec<u64> {
    let ids: Vec<u64> = match definition {
        CapturedObjectDefinition::TextureView { texture, .. } => vec![texture.as_u64()],
        CapturedObjectDefinition::BindGroup { definition, .. } => {
            let mut ids = vec![definition.layout.as_u64()];
            for entry in &definition.entries {
                match &entry.resource {
                    CapturedBindingResource::Buffer { buffer, .. } => {
                        ids.push(buffer.as_u64());
                    }
                    CapturedBindingResource::TextureView { view } => ids.push(view.as_u64()),
                    CapturedBindingResource::Sampler { sampler } => ids.push(sampler.as_u64()),
                    CapturedBindingResource::BufferArray(values) => {
                        ids.extend(values.iter().map(|(buffer, _)| buffer.as_u64()));
                    }
                    CapturedBindingResource::TextureViewArray(values) => {
                        ids.extend(values.iter().map(|view| view.as_u64()));
                    }
                    CapturedBindingResource::SamplerArray(values) => {
                        ids.extend(values.iter().map(|sampler| sampler.as_u64()));
                    }
                }
            }
            ids
        }
        CapturedObjectDefinition::PipelineInterface { definition, .. } => {
            definition.groups.iter().map(|group| group.as_u64()).collect()
        }
        CapturedObjectDefinition::ConfiguredPresentation { definition, .. } => {
            vec![definition.target.as_u64()]
        }
        _ => Vec::new(),
    };
    let mut sorted = ids;
    sorted.sort_unstable();
    sorted
}

#[test]
fn an_event_carries_only_logical_identities() {
    let device = FakeDevice::new();
    let target = presentation_target("host-window");
    let target_id = definition_id(&target);
    device.with_object(target_id, target);

    let access = device.access();
    let recorder = Arc::new(Recorder::default());
    let _subscription = subscribe(&access, Arc::clone(&recorder) as Arc<dyn SemanticObserver>);

    let lease_id = next_object_id();
    let frame = AcquiredFrameId::new(access.device_identity(), 7);
    let configuration = PresentationConfiguration::new(TextureFormat::Rgba8Unorm);
    let event = device
        .registry
        .emit_with(|event| SemanticEvent::FrameAcquired {
            event,
            target: target_id,
            configured_presentation: lease_id,
            frame,
            configuration: &configuration,
            extent: Extent3d::d2(320, 200),
        })
        .expect("the emitter is not inside a callback");

    // Every identity a consumer can read out of the event is one the producer
    // supplied as a logical id; there is no other kind of identity in it.
    assert_eq!(
        recorder.events(),
        vec![Observed::Frame {
            event: event.as_u64(),
            target: target_id,
            lease: lease_id,
            frame,
        }]
    );

    // The id the event names as a target is describable without any native
    // presentation handle, through the fixture the host binds itself.
    match access
        .describe_object(target_id)
        .expect("the target is describable")
    {
        CapturedObjectDefinition::PresentationTarget { definition, .. } => {
            assert_eq!(definition.fixture.key, "host-window");
        }
        _ => panic!("the fixture target must stay a fixture target"),
    }
}

#[test]
fn a_definition_graph_is_reachable_through_object_ids_alone() {
    let layout = CapturedObjectDefinition::BindGroupLayout {
        id: next_object_id(),
        descriptor: BindGroupLayoutDescriptor::new(Vec::new()),
    };
    let layout_id = definition_id(&layout);

    let buffer_id = next_object_id();
    let view_texture = next_object_id();
    let view = CapturedObjectDefinition::TextureView {
        id: next_object_id(),
        texture: view_texture,
        descriptor: TextureViewDescriptor::new(
            TextureViewDimension::D2,
            TextureAspects::COLOR,
            0,
            1,
            0,
            1,
        ),
    };
    let view_id = definition_id(&view);

    let group = CapturedObjectDefinition::BindGroup {
        id: next_object_id(),
        definition: CapturedBindGroupDefinition {
            label: Label::none(),
            layout: layout_id,
            entries: vec![
                CapturedBindGroupEntry {
                    slot: BindingSlotId::new(0),
                    resource: CapturedBindingResource::Buffer {
                        buffer: buffer_id,
                        range: BufferRange::new(0, 256),
                    },
                },
                CapturedBindGroupEntry {
                    slot: BindingSlotId::new(1),
                    resource: CapturedBindingResource::TextureView { view: view_id },
                },
            ],
        },
    };

    assert_eq!(definition_ids(&view), vec![view_texture.as_u64()]);

    let mut expected = vec![layout_id.as_u64(), buffer_id.as_u64(), view_id.as_u64()];
    expected.sort_unstable();
    assert_eq!(
        definition_ids(&group),
        expected,
        "a bind group's whole dependency set is expressible as object ids"
    );
}

#[test]
fn a_raster_scope_carries_value_state_rather_than_a_native_attachment() {
    let view = next_object_id();
    let scope = CapturedRasterScope {
        label: Label::new("forward"),
        colors: vec![Some(CapturedColorAttachment {
            view: CapturedColorAttachmentView::TextureView(view),
            load: LoadOp::Clear(ColorClearValue::Float([0.0, 0.0, 0.0, 1.0])),
            store: StoreOp::Store,
            resolve: None,
        })],
        depth_stencil: None,
    };

    // The recorded clear value keeps its numeric class, which a float-only form
    // could not express: a sint or uint attachment clear is not losslessly a
    // float quadruple.
    let attachment = match &scope.colors[0] {
        Some(attachment) => attachment,
        None => panic!("the fixture has one color location"),
    };
    match &attachment.load {
        LoadOp::Clear(value) => assert_eq!(value.class(), ClearValueClass::Float),
        LoadOp::Load => panic!("the fixture clears"),
    }
    match &attachment.view {
        CapturedColorAttachmentView::TextureView(id) => assert_eq!(*id, view),
        CapturedColorAttachmentView::Frame(frame) => {
            panic!("the fixture attaches a texture view, not a frame: {frame:?}")
        }
    }
}

#[test]
fn the_submission_ir_expresses_the_complete_relation_not_counts() {
    let device = test_identity();
    let plan = SubmissionPlanId::new(device, 1);
    let first = PlanPoint::new(plan, SubmissionBatchId::new(0));
    let second = PlanPoint::new(plan, SubmissionBatchId::new(1));

    let captured = CapturedSubmissionPlan {
        device,
        plan,
        batches: vec![
            CapturedSubmissionBatch {
                point: first,
                lane: SubmissionLaneId::new(0),
                work: vec![next_object_id()],
            },
            CapturedSubmissionBatch {
                point: second,
                lane: SubmissionLaneId::new(0),
                work: vec![next_object_id()],
            },
        ],
        dependencies: vec![CapturedPlanDependency {
            before: CapturedDependencySource::PlanPoint(first),
            after: second,
        }],
        presents: Vec::<CapturedPresentPlan>::new(),
    };

    // A count could not say which batch waits for which; this does.
    assert_eq!(captured.batches.len(), 2);
    assert_eq!(captured.dependencies.len(), 1);
    assert_eq!(captured.dependencies[0].after, second);
    assert_ne!(captured.dependencies[0].after, first);

    let receipt = CapturedSubmissionReceipt {
        submitted: SubmissionPoint::new(device, 3),
        overall_completion: CompletionPoint::new(device, 4),
        point_completions: vec![(first, CompletionPoint::new(device, 4))],
        presents: Vec::new(),
    };
    assert_eq!(receipt.point_completions.len(), 1);
    assert_eq!(receipt.overall_completion, CompletionPoint::new(device, 4));
}
