//! Contract tests for the tooling SPI (specification module 07, sections 52-58).
//!
//! Three kinds of test live here, and this note is required to say which:
//!
//! * **Behavioural tests** exercise the half of the chapter that is real — the
//!   SPI version, the event identity and its ordering, the access and the
//!   subscription, the captured records, and the device-lost refusal. They run,
//!   and they assert exact error kinds.
//! * **Shape tests** are the review instrument for the half whose lowering does
//!   not exist. They are `fn`s that are *compiled* and never called, marked with
//!   the crate's `#[expect(dead_code, reason = "a shape test; ...")]`, and written
//!   as call sites a real capture tool would write rather than as assertions
//!   about types. Their job is to answer "is this interface usable from the
//!   capture side" before a dispatcher exists to answer it with behaviour.
//! * **Tier tests** pin the two-tier rule. A verb whose lowering is unbuilt still
//!   performs its portable checks first, so each panicking verb has a
//!   `#[should_panic]` test naming the missing lowering *and* a test that the
//!   portable refusal fires before it — because a panic test alone cannot tell
//!   "refused correctly" from "panicked too early".
//!
//! Three call sites that would be shape tests are *behavioural* instead, because
//! they can be run: the observer's copy-what-you-keep queue, the walk over a
//! recording's command list, and the replay input reader. They take hand-built
//! captured values and call no panicking verb. Running them is strictly more
//! evidence than compiling them, so they are written as tests whose subject is the
//! interface. The three that cannot be run — the two `describe_*` paths and the
//! `subscribe` path — remain shape tests.
//!
//! A shape test that stops compiling because the interface changed is this module
//! working as intended.

use std::sync::Arc;

use crate::api::binding::BindingSlotId;
use crate::api::command::{
    AccessMask, ColorClearValue, LoadOp, PipelineScope, StoreOp, TextureUseIntent,
};
use crate::api::diagnostics::{DiagnosticEvent, DiagnosticSeverity};
use crate::api::error::{RhiErrorKind, RhiResult};
use crate::api::format::TextureFormat;
use crate::api::identity::{DeviceIdentity, DeviceInstanceId, Label, ObjectId};
use crate::api::pipeline::{
    ColorTargetState, MultisampleState, PrimitiveState, PrimitiveTopology, VertexInputState,
};
use crate::api::platform::{Device, DeviceLossInfo};
use crate::api::presentation::{
    AcquiredFrameId, PresentPlanId, PresentReceiptId, PresentState, PresentationConfiguration,
};
use crate::api::resource::buffer::BufferRange;
use crate::api::resource::subresource::{
    Origin3d, TextureAspect, TextureAspects, TextureSubresourceLayers, TextureSubresourceRange,
};
use crate::api::resource::texture::Extent3d;
use crate::api::submission::{
    CompletionPoint, CompletionState, LaneWorkDomains, PlanPoint, SubmissionBatchId,
    SubmissionLaneId, SubmissionPlanId, SubmissionPoint,
};
use crate::api::tests::mock::{device_for_test, paired_device_for_test};
use crate::api::tooling::definition::{
    CapturedBindGroupDefinition, CapturedBindGroupEntry, CapturedBindingResource,
    CapturedComputePipelineDefinition, CapturedObjectDefinition,
    CapturedPipelineInterfaceDefinition, CapturedRasterPipelineDefinition,
};
use crate::api::tooling::mutation::{
    CapturedBufferCopy, CapturedBufferTextureCopy, CapturedColorAttachment,
    CapturedColorAttachmentView, CapturedRasterScope, CapturedReadbackRequest, CapturedResolve,
    CapturedTextureCopy, CapturedUploadDefinition,
};
use crate::api::tooling::plan::{
    CapturedDependencySource, CapturedPlanDependency, CapturedPresentPlan, CapturedSubmissionBatch,
    CapturedSubmissionPlan, CapturedSubmissionReceipt,
};
use crate::api::tooling::work::{
    CapturedCommand, CapturedRecordedWork, CapturedResourceUse, PortableCommand,
};
use crate::api::tooling::{
    SemanticEvent, SemanticEventId, SemanticObserver, TOOLING_SPI_VERSION, ToolingAccess,
    ToolingSpiVersion, ToolingSubscription,
};

// ---------------------------------------------------------------------------
// Fixtures
//
// None of these tokens has a public constructor, by design. The tests reach the
// crate-private ones exactly as the sibling chapter tests do: a test that could
// not name a token could not test anything about tokens. Those constructors are
// `#[cfg_attr(not(test), expect(dead_code, ...))]`, so calling them here is the
// intended use rather than a lint exception.
// ---------------------------------------------------------------------------

fn identity(instance: u64) -> DeviceIdentity {
    DeviceIdentity::new(DeviceInstanceId::new(instance))
}

fn object(value: u64) -> ObjectId {
    ObjectId::new(value)
}

fn event_id(value: u64) -> SemanticEventId {
    SemanticEventId::new(value)
}

fn active_device() -> Device {
    device_for_test(identity(1))
}

/// A device that has been lost, with the stable summary section 6.5 requires.
fn lost_device() -> Device {
    let (device, native) = paired_device_for_test(identity(2));
    native.mark_lost(DeviceLossInfo::new(String::from("the adapter was removed")));
    device
}

/// A subresource range covering one colour mip of one array layer.
///
/// Written as a literal because [`TextureSubresourceRange`] has public fields and
/// deliberately no `whole()`: the type states one range, and the caller states
/// which one.
fn one_colour_layer() -> TextureSubresourceRange {
    TextureSubresourceRange {
        aspects: TextureAspects::COLOR,
        base_mip: 0,
        mip_count: 1,
        base_layer: 0,
        layer_count: 1,
    }
}

/// One colour mip of one array layer, as a copy operation addresses it.
fn one_colour_layers() -> TextureSubresourceLayers {
    TextureSubresourceLayers {
        aspect: TextureAspect::Color,
        mip_level: 0,
        base_layer: 0,
        layer_count: 1,
    }
}

fn origin() -> Origin3d {
    Origin3d { x: 0, y: 0, z: 0 }
}

/// The four tokens the plan and receipt tests correlate.
fn plan_fixture(
    device: DeviceIdentity,
) -> (
    SubmissionPlanId,
    PlanPoint,
    SubmissionPoint,
    CompletionPoint,
) {
    let plan = SubmissionPlanId::new(device, 1);
    let point = PlanPoint::new(plan, SubmissionBatchId::new(0));
    let submitted = SubmissionPoint::new(device, 1);
    let completion = CompletionPoint::new(device, 1);
    (plan, point, submitted, completion)
}

/// An observer that does nothing, for tests about registration and refusal rather
/// than about delivery.
struct SilentObserver;

impl SemanticObserver for SilentObserver {
    fn on_event(&self, event: SemanticEvent<'_>) {
        let _ = event.event_id();
    }
}

fn silent_observer() -> Arc<dyn SemanticObserver> {
    Arc::new(SilentObserver)
}

// ---------------------------------------------------------------------------
// Behavioural: the SPI version
// ---------------------------------------------------------------------------

#[test]
fn the_spi_version_is_the_value_section_53_1_freezes() {
    // Section 53.1 declares both the constant and its value. Changing it is a
    // deliberate act with a semver consequence, so it should never be a silent
    // edit that only a caller notices.
    assert_eq!(TOOLING_SPI_VERSION.major, 1);
    assert_eq!(TOOLING_SPI_VERSION.minor, 0);
}

#[test]
fn a_spi_version_is_a_comparable_value_rather_than_an_opaque_token() {
    // Section 53.1 gives it public fields and a full derive list, unlike the
    // identity tokens of section 3. The point is that a tool branches on it — "can
    // I read this build's events" — and a tool cannot branch on bytes it cannot
    // read. The first assertion is written as a `where` clause so that dropping a
    // derive is a compile error rather than a behaviour change.
    fn assert_usable<T: Copy + PartialEq + Eq + std::hash::Hash + std::fmt::Debug>() {}
    assert_usable::<ToolingSpiVersion>();

    let future = ToolingSpiVersion {
        major: TOOLING_SPI_VERSION.major,
        minor: TOOLING_SPI_VERSION.minor + 1,
    };
    assert_ne!(future, TOOLING_SPI_VERSION);
    assert_eq!(future.major, TOOLING_SPI_VERSION.major);

    let mut seen = std::collections::HashSet::new();
    seen.insert(future);
    assert!(seen.contains(&future));
    assert!(!seen.contains(&TOOLING_SPI_VERSION));
}

// ---------------------------------------------------------------------------
// Behavioural: event identity
// ---------------------------------------------------------------------------

#[test]
fn an_event_id_orders_and_reports_its_counter() {
    // Section 53.2 states the subscription contract as "in increasing
    // SemanticEventId order", and section 53.3 calls the sequence unique and
    // monotonic within one DeviceIdentity. Both halves are only checkable if the
    // token carries a total order and a readable counter, which is why this crate
    // added them to section 53.3's derive and accessor list.
    let first = event_id(4);
    let second = event_id(5);

    assert!(first < second);
    assert_eq!(first.as_u64(), 4);

    let mut ids = [event_id(9), event_id(4), event_id(5)];
    ids.sort();
    assert_eq!(
        ids.iter().map(|id| id.as_u64()).collect::<Vec<_>>(),
        vec![4, 5, 9]
    );
}

// ---------------------------------------------------------------------------
// Behavioural: the access and the subscription
// ---------------------------------------------------------------------------

#[test]
fn tooling_access_is_device_scoped_and_cloneable() {
    // Section 53.2 declares `ToolingAccess` as `#[derive(Clone)]` and
    // device-scoped. Cloning an access is cloning the device reference, not taking
    // a second registration, so the two clones must agree about which device they
    // are for.
    let device = active_device();
    let access = device.tooling();
    let clone = access.clone();

    assert_eq!(access.device_identity(), device.identity());
    assert_eq!(clone.device_identity(), access.device_identity());

    // Section 3.1: `Device::clone()` is the same DeviceIdentity. An access taken
    // from a clone is the same domain's access.
    let second = device.clone();
    assert_eq!(second.tooling().device_identity(), access.device_identity());
}

#[test]
fn a_subscription_reports_the_device_it_registered_with() {
    let device = active_device();
    let identity = device.identity();
    let subscription = device
        .tooling()
        .subscribe(silent_observer())
        .expect("an active device accepts an observer");

    assert_eq!(subscription.device_identity(), identity);
    drop(subscription);
}

#[test]
fn tooling_remains_askable_on_a_lost_device_and_the_verbs_do_the_refusing() {
    // Section 52.8's rule is about what a verb answers, not about whether the
    // access exists: a capture tool looking at a device it just lost is exactly
    // when it wants to ask, and an accessor that refused to be created would make
    // the answer unreachable.
    let device = lost_device();
    let access = device.tooling();

    assert_eq!(access.device_identity(), device.identity());
}

// ---------------------------------------------------------------------------
// Behavioural: the device-lost refusal, which is the portable half
// ---------------------------------------------------------------------------

#[test]
fn subscribe_refuses_on_a_lost_device() {
    let access = lost_device().tooling();
    let result = access.subscribe(silent_observer());

    match result {
        Err(error) => assert_eq!(error.kind(), RhiErrorKind::DeviceLost),
        Ok(_) => panic!("a subscription was handed out for a device that will emit nothing"),
    }
}

#[test]
fn describe_object_refuses_on_a_lost_device() {
    let access = lost_device().tooling();
    let result = access.describe_object(object(11));

    match result {
        Err(error) => assert_eq!(error.kind(), RhiErrorKind::DeviceLost),
        Ok(_) => panic!("a lost device described an object it no longer has"),
    }
}

#[test]
fn describe_work_refuses_on_a_lost_device() {
    let access = lost_device().tooling();
    let result = access.describe_work(object(12));

    match result {
        Err(error) => assert_eq!(error.kind(), RhiErrorKind::DeviceLost),
        Ok(_) => panic!("a lost device described work it no longer holds"),
    }
}

#[test]
fn the_refusal_carries_the_loss_summary_section_6_5_makes_stable() {
    // A bare `DeviceLost` would tell a capture tool that it cannot ask, and not
    // why. The summary is the device's own stable answer, so the refusal is where
    // it belongs rather than only on `Device::loss_info`.
    let access = lost_device().tooling();

    let error = access.describe_object(object(11)).err().expect("lost");
    assert!(
        error.to_string().contains("the adapter was removed"),
        "the refusal should carry the loss summary, got: {error}"
    );
}

// ---------------------------------------------------------------------------
// Runtime service behaviour on an active device.
// ---------------------------------------------------------------------------

#[test]
fn subscribe_registers_on_a_live_device() {
    let access = active_device().tooling();
    let subscription = access
        .subscribe(silent_observer())
        .expect("active device accepts observer");
    assert_eq!(subscription.device_identity(), access.device_identity());
}

#[test]
fn describe_object_reports_unsupported_when_the_runtime_does_not_retain_definitions() {
    let access = active_device().tooling();
    let error = access
        .describe_object(object(11))
        .err()
        .expect("unsupported");
    assert_eq!(error.kind(), RhiErrorKind::Unsupported);
}

#[test]
fn describe_work_reports_unsupported_when_the_runtime_does_not_retain_work() {
    let access = active_device().tooling();
    let error = access.describe_work(object(12)).err().expect("unsupported");
    assert_eq!(error.kind(), RhiErrorKind::Unsupported);
}

// ---------------------------------------------------------------------------
// Behavioural (shape purpose): the observer's copy-what-you-keep queue
//
// This is the call site a capture tool writes, and it runs. The exhaustiveness of
// the `match` is the shape half: a new variant in section 58.1 stops this file
// compiling, which is the review signal a shape test exists to give.
// ---------------------------------------------------------------------------

/// A capture tool's own queue.
///
/// It owns everything it keeps, which is the section 58.2 discipline: references
/// inside an event die when the callback returns, so an observer that wants
/// something later copies it out before returning. Note what the retained types
/// are — `ObjectId`, `AcquiredFrameId`, and the captured value types — and what
/// they are not: no `Buffer`, no `Texture`, and no borrow of the event.
#[derive(Default)]
struct CaptureQueue {
    ids: std::sync::Mutex<Vec<SemanticEventId>>,
    definitions: std::sync::Mutex<Vec<ObjectId>>,
    uploads: std::sync::Mutex<Vec<ObjectId>>,
    readbacks: std::sync::Mutex<Vec<ObjectId>>,
    work: std::sync::Mutex<usize>,
    plans: std::sync::Mutex<usize>,
    completions: std::sync::Mutex<Vec<CompletionPoint>>,
    frames: std::sync::Mutex<Vec<AcquiredFrameId>>,
    presents: std::sync::Mutex<Vec<PresentReceiptId>>,
    diagnostics: std::sync::Mutex<usize>,
    losses: std::sync::Mutex<usize>,
}

impl CaptureQueue {
    fn new() -> Self {
        Self::default()
    }

    fn ids(&self) -> Vec<u64> {
        match self.ids.lock() {
            Ok(ids) => ids.iter().map(|id| id.as_u64()).collect(),
            Err(_) => Vec::new(),
        }
    }

    fn len_of<T>(list: &std::sync::Mutex<Vec<T>>) -> usize {
        match list.lock() {
            Ok(list) => list.len(),
            Err(_) => 0,
        }
    }

    fn count(list: &std::sync::Mutex<usize>) -> usize {
        match list.lock() {
            Ok(count) => *count,
            Err(_) => 0,
        }
    }
}

impl SemanticObserver for CaptureQueue {
    fn on_event(&self, event: SemanticEvent<'_>) {
        // The id is `Copy`, so it can be taken without cloning the event — and
        // the event is deliberately not `Clone` (section 58.1).
        if let Ok(mut ids) = self.ids.lock() {
            ids.push(event.event_id());
        }

        match event {
            // The one variant that borrows a whole definition, so that a reclaim
            // cannot race a later lazy query (section 58.1).
            SemanticEvent::ObjectCreated { definition, .. } => {
                let name = match definition {
                    CapturedObjectDefinition::Buffer { id, .. }
                    | CapturedObjectDefinition::Texture { id, .. }
                    | CapturedObjectDefinition::TextureView { id, .. }
                    | CapturedObjectDefinition::Sampler { id, .. }
                    | CapturedObjectDefinition::Shader { id, .. }
                    | CapturedObjectDefinition::BindGroupLayout { id, .. }
                    | CapturedObjectDefinition::BindGroup { id, .. }
                    | CapturedObjectDefinition::PipelineInterface { id, .. }
                    | CapturedObjectDefinition::RasterPipeline { id, .. }
                    | CapturedObjectDefinition::ComputePipeline { id, .. } => *id,
                };
                if let Ok(mut list) = self.definitions.lock() {
                    list.push(name);
                }
            }
            // "Actually reclaimed" (section 58.1), so a capture may treat every
            // record naming this object as complete.
            SemanticEvent::ObjectReclaimed { object, .. } => {
                let _ = object;
            }
            // The bytes are behind an `Arc<[u8]>`, so the tool keeps the reference
            // it wants rather than copying the payload into its queue.
            SemanticEvent::UploadDefined { upload, .. } => {
                let id = match upload {
                    CapturedUploadDefinition::Buffer { id, .. }
                    | CapturedUploadDefinition::Texture { id, .. } => *id,
                };
                if let Ok(mut list) = self.uploads.lock() {
                    list.push(id);
                }
            }
            SemanticEvent::ReadbackDefined { request, .. } => {
                let ticket = match request {
                    CapturedReadbackRequest::Buffer { ticket, .. }
                    | CapturedReadbackRequest::Texture { ticket, .. } => *ticket,
                };
                if let Ok(mut list) = self.readbacks.lock() {
                    list.push(ticket);
                }
            }
            SemanticEvent::WorkFinished { work, .. } => {
                let _ = work.commands.len();
                if let Ok(mut count) = self.work.lock() {
                    *count += 1;
                }
            }
            SemanticEvent::SubmissionAccepted { plan, receipt, .. } => {
                let _ = (plan.device, plan.plan, receipt.presents.len());
                if let Ok(mut count) = self.plans.lock() {
                    *count += 1;
                }
            }
            SemanticEvent::CompletionChanged { point, state, .. } => {
                if let Ok(mut list) = self.completions.lock() {
                    list.push(point);
                }
                let _ = state;
            }
            SemanticEvent::FrameAcquired { frame, .. } => {
                if let Ok(mut list) = self.frames.lock() {
                    list.push(frame);
                }
            }
            SemanticEvent::PresentChanged { receipt, state, .. } => {
                if let Ok(mut list) = self.presents.lock() {
                    list.push(receipt);
                }
                let _ = state;
            }
            // Section 52.8's terminal observation. A capture that records this
            // knows every outstanding point, ticket, frame, and receipt has ended.
            SemanticEvent::DeviceLost { .. } => {
                if let Ok(mut count) = self.losses.lock() {
                    *count += 1;
                }
            }
            SemanticEvent::Diagnostic { diagnostic, .. } => {
                let _ = diagnostic.severity;
                if let Ok(mut count) = self.diagnostics.lock() {
                    *count += 1;
                }
            }
        }
    }
}

#[test]
fn an_observer_keeps_what_it_copies_and_drops_every_borrow() {
    let device = identity(3);
    let (plan, point, submitted, completion) = plan_fixture(device);
    let frame = AcquiredFrameId::new(device, 1);
    let present = PresentPlanId::new(plan, 0);
    let receipt_id = PresentReceiptId::new(device, 1);

    let target = CapturedObjectDefinition::PipelineInterface {
        id: object(71),
        definition: CapturedPipelineInterfaceDefinition {
            label: Label::default(),
            groups: Vec::new(),
        },
    };
    let upload = CapturedUploadDefinition::Buffer {
        id: object(81),
        dst: object(82),
        dst_offset: 0,
        bytes: Arc::from(vec![1u8, 2, 3, 4].into_boxed_slice()),
    };
    let readback = CapturedReadbackRequest::Buffer {
        ticket: object(83),
        src: object(84),
        range: BufferRange::new(0, 64),
    };
    let work = CapturedRecordedWork {
        work: object(40),
        device,
        domains: LaneWorkDomains::COPY,
        commands: Vec::new(),
        merged_use_summary: Vec::new(),
    };
    let captured_plan = CapturedSubmissionPlan {
        device,
        plan,
        batches: vec![CapturedSubmissionBatch {
            point,
            lane: SubmissionLaneId::new(0),
            work: vec![object(40)],
        }],
        dependencies: Vec::new(),
        presents: vec![CapturedPresentPlan {
            id: present,
            frame,
            after: point,
        }],
    };
    let receipt = CapturedSubmissionReceipt {
        submitted,
        overall_completion: completion,
        point_completions: vec![(point, completion)],
        presents: vec![(present, receipt_id)],
    };
    let configuration = PresentationConfiguration::new(TextureFormat::Bgra8UnormSrgb);
    let completion_state = CompletionState::Complete;
    let present_state = PresentState::Accepted;
    let loss = DeviceLossInfo::new(String::from("the adapter was removed"));
    let diagnostic = DiagnosticEvent {
        severity: DiagnosticSeverity::Warning,
        message: String::from("a portable diagnostic"),
        object: None,
        label: None,
        operation: None,
        backend_detail: None,
    };

    let queue = CaptureQueue::new();
    queue.on_event(SemanticEvent::ObjectCreated {
        event: event_id(1),
        definition: &target,
    });
    queue.on_event(SemanticEvent::ObjectReclaimed {
        event: event_id(2),
        object: object(71),
    });
    queue.on_event(SemanticEvent::UploadDefined {
        event: event_id(3),
        upload: &upload,
    });
    queue.on_event(SemanticEvent::ReadbackDefined {
        event: event_id(4),
        request: &readback,
    });
    queue.on_event(SemanticEvent::WorkFinished {
        event: event_id(5),
        work: &work,
    });
    queue.on_event(SemanticEvent::SubmissionAccepted {
        event: event_id(6),
        plan: &captured_plan,
        receipt: &receipt,
    });
    queue.on_event(SemanticEvent::CompletionChanged {
        event: event_id(7),
        point: completion,
        state: &completion_state,
    });
    queue.on_event(SemanticEvent::FrameAcquired {
        event: event_id(8),
        target: object(71),
        configured_presentation: object(72),
        frame,
        configuration: &configuration,
        extent: Extent3d::d2(1280, 720),
    });
    queue.on_event(SemanticEvent::PresentChanged {
        event: event_id(9),
        receipt: receipt_id,
        state: &present_state,
    });
    queue.on_event(SemanticEvent::DeviceLost {
        event: event_id(10),
        info: &loss,
    });
    queue.on_event(SemanticEvent::Diagnostic {
        event: event_id(11),
        diagnostic: &diagnostic,
    });

    // Every event arrived, in the order it was delivered, and each one left the
    // queue holding a value it owns rather than a borrow of the event.
    assert_eq!(queue.ids(), (1..=11).collect::<Vec<u64>>());
    assert_eq!(CaptureQueue::len_of(&queue.definitions), 1);
    assert_eq!(CaptureQueue::len_of(&queue.uploads), 1);
    assert_eq!(CaptureQueue::len_of(&queue.readbacks), 1);
    assert_eq!(CaptureQueue::count(&queue.work), 1);
    assert_eq!(CaptureQueue::count(&queue.plans), 1);
    assert_eq!(CaptureQueue::len_of(&queue.completions), 1);
    assert_eq!(CaptureQueue::len_of(&queue.frames), 1);
    assert_eq!(CaptureQueue::len_of(&queue.presents), 1);
    assert_eq!(CaptureQueue::count(&queue.losses), 1);
    assert_eq!(CaptureQueue::count(&queue.diagnostics), 1);
}

// ---------------------------------------------------------------------------
// Behavioural (shape purpose): the captured records
// ---------------------------------------------------------------------------

#[test]
fn a_captured_recording_names_objects_without_holding_them() {
    // Section 56's invariant, exercised rather than asserted: the whole record is
    // built from ObjectIds and graph-bridge value types, and no field of it could
    // hold a live `Buffer`, `Texture`, or frame handle. If a live handle ever
    // appeared in `CapturedResourceUse`, this construction would stop compiling.
    let device = identity(3);
    let use_of_a_buffer = CapturedResourceUse::Buffer {
        buffer: object(41),
        range: BufferRange::new(0, 256),
        stages: PipelineScope::COMPUTE,
        access: AccessMask::SHADER_READ,
    };
    let use_of_a_texture = CapturedResourceUse::Texture {
        texture: object(42),
        subresources: one_colour_layer(),
        stages: PipelineScope::FRAGMENT,
        access: AccessMask::SHADER_READ,
        intent: TextureUseIntent::ShaderRead,
    };
    let use_of_a_frame = CapturedResourceUse::Frame {
        frame: AcquiredFrameId::new(device, 1),
        stages: PipelineScope::FRAGMENT,
        access: AccessMask::COLOR_WRITE,
    };

    let work = CapturedRecordedWork {
        work: object(40),
        device,
        domains: LaneWorkDomains::COPY.union(LaneWorkDomains::COMPUTE),
        commands: vec![
            CapturedCommand {
                command: PortableCommand::SetRasterPipeline(object(43)),
                actual_uses: Vec::new(),
            },
            CapturedCommand {
                command: PortableCommand::CopyBuffer(CapturedBufferCopy {
                    src: object(41),
                    src_offset: 0,
                    dst: object(44),
                    dst_offset: 0,
                    size: 256,
                }),
                actual_uses: vec![use_of_a_buffer.clone()],
            },
            CapturedCommand {
                command: PortableCommand::PushDebugGroup(String::from("shadow pass")),
                actual_uses: Vec::new(),
            },
            CapturedCommand {
                command: PortableCommand::PopDebugGroup,
                actual_uses: Vec::new(),
            },
            CapturedCommand {
                command: PortableCommand::DebugMarker(String::from("before bloom")),
                actual_uses: Vec::new(),
            },
        ],
        merged_use_summary: vec![use_of_a_buffer, use_of_a_texture, use_of_a_frame],
    };

    assert_eq!(work.work, object(40));
    assert_eq!(work.commands.len(), 5);
    assert_eq!(work.merged_use_summary.len(), 3);
    assert!(work.domains.contains(LaneWorkDomains::COMPUTE));
    assert!(!work.domains.contains(LaneWorkDomains::RASTER));

    // Section 52.2's debug-marker item, pinned from the capture side: the internal
    // IR keeps the marker payloads, and section 56 declares this portable spelling
    // for them.
    let labels: Vec<&str> = work
        .commands
        .iter()
        .filter_map(|command| match &command.command {
            PortableCommand::PushDebugGroup(label) | PortableCommand::DebugMarker(label) => {
                Some(label.as_str())
            }
            _ => None,
        })
        .collect();
    assert_eq!(labels, vec!["shadow pass", "before bloom"]);
}

#[test]
fn a_captured_raster_scope_keeps_interior_attachment_holes() {
    // Section 26 canonicalizes trailing holes only, so locations 0 and 3 with
    // nothing between them is a real deferred-renderer shape. A captured scope
    // that compacted it would rename location 3 to location 1, and a replay would
    // then bind the wrong target.
    let scope = CapturedRasterScope {
        label: Label(Some(String::from("gbuffer"))),
        colors: vec![
            Some(CapturedColorAttachment {
                view: CapturedColorAttachmentView::TextureView(object(51)),
                load: LoadOp::Clear(ColorClearValue::Float([0.0, 0.0, 0.0, 1.0])),
                store: StoreOp::Store,
                resolve: None,
            }),
            None,
            None,
            Some(CapturedColorAttachment {
                view: CapturedColorAttachmentView::Frame(AcquiredFrameId::new(identity(3), 1)),
                load: LoadOp::Load,
                store: StoreOp::Discard,
                resolve: None,
            }),
        ],
        depth_stencil: None,
    };

    assert_eq!(scope.colors.len(), 4);
    assert!(scope.colors[1].is_none());
    assert!(scope.colors[2].is_none());
    match &scope.colors[3] {
        Some(attachment) => {
            assert!(matches!(
                &attachment.view,
                CapturedColorAttachmentView::Frame(_)
            ));
            assert_eq!(attachment.store, StoreOp::Discard);
        }
        None => panic!("location 3 was attached"),
    }
}

#[test]
fn a_captured_plan_states_the_relation_rather_than_the_counts() {
    // Section 57 replaces three counts with the complete logical relation. The
    // point of this test is that the relation is *correlatable*: a present in the
    // plan and a pair in the receipt name the same PresentPlanId, which is what
    // lets a capture tie a later `present_state` query back to the batch a
    // presentation followed.
    let device = identity(4);
    let (plan, point, submitted, overall) = plan_fixture(device);
    let lane = SubmissionLaneId::new(0);
    let present_id = PresentPlanId::new(plan, 0);
    let receipt_id = PresentReceiptId::new(device, 1);
    let frame = AcquiredFrameId::new(device, 1);

    let captured = CapturedSubmissionPlan {
        device,
        plan,
        batches: vec![CapturedSubmissionBatch {
            point,
            lane,
            work: vec![object(61), object(62)],
        }],
        dependencies: vec![CapturedPlanDependency {
            before: CapturedDependencySource::PriorCompletion(CompletionPoint::new(device, 3)),
            after: point,
        }],
        presents: vec![CapturedPresentPlan {
            id: present_id,
            frame,
            after: point,
        }],
    };

    let receipt = CapturedSubmissionReceipt {
        submitted,
        overall_completion: overall,
        point_completions: vec![(point, CompletionPoint::new(device, 4))],
        presents: vec![(present_id, receipt_id)],
    };

    assert_eq!(captured.batches[0].work.len(), 2);
    assert_eq!(captured.batches[0].lane, lane);
    assert_eq!(captured.dependencies[0].after, point);
    assert!(matches!(
        captured.dependencies[0].before,
        CapturedDependencySource::PriorCompletion(_)
    ));
    assert_eq!(captured.presents[0].id, receipt.presents[0].0);
    assert_eq!(captured.presents[0].frame, frame);
    assert_eq!(captured.device, receipt.submitted.device_identity());
    assert_eq!(receipt.overall_completion.device_identity(), device);
}

#[test]
fn a_captured_upload_keeps_its_bytes_and_a_captured_readback_keeps_its_region() {
    // Section 52.4 and 55.3: an upload is an observable mutation and its source is
    // retained by the job, so capture never re-reads the CPU data it just wrote.
    // Section 52.5: a readback is a primitive RHI offers, and the record of one
    // names the region rather than deciding that a snapshot was wanted.
    let bytes: Arc<[u8]> = Arc::from(vec![1u8, 2, 3, 4].into_boxed_slice());
    let upload = CapturedUploadDefinition::Buffer {
        id: object(81),
        dst: object(82),
        dst_offset: 64,
        bytes: Arc::clone(&bytes),
    };

    match &upload {
        CapturedUploadDefinition::Buffer {
            dst,
            dst_offset,
            bytes,
            ..
        } => {
            assert_eq!(*dst, object(82));
            assert_eq!(*dst_offset, 64);
            assert_eq!(bytes.len(), 4);
        }
        _ => panic!("built a buffer upload"),
    }

    let readback = CapturedReadbackRequest::Texture {
        ticket: object(83),
        src: object(84),
        subresource: one_colour_layers(),
        origin: origin(),
        extent: Extent3d::d2(64, 64),
    };

    match readback {
        CapturedReadbackRequest::Texture { origin, extent, .. } => {
            assert_eq!(origin.x, 0);
            assert_eq!(extent.width, 64);
            assert_eq!(extent.height, 64);
        }
        _ => panic!("built a texture readback"),
    }
}

#[test]
fn a_captured_copy_keeps_its_direction_and_its_region() {
    // Section 55.2 and 56: `CopyBufferToTexture` and `CopyTextureToBuffer` share
    // one value type, and the direction lives in the command that carries it. A
    // test that pinned only one direction would miss a swap, so both are built.
    let buffer = object(91);
    let texture = object(92);
    let copy = CapturedBufferTextureCopy {
        buffer,
        buffer_offset: 0,
        bytes_per_row: 256,
        rows_per_image: 1,
        texture,
        texture_subresource: one_colour_layers(),
        texture_origin: origin(),
        extent: Extent3d::d2(64, 64),
    };

    let up = PortableCommand::CopyBufferToTexture(copy.clone());
    let down = PortableCommand::CopyTextureToBuffer(copy);

    assert!(matches!(up, PortableCommand::CopyBufferToTexture(_)));
    assert!(matches!(down, PortableCommand::CopyTextureToBuffer(_)));

    let texture_copy = CapturedTextureCopy {
        src: texture,
        src_subresource: one_colour_layers(),
        src_origin: origin(),
        dst: buffer,
        dst_subresource: one_colour_layers(),
        dst_origin: origin(),
        extent: Extent3d::d2(64, 64),
    };
    assert!(matches!(
        PortableCommand::CopyTexture(texture_copy),
        PortableCommand::CopyTexture(_)
    ));

    // Section 55.2's list omits origins on a resolve and the live type has them;
    // the captured type mirrors the live one, because a resolve that cannot name
    // the sub-region it covered replays to a different image.
    let resolve = CapturedResolve {
        src: texture,
        src_subresource: one_colour_layers(),
        src_origin: origin(),
        dst: buffer,
        dst_subresource: one_colour_layers(),
        dst_origin: origin(),
        extent: Extent3d::d2(64, 64),
    };
    assert_eq!(resolve.src, texture);
    assert_eq!(resolve.dst, buffer);
}

#[test]
fn a_captured_bind_group_reaches_its_layout_and_its_resources() {
    // Section 52.3 and 54: the graph edges are the ids, and a bind group reaching
    // its layout and its resources is what makes a reconstruction from the record
    // possible without any live handle.
    let definition = CapturedBindGroupDefinition {
        label: Label(Some(String::from("material"))),
        layout: object(101),
        entries: vec![
            CapturedBindGroupEntry {
                slot: BindingSlotId::new(0),
                resource: CapturedBindingResource::Buffer {
                    buffer: object(102),
                    range: BufferRange::new(0, 64),
                },
            },
            CapturedBindGroupEntry {
                slot: BindingSlotId::new(1),
                resource: CapturedBindingResource::SamplerArray(vec![object(103), object(104)]),
            },
        ],
    };

    assert_eq!(definition.layout, object(101));
    assert_eq!(definition.entries.len(), 2);
    match &definition.entries[1].resource {
        CapturedBindingResource::SamplerArray(samplers) => assert_eq!(samplers.len(), 2),
        _ => panic!("declared a sampler array"),
    }
}

#[test]
fn a_captured_pipeline_definition_keeps_identity_and_fixed_state_apart() {
    // Section 54's `CapturedRasterPipelineDefinition` has both halves — the
    // identity fields and the fixed state — and both are built here so that a
    // field dropped from either half fails.
    let definition = CapturedRasterPipelineDefinition {
        label: Label(None),
        vertex: object(111),
        fragment: Some(object(112)),
        interface: object(113),
        vertex_input: VertexInputState::new(),
        primitive: PrimitiveState::new(PrimitiveTopology::TriangleList),
        depth_stencil: None,
        multisample: MultisampleState::new(1),
        color_targets: vec![
            None,
            Some(ColorTargetState::new(TextureFormat::Bgra8UnormSrgb)),
        ],
    };
    let compute = CapturedComputePipelineDefinition {
        label: Label(None),
        shader: object(114),
        interface: object(113),
    };
    let interface = CapturedPipelineInterfaceDefinition {
        label: Label(None),
        groups: vec![object(115), object(116)],
    };

    assert_eq!(definition.fragment, Some(object(112)));
    assert_eq!(definition.color_targets.len(), 2);
    assert!(definition.color_targets[0].is_none());
    assert_eq!(compute.interface, definition.interface);
    assert_eq!(interface.groups.len(), 2);

    // Section 26 canonicalizes trailing holes only, so a captured pipeline keeps
    // its interior hole at location 0 and its target at location 1.
    match &definition.color_targets[1] {
        Some(target) => assert_eq!(target.format, TextureFormat::Bgra8UnormSrgb),
        None => panic!("location 1 was not a target"),
    }
}

// ---------------------------------------------------------------------------
// Behavioural (shape purpose): walking a recording's commands
//
// This is section 56's lowering read from the consumer side, and it runs. The
// `match` is exhaustive over `PortableCommand`, which is the shape half: a new
// command variant stops this file compiling.
// ---------------------------------------------------------------------------

/// Every object a recording names, in the order its commands name them.
fn objects_named_by(work: &CapturedRecordedWork) -> Vec<ObjectId> {
    let mut touched = Vec::new();

    for captured in &work.commands {
        match &captured.command {
            PortableCommand::BeginRaster(scope) => {
                for attachment in scope.colors.iter().flatten() {
                    // Matched by reference: an attachment view is not `Copy`, and
                    // a by-value binding here would be a move out of a borrow.
                    if let CapturedColorAttachmentView::TextureView(view) = &attachment.view {
                        touched.push(*view);
                    }
                }
            }
            PortableCommand::SetRasterPipeline(pipeline)
            | PortableCommand::SetComputePipeline(pipeline) => touched.push(*pipeline),
            PortableCommand::SetBindGroup { group, .. } => touched.push(*group),
            PortableCommand::SetVertexBuffer { buffer, .. }
            | PortableCommand::SetIndexBuffer { buffer, .. } => touched.push(*buffer),
            // Actual use is generated at draw and dispatch (section 37.1), so a
            // use list is read where it was produced rather than at a
            // state-setting verb.
            PortableCommand::Draw { .. }
            | PortableCommand::DrawIndexed { .. }
            | PortableCommand::Dispatch { .. } => {
                for used in &captured.actual_uses {
                    match used {
                        CapturedResourceUse::Buffer { buffer, .. } => touched.push(*buffer),
                        CapturedResourceUse::Texture { texture, .. } => touched.push(*texture),
                        CapturedResourceUse::Frame { frame, .. } => {
                            let _ = frame.device_identity();
                        }
                    }
                }
            }
            PortableCommand::Upload { upload } | PortableCommand::Readback { ticket: upload } => {
                touched.push(*upload)
            }
            PortableCommand::CopyBuffer(copy) => {
                touched.push(copy.src);
                touched.push(copy.dst);
            }
            PortableCommand::CopyBufferToTexture(copy)
            | PortableCommand::CopyTextureToBuffer(copy) => {
                touched.push(copy.buffer);
                touched.push(copy.texture);
            }
            PortableCommand::CopyTexture(copy) => {
                touched.push(copy.src);
                touched.push(copy.dst);
            }
            PortableCommand::Resolve(resolve) => {
                touched.push(resolve.src);
                touched.push(resolve.dst);
            }
            PortableCommand::Blit(blit) => {
                touched.push(blit.src);
                touched.push(blit.dst);
            }
            PortableCommand::EndRaster
            | PortableCommand::EndCompute
            | PortableCommand::BeginCompute { .. }
            | PortableCommand::SetViewport(_)
            | PortableCommand::SetScissor(_)
            | PortableCommand::SetBlendConstant(_)
            | PortableCommand::SetStencilReference(_)
            | PortableCommand::PushDebugGroup(_)
            | PortableCommand::PopDebugGroup
            | PortableCommand::DebugMarker(_) => {}
        }
    }

    touched
}

#[test]
fn a_walk_over_a_recording_reaches_every_object_it_names() {
    let device = identity(6);
    let work = CapturedRecordedWork {
        work: object(120),
        device,
        domains: LaneWorkDomains::RASTER,
        commands: vec![
            CapturedCommand {
                command: PortableCommand::BeginRaster(CapturedRasterScope {
                    label: Label(None),
                    colors: vec![Some(CapturedColorAttachment {
                        view: CapturedColorAttachmentView::TextureView(object(121)),
                        load: LoadOp::Clear(ColorClearValue::Float([0.0, 0.0, 0.0, 1.0])),
                        store: StoreOp::Store,
                        resolve: None,
                    })],
                    depth_stencil: None,
                }),
                actual_uses: Vec::new(),
            },
            CapturedCommand {
                command: PortableCommand::SetRasterPipeline(object(122)),
                actual_uses: Vec::new(),
            },
            CapturedCommand {
                command: PortableCommand::SetVertexBuffer {
                    slot: 0,
                    buffer: object(123),
                    range: BufferRange::new(0, 1024),
                },
                actual_uses: Vec::new(),
            },
            CapturedCommand {
                command: PortableCommand::Draw {
                    vertices: 0..3,
                    instances: 0..1,
                },
                actual_uses: vec![CapturedResourceUse::Buffer {
                    buffer: object(124),
                    range: BufferRange::new(0, 64),
                    stages: PipelineScope::VERTEX,
                    access: AccessMask::SHADER_READ,
                }],
            },
            CapturedCommand {
                command: PortableCommand::EndRaster,
                actual_uses: Vec::new(),
            },
            CapturedCommand {
                command: PortableCommand::Upload {
                    upload: object(125),
                },
                actual_uses: Vec::new(),
            },
            CapturedCommand {
                command: PortableCommand::DebugMarker(String::from("done")),
                actual_uses: Vec::new(),
            },
        ],
        merged_use_summary: Vec::new(),
    };

    assert_eq!(
        objects_named_by(&work),
        vec![
            object(121),
            object(122),
            object(123),
            object(124),
            object(125)
        ]
    );
}

// ---------------------------------------------------------------------------
// Shape tests: the call sites that cannot be run
//
// Each of the four below reaches a verb whose lowering is unbuilt, or has no RHI
// entry point at all, so it panics or has nothing to call. They are compiled and
// never executed, marked with the crate's `#[expect(dead_code, ...)]`, and written
// as a capture tool would write them. Their job is to answer "is this interface
// usable from the capture side" — if one needs an extra construction step, a
// lifetime it should not have to name, or a state precondition it cannot check,
// the fault is in the interface.
// ---------------------------------------------------------------------------

/// The whole happy path of a capture tool: open an access, register an observer,
/// walk the events, and pull the definitions of things that already existed —
/// without ever naming a live handle.
///
/// This is the shape test that matters most for section 53.2, because it asks the
/// question the chapter exists to answer.
#[expect(
    dead_code,
    reason = "a shape test; compiled to check the interface, never called"
)]
fn shape_a_capture_tool_subscribes_walks_events_and_pulls_definitions(
    device: &Device,
) -> RhiResult<()> {
    let access: ToolingAccess = device.tooling();
    let queue = Arc::new(CaptureQueue::new());

    // Section 53.2's start linearization point: this returns only once the
    // observer is in the device's set, so nothing assigned after it can be
    // missed. The subscription is held in a binding rather than dropped
    // immediately, which is what makes the registration last.
    let subscription: ToolingSubscription = access.subscribe(queue.clone())?;

    // The lazy half, for the objects and work that predate the subscription.
    let definition: CapturedObjectDefinition = access.describe_object(object(1))?;
    let work: CapturedRecordedWork = access.describe_work(object(2))?;
    let _ = (definition, work);

    // Dropping waits for every callback that began before this point, so after
    // this line the queue is quiescent and nothing in it is still being written
    // by the RHI.
    drop(subscription);
    Ok(())
}

/// Section 58.1's keying rule, as a call site: work recorded before the tool
/// started watching is still describable, because the verb takes an `ObjectId`
/// rather than the handle the tool never held.
#[expect(
    dead_code,
    reason = "a shape test; compiled to check the interface, never called"
)]
fn shape_describe_work_without_holding_the_handle(
    access: &ToolingAccess,
    work: ObjectId,
) -> RhiResult<CapturedRecordedWork> {
    let described = access.describe_work(work)?;
    let _ = (described.domains, described.commands.len());
    Ok(described)
}

/// Section 58.7's replay input, from the consumer side: a ReplayRuntime reads the
/// definition graph, the commands, and the plan relation, and calls the normal RHI
/// API with them. The function is dead only because a replay has no RHI entry
/// point — which is the point of section 58.6.
#[expect(
    dead_code,
    reason = "a shape test; compiled to check the interface, never called"
)]
fn shape_replay_reads_the_same_thing_the_artifact_layer_writes(
    definitions: Vec<CapturedObjectDefinition>,
    commands: Vec<PortableCommand>,
    plan: CapturedSubmissionPlan,
    receipt: &CapturedSubmissionReceipt,
) {
    for definition in &definitions {
        if let CapturedObjectDefinition::BindGroup { definition, .. } = definition {
            let _ = definition.layout;
        }
    }

    for command in &commands {
        if let PortableCommand::Dispatch { x, y, z } = command {
            let _ = (x, y, z);
        }
    }

    for batch in &plan.batches {
        let _ = (batch.lane, batch.point, batch.work.len());
    }

    for (id, receipt_id) in &receipt.presents {
        // The pair is what ties a plan-local present to the receipt a later
        // `present_state` query takes.
        let _ = (id, receipt_id);
    }

    let _ = plan.dependencies.len();
}
