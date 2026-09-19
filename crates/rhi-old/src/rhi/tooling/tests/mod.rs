//! Contract tests for the tooling SPI (rhi-design sections 52 through 58).
//!
//! The three submodules test three separate contract areas:
//!
//! ```text
//! linearization  the subscription points, ordered exactly-once delivery,
//!                Drop waiting for an in-flight callback, and the two
//!                prohibited callback cases being refused
//! description    lazy description of an object that predates a subscription
//! shapes         an event and a definition carrying logical identities only
//! ```
//!
//! Everything they share is here: a device built over the crate-private
//! `ToolingBackend` seam, and an observing recorder. That fake device is the
//! whole backend — the registry, the event-id mint, and the linearization
//! enforcement are the module's own code under test, not part of the fake.

mod description;
mod linearization;
mod shapes;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::registry::ToolingRegistry;
use super::super::platform::{
    DeviceIdentity, ObjectId, RhiError, RhiErrorKind, next_identity, next_object_id,
};
use super::super::presentation::AcquiredFrameId;
use super::super::resource::{BufferDescriptor, BufferUsage};
use super::{
    CapturedObjectDefinition, CapturedPresentationTargetDefinition,
    CapturedPresentationTargetFixture, CapturedRecordedWork, SemanticEvent, SemanticEventId,
    SemanticObserver, ToolingAccess, ToolingBackend, ToolingSubscription,
};

/// The device identity the fake's access is scoped to.
///
/// Every definition that names a device needs one, and only a backend's own
/// `request_device` path may mint one in production, so a tooling test takes the
/// same mint a backend would.
fn test_identity() -> DeviceIdentity {
    next_identity()
}

/// One observed event, reduced to what the assertions can compare.
///
/// The event itself must not outlive `on_event`, so a test observer copies the
/// parts it keeps — which is exactly what the callback contract requires.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Observed {
    /// An object-created event and the id it described.
    Created { event: u64, object: ObjectId },
    /// A frame-acquired event and the identities it carried.
    Frame {
        event: u64,
        target: ObjectId,
        lease: ObjectId,
        frame: AcquiredFrameId,
    },
    /// An object-reclaimed event.
    Reclaimed { event: u64, object: ObjectId },
}

/// The id a definition carries, which is the only kind of identity it may carry.
fn definition_id(definition: &CapturedObjectDefinition) -> ObjectId {
    match definition {
        CapturedObjectDefinition::PresentationTarget { id, .. }
        | CapturedObjectDefinition::ConfiguredPresentation { id, .. }
        | CapturedObjectDefinition::Buffer { id, .. }
        | CapturedObjectDefinition::Texture { id, .. }
        | CapturedObjectDefinition::TextureView { id, .. }
        | CapturedObjectDefinition::Sampler { id, .. }
        | CapturedObjectDefinition::Shader { id, .. }
        | CapturedObjectDefinition::BindGroupLayout { id, .. }
        | CapturedObjectDefinition::BindGroup { id, .. }
        | CapturedObjectDefinition::PipelineInterface { id, .. }
        | CapturedObjectDefinition::RasterPipeline { id, .. }
        | CapturedObjectDefinition::ComputePipeline { id, .. } => *id,
    }
}

/// A presentation-target definition.
fn presentation_target(key: &str) -> CapturedObjectDefinition {
    CapturedObjectDefinition::PresentationTarget {
        id: next_object_id(),
        definition: CapturedPresentationTargetDefinition {
            fixture: CapturedPresentationTargetFixture {
                key: key.to_string(),
            },
        },
    }
}

/// A small copyable buffer definition.
fn small_buffer() -> CapturedObjectDefinition {
    CapturedObjectDefinition::Buffer {
        id: next_object_id(),
        descriptor: BufferDescriptor::new(256, BufferUsage::COPY_SRC.union(BufferUsage::COPY_DST)),
    }
}

/// A device that owns a registry, some definitions, and some work.
struct FakeDevice {
    registry: Arc<ToolingRegistry>,
    objects: Mutex<HashMap<ObjectId, CapturedObjectDefinition>>,
    work: Mutex<HashMap<ObjectId, CapturedRecordedWork>>,
}

impl FakeDevice {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            registry: Arc::new(ToolingRegistry::new()),
            objects: Mutex::new(HashMap::new()),
            work: Mutex::new(HashMap::new()),
        })
    }

    fn with_object(self: &Arc<Self>, id: ObjectId, definition: CapturedObjectDefinition) {
        self.objects
            .lock()
            .expect("object table")
            .insert(id, definition);
    }

    fn with_work(self: &Arc<Self>, work: CapturedRecordedWork) {
        self.work
            .lock()
            .expect("work table")
            .insert(work.work, work);
    }

    /// The access a consumer of this device would hold.
    fn access(self: &Arc<Self>) -> ToolingAccess {
        ToolingAccess::new(
            test_identity(),
            Some(Arc::clone(self) as Arc<dyn ToolingBackend>),
        )
    }

    /// Emits one object-created event for `id`, as a backend would.
    ///
    /// Assignment and delivery are one call, which is what the seam requires:
    /// the definition is borrowed for the call only, and an observer that keeps
    /// it must clone it, which is the documented lifetime.
    fn emit_created(&self, id: ObjectId) -> SemanticEventId {
        let definition = self
            .objects
            .lock()
            .expect("object table")
            .get(&id)
            .expect("the test emits only objects it registered")
            .clone();
        self.registry
            .emit_with(|event| SemanticEvent::ObjectCreated {
                event,
                definition: &definition,
            })
            .expect("the emitter is not inside a callback")
    }
}

impl ToolingBackend for FakeDevice {
    fn observers(&self) -> Option<Arc<ToolingRegistry>> {
        Some(Arc::clone(&self.registry))
    }

    fn describe_object(&self, id: ObjectId) -> Result<CapturedObjectDefinition, RhiError> {
        match self.objects.lock().expect("object table").get(&id) {
            Some(definition) => Ok(definition.clone()),
            None => Err(
                RhiError::new(RhiErrorKind::InvalidUsage, "no live object has that id").on(id),
            ),
        }
    }

    fn describe_work(&self, work: ObjectId) -> Result<CapturedRecordedWork, RhiError> {
        match self.work.lock().expect("work table").get(&work) {
            Some(definition) => Ok(definition.clone()),
            None => Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "no live recorded work has that id",
            )
            .on(work)),
        }
    }
}

/// An observer that records what it saw.
#[derive(Default)]
struct Recorder {
    seen: Mutex<Vec<Observed>>,
}

impl Recorder {
    fn events(&self) -> Vec<Observed> {
        self.seen.lock().expect("recorder").clone()
    }

    /// The ids of the events this observer saw, in the order it saw them.
    fn event_ids(&self) -> Vec<u64> {
        self.events()
            .into_iter()
            .map(|observed| match observed {
                Observed::Created { event, .. }
                | Observed::Frame { event, .. }
                | Observed::Reclaimed { event, .. } => event,
            })
            .collect()
    }
}

impl SemanticObserver for Recorder {
    fn on_event(&self, event: SemanticEvent<'_>) {
        let observed = match event {
            SemanticEvent::ObjectCreated { event, definition } => Observed::Created {
                event: event.as_u64(),
                object: definition_id(definition),
            },
            SemanticEvent::ObjectReclaimed { event, object } => Observed::Reclaimed {
                event: event.as_u64(),
                object,
            },
            SemanticEvent::FrameAcquired {
                event,
                target,
                configured_presentation,
                frame,
                ..
            } => Observed::Frame {
                event: event.as_u64(),
                target,
                lease: configured_presentation,
                frame,
            },
            _ => return,
        };
        self.seen.lock().expect("recorder").push(observed);
    }
}

/// Subscribes `observer`, for the common case.
fn subscribe(access: &ToolingAccess, observer: Arc<dyn SemanticObserver>) -> ToolingSubscription {
    access
        .subscribe(observer)
        .expect("the fake device implements the observer registry")
}
