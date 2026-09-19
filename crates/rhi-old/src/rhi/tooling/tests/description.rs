//! Lazy description (rhi-design sections 53.2 and 58.2).
//!
//! `describe_object` and `describe_work` are the seam that makes a capture scope
//! possible: an object or a recording that existed before the scope began never
//! produced an event a consumer saw, so it must be describable on demand for as
//! long as it is live. These tests also pin what a device without the seam
//! answers, and that the SPI version is a value the consumer reads.

use std::sync::Arc;

use super::{FakeDevice, Recorder, definition_id, small_buffer, subscribe, test_identity};
use crate::rhi::format::LaneWorkDomains;
use crate::rhi::graph_bridge::{AccessMask, PipelineScope};
use crate::rhi::platform::{RhiErrorKind, next_object_id};
use crate::rhi::resource::BufferRange;
use crate::rhi::tooling::{
    CapturedBufferCopy, CapturedCommand, CapturedObjectDefinition, CapturedRecordedWork,
    CapturedResourceUse, PortableCommand, SemanticObserver, TOOLING_SPI_VERSION, ToolingAccess,
    ToolingSpiVersion, ToolingViolationCounts,
};

#[test]
fn describe_object_describes_an_object_that_predates_the_subscription() {
    let device = FakeDevice::new();
    let object = small_buffer();
    let id = definition_id(&object);
    let descriptor = match &object {
        CapturedObjectDefinition::Buffer { descriptor, .. } => descriptor.clone(),
        _ => unreachable!("the fixture is a buffer"),
    };
    device.with_object(id, object);

    let access = device.access();
    let recorder = Arc::new(Recorder::default());
    let _subscription = subscribe(&access, Arc::clone(&recorder) as Arc<dyn SemanticObserver>);

    // The observer never saw an ObjectCreated event for this object.
    assert!(recorder.events().is_empty());

    let definition = access
        .describe_object(id)
        .expect("an object created before the capture scope is still describable");
    assert_eq!(definition_id(&definition), id);
    match definition {
        CapturedObjectDefinition::Buffer {
            descriptor: captured,
            ..
        } => assert_eq!(captured, descriptor),
        _ => panic!("the descriptor must survive the round trip"),
    }
}

#[test]
fn describing_an_unknown_object_is_a_structured_error() {
    let device = FakeDevice::new();
    let access = device.access();
    let unknown = next_object_id();

    // `describe_object` succeeds with a definition that is deliberately not
    // `Debug`, so the result is collapsed before the error is taken.
    let error = access
        .describe_object(unknown)
        .map(|_| ())
        .expect_err("nothing describes an object this device never created");
    assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
    assert_eq!(error.object(), Some(unknown));
    assert_eq!(error.operation(), Some("ToolingAccess::describe_object"));
}

#[test]
fn describe_work_returns_complete_portable_semantics() {
    let device = FakeDevice::new();
    let buffer = next_object_id();
    let work = CapturedRecordedWork {
        work: next_object_id(),
        device: test_identity(),
        domains: LaneWorkDomains::COPY,
        commands: vec![CapturedCommand {
            command: PortableCommand::CopyBuffer(CapturedBufferCopy {
                src: buffer,
                src_offset: 0,
                dst: buffer,
                dst_offset: 64,
                size: 64,
            }),
            actual_uses: vec![CapturedResourceUse::Buffer {
                buffer,
                range: BufferRange::new(0, 128),
                stages: PipelineScope::COPY,
                access: AccessMask::COPY_READ.union(AccessMask::COPY_WRITE),
            }],
        }],
        merged_use_summary: Vec::new(),
    };
    let work_id = work.work;
    device.with_work(work);

    let access = device.access();
    let described = access.describe_work(work_id).expect("describe_work");

    assert_eq!(described.work, work_id);
    assert_eq!(described.commands.len(), 1);
    match &described.commands[0].command {
        PortableCommand::CopyBuffer(copy) => {
            assert_eq!(copy.src, buffer);
            assert_eq!(copy.size, 64);
        }
        _ => panic!("the portable command must survive the round trip"),
    }
    assert_eq!(described.commands[0].actual_uses.len(), 1);
}

#[test]
fn a_device_without_a_tooling_backend_answers_unsupported() {
    let access = ToolingAccess::new(test_identity(), None);

    assert_eq!(access.spi_version(), TOOLING_SPI_VERSION);
    assert_eq!(
        access
            .subscribe(Arc::new(Recorder::default()) as Arc<dyn SemanticObserver>)
            .expect_err("no backend accepts observers")
            .kind(),
        RhiErrorKind::Unsupported
    );
    // A definition is not `Debug`, so each result is collapsed before the
    // error is taken.
    assert_eq!(
        access
            .describe_object(next_object_id())
            .map(|_| ())
            .expect_err("no backend describes objects")
            .kind(),
        RhiErrorKind::Unsupported
    );
    assert_eq!(
        access
            .describe_work(next_object_id())
            .map(|_| ())
            .expect_err("no backend describes work")
            .kind(),
        RhiErrorKind::Unsupported
    );
    assert_eq!(access.violations(), ToolingViolationCounts::default());
}

#[test]
fn the_spi_version_is_a_value_the_consumer_reads() {
    // The SPI version is not the artifact schema version: it is a value the
    // consumer compares before it interprets anything.
    let version: ToolingSpiVersion = TOOLING_SPI_VERSION;
    assert_eq!(version.major, 1);
    assert_eq!(version.minor, 0);
}
