//! Contract tests for [`crate::rhi::submission`] and its plan builder.
//!
//! These are rhi-design section 40.4 (unordered write hazards) and section 45.3
//! (frame closure), run over [`Mock`]. The plan builder is the part of the API
//! that decides, before any native queue exists, whether an ordering a caller
//! described is safe to lower. The mock supplies real recorded work — produced
//! by the same recorder the command tests use — so the resource uses the plan
//! reasons about are the uses the library actually generates, not uses a test
//! wrote by hand.
//!
//! # Why the mock's device has two lanes
//!
//! Section 40.4 is only reachable when two batches can be *unordered*. With a
//! single lane, insertion order already orders every pair, and the hazard check
//! could never fire. The mock's capability database therefore declares two
//! general lanes; that they are distinct is the whole point of the fixture.
//!
//! # What is reached honestly and what is not
//!
//! The "a frame may be presented at most once" rule has two paths into it. The
//! honest one — calling `present_after` twice with the same frame — cannot
//! happen, because `present_after` consumes the frame and the first call moves
//! the frame's state past `Acquired`. The duplicate scan therefore protects
//! against a backend handing out two tokens for one drawable, and the test
//! below reaches it through the mock's `duplicate_outstanding_frame`, which is
//! that fault and nothing else.

use std::sync::Arc;

use crate::rhi::command::{
    BufferCopy, ColorAttachment, ColorAttachmentView, CommandRecorder, RasterScopeDescriptor,
};
use crate::rhi::format::{SubmissionLaneId, TextureFormat};
use crate::rhi::mock::Mock;
use crate::rhi::pipeline::{
    ColorTargetState, PipelineInterfaceDescriptor, RasterPipeline, RasterPipelineDescriptor,
};
use crate::rhi::platform::RhiErrorKind;
use crate::rhi::presentation::PresentationConfiguration;
use crate::rhi::resource::{Buffer, BufferUsage};
use crate::rhi::shader::{
    ArtifactHash, ArtifactProducerId, ArtifactProducerVersion, ShaderAbiVersion, ShaderArtifact,
    ShaderCode, ShaderInterface, ShaderLocation, ShaderLocationInterface, ShaderModule,
    ShaderNumericType, ShaderRequirements, ShaderStage,
};
use crate::rhi::submission::{SubmissionPlanBuilder, SubmissionPlanId};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// The two lanes the mock's device declares, in capability order.
fn lanes(mock: &Mock) -> Vec<SubmissionLaneId> {
    mock.capabilities()
        .submission()
        .lanes()
        .iter()
        .map(|lane| lane.id())
        .collect()
}

/// A finished recording holding one buffer-to-buffer copy.
///
/// The copy is what gives the work a resource use the plan can see: the copy
/// reads `src` and writes `dst`, which is exactly the pair the hazard check
/// reasons about.
fn copy_work(mock: &Mock, src: &Buffer, dst: &Buffer, size: u64) -> crate::rhi::command::RecordedWork {
    let mut recorder: CommandRecorder = mock.recorder();
    recorder
        .copy_buffer(&BufferCopy {
            src: src.clone(),
            src_offset: 0,
            dst: dst.clone(),
            dst_offset: 0,
            size,
        })
        .expect("a whole-buffer copy is legal");
    recorder.finish().expect("the recording finalizes")
}

/// A shader module for `interface`, accepted by the mock's capability database.
fn shader(mock: &Mock, stage: ShaderStage, interface: ShaderInterface) -> ShaderModule {
    let artifact = ShaderArtifact::new(
        stage,
        "main",
        ShaderCode::SpirV(Arc::from(vec![0u32; 4])),
        ShaderAbiVersion::CURRENT,
        interface,
        ShaderRequirements::new(),
        ArtifactHash([0u8; 32]),
        ArtifactProducerId("fluxel-rhi-mock-contract-tests".to_string()),
        ArtifactProducerVersion { major: 1, minor: 0 },
    );
    mock.device()
        .create_shader(&artifact)
        .expect("the mock device accepts a current-ABI SpirV artifact")
}

/// A raster pipeline that writes `format` at color location 0 and reads nothing.
fn plain_pipeline(mock: &Mock, format: TextureFormat) -> RasterPipeline {
    let vertex = shader(
        mock,
        ShaderStage::Vertex,
        ShaderInterface::new().with_writes_position(true),
    );
    let fragment = shader(
        mock,
        ShaderStage::Fragment,
        ShaderInterface::new().with_output(ShaderLocationInterface {
            location: ShaderLocation::new(0),
            numeric_type: ShaderNumericType::Float32,
            components: 4,
            interpolation: None,
        }),
    );
    let interface = mock
        .device()
        .create_pipeline_interface(&PipelineInterfaceDescriptor::new(Vec::new()))
        .expect("the mock device accepts an interface over no groups");
    let descriptor = RasterPipelineDescriptor::new(vertex, interface)
        .with_fragment(fragment)
        .with_color_target(ShaderLocation::new(0), ColorTargetState::new(format));
    mock.device()
        .create_raster_pipeline(&descriptor)
        .expect("the mock device accepts a pipeline for this attachment format")
}

/// A finished recording holding one raster scope that draws into a presentation
/// frame.
///
/// The draw is what matters, not the scope: a frame use is generated by the
/// command that uses the attachment, so a scope that draws nothing would leave
/// the plan with nothing to close.
fn frame_work(mock: &Mock, frame: &crate::rhi::presentation::AcquiredFrame) -> crate::rhi::command::RecordedWork {
    let attachment = frame.attachment();
    let format = attachment.format();
    let descriptor = RasterScopeDescriptor::new().with_color(
        ShaderLocation::new(0),
        ColorAttachment::new(ColorAttachmentView::Frame(attachment)),
    );
    let pipeline = plain_pipeline(mock, format);

    let mut recorder = mock.recorder();
    {
        let mut scope = recorder
            .begin_raster(&descriptor)
            .expect("a frame attachment opens a raster scope");
        scope
            .set_pipeline(&pipeline)
            .expect("the pipeline was built for the frame's format");
        scope
            .draw(0..3, 0..1)
            .expect("a draw into a presentation frame is recorded");
        scope.end().expect("the scope closes");
    }
    let work = recorder.finish().expect("the recording finalizes");
    assert!(
        work.resource_uses().iter().any(|use_| {
            matches!(use_, crate::rhi::graph_bridge::ResourceUse::Frame(use_) if use_.frame == frame.id())
        }),
        "the draw into the frame declared a frame use for {:?}: {:?}",
        frame.id(),
        work.resource_uses()
    );
    work
}

// ---------------------------------------------------------------------------
// Section 40.4: unordered write hazards
// ---------------------------------------------------------------------------

#[test]
fn two_unordered_batches_that_race_are_refused() {
    let mock = Mock::new();
    let lane_ids = lanes(&mock);
    assert_eq!(
        lane_ids.len(),
        2,
        "the design's unordered-hazard rule needs two distinct lanes to be reachable"
    );

    let (src, dst, out) = (
        mock.buffer(64, BufferUsage::COPY_SRC),
        mock.buffer(64, BufferUsage::COPY_SRC.union(BufferUsage::COPY_DST)),
        mock.buffer(64, BufferUsage::COPY_DST),
    );
    let writer = copy_work(&mock, &src, &dst, 64);
    let reader = copy_work(&mock, &dst, &out, 64);

    let mut builder = SubmissionPlanBuilder::new(mock.device());
    builder
        .add_batch(lane_ids[0], vec![writer])
        .expect("the first batch is accepted");
    builder
        .add_batch(lane_ids[1], vec![reader])
        .expect("the second batch is accepted");

    let error = builder
        .build()
        .expect_err("two unordered batches racing on the same bytes are refused");
    assert_eq!(error.kind(), RhiErrorKind::MissingDependency);
}

#[test]
fn the_same_plan_builds_once_the_dependency_is_declared() {
    let mock = Mock::new();
    let lane_ids = lanes(&mock);
    let (src, dst, out) = (
        mock.buffer(64, BufferUsage::COPY_SRC),
        mock.buffer(64, BufferUsage::COPY_SRC.union(BufferUsage::COPY_DST)),
        mock.buffer(64, BufferUsage::COPY_DST),
    );

    let mut builder = SubmissionPlanBuilder::new(mock.device());
    let first = builder
        .add_batch(lane_ids[0], vec![copy_work(&mock, &src, &dst, 64)])
        .expect("the first batch is accepted");
    let second = builder
        .add_batch(lane_ids[1], vec![copy_work(&mock, &dst, &out, 64)])
        .expect("the second batch is accepted");

    builder
        .add_dependency(first, second)
        .expect("the two lanes have a declared ordered route");

    let plan = builder
        .build()
        .expect("the declared edge removes the hazard");
    assert_eq!(plan.batches().len(), 2);
    assert_eq!(plan.dependencies(), &[(first, second)]);
    assert_eq!(plan.batches()[0].lane(), lane_ids[0]);
    assert_eq!(plan.batches()[1].lane(), lane_ids[1]);
    assert!(plan.presents().is_empty());
}

#[test]
fn two_reads_of_the_same_bytes_need_no_order() {
    let mock = Mock::new();
    let lane_ids = lanes(&mock);
    let src = mock.buffer(64, BufferUsage::COPY_SRC);
    let left = mock.buffer(64, BufferUsage::COPY_DST);
    let right = mock.buffer(64, BufferUsage::COPY_DST);

    let mut builder = SubmissionPlanBuilder::new(mock.device());
    builder
        .add_batch(lane_ids[0], vec![copy_work(&mock, &src, &left, 64)])
        .expect("the first batch is accepted");
    builder
        .add_batch(lane_ids[1], vec![copy_work(&mock, &src, &right, 64)])
        .expect("the second batch is accepted");

    let plan = builder
        .build()
        .expect("a read/read pair on the same bytes is not a hazard");
    assert_eq!(plan.batches().len(), 2);
    assert!(
        plan.dependencies().is_empty(),
        "no edge was needed and none was invented"
    );
}

#[test]
fn two_disjoint_writes_need_no_order() {
    let mock = Mock::new();
    let lane_ids = lanes(&mock);
    let src = mock.buffer(64, BufferUsage::COPY_SRC);
    let left = mock.buffer(64, BufferUsage::COPY_DST);
    let right = mock.buffer(64, BufferUsage::COPY_DST);

    let mut builder = SubmissionPlanBuilder::new(mock.device());
    builder
        .add_batch(lane_ids[0], vec![copy_work(&mock, &src, &left, 64)])
        .expect("the first batch is accepted");
    builder
        .add_batch(lane_ids[1], vec![copy_work(&mock, &src, &right, 64)])
        .expect("the second batch is accepted");

    builder
        .build()
        .expect("two writes to different buffers do not overlap");
}

#[test]
fn a_batch_that_mixes_devices_is_refused() {
    let mock = Mock::new();
    let other = Mock::new();
    let lane_ids = lanes(&mock);
    let src = other.buffer(64, BufferUsage::COPY_SRC);
    let dst = other.buffer(64, BufferUsage::COPY_DST);

    let mut builder = SubmissionPlanBuilder::new(mock.device());
    let error = builder
        .add_batch(lane_ids[0], vec![copy_work(&other, &src, &dst, 64)])
        .expect_err("recorded work from another device may not be planned here");
    assert_eq!(error.kind(), RhiErrorKind::WrongDevice);
}

#[test]
fn an_empty_batch_is_refused() {
    let mock = Mock::new();
    let lane_ids = lanes(&mock);
    let mut builder = SubmissionPlanBuilder::new(mock.device());

    let error = builder
        .add_batch(lane_ids[0], Vec::new())
        .expect_err("a batch must carry at least one recording");
    assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
}

#[test]
fn a_foreign_plan_point_is_refused() {
    let mock = Mock::new();
    let lane_ids = lanes(&mock);
    let src = mock.buffer(64, BufferUsage::COPY_SRC);
    let dst = mock.buffer(64, BufferUsage::COPY_DST);

    let mut first = SubmissionPlanBuilder::new(mock.device());
    let point = first
        .add_batch(lane_ids[0], vec![copy_work(&mock, &src, &dst, 64)])
        .expect("the batch is accepted");

    let mut second = SubmissionPlanBuilder::new(mock.device());
    let own = second
        .add_batch(lane_ids[1], vec![copy_work(&mock, &src, &dst, 64)])
        .expect("the batch is accepted");

    let error = second
        .add_dependency(point, own)
        .expect_err("a point from another plan cannot order this one");
    assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
}

#[test]
fn a_plan_identity_is_device_scoped_and_serial_numbered() {
    let mock = Mock::new();
    let first = SubmissionPlanBuilder::new(mock.device());
    let second = SubmissionPlanBuilder::new(mock.device());

    assert_eq!(first.id().device_identity(), mock.identity());
    assert_ne!(
        first.id(),
        second.id(),
        "two plans never share an identity"
    );
    let _: SubmissionPlanId = first.id();
}

// ---------------------------------------------------------------------------
// Section 45.3: frame closure
// ---------------------------------------------------------------------------

#[test]
fn a_plan_that_uses_a_frame_without_presenting_it_is_refused() {
    let mock = Mock::new();
    let lane_ids = lanes(&mock);
    let frame = mock.frame();
    let work = frame_work(&mock, &frame);

    let mut builder = SubmissionPlanBuilder::new(mock.device());
    builder
        .add_batch(lane_ids[0], vec![work])
        .expect("the batch is accepted");

    let error = builder
        .build()
        .expect_err("a plan that uses a frame on the GPU must present it");
    assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
    assert!(
        error.message().contains("does not present it"),
        "the refusal names the missing closure: {}",
        error.message()
    );

    // The unused frame is released rather than leaked when the plan is refused.
    frame.abandon().expect("a never-presented frame can be abandoned");
}

#[test]
fn presenting_the_used_frame_closes_the_plan() {
    let mock = Mock::new();
    let lane_ids = lanes(&mock);
    let frame = mock.frame();
    let frame_id = frame.id();
    let work = frame_work(&mock, &frame);

    let mut builder = SubmissionPlanBuilder::new(mock.device());
    let point = builder
        .add_batch(lane_ids[0], vec![work])
        .expect("the batch is accepted");
    let present = builder
        .present_after(frame, point)
        .expect("the frame follows the batch that uses it");

    let plan = builder.build().expect("the frame use is closed by its present");
    assert_eq!(plan.presents().len(), 1);
    assert_eq!(plan.presents()[0].id(), present);
    assert_eq!(plan.presents()[0].frame_id(), frame_id);
    assert_eq!(plan.presents()[0].after(), point);
    assert_eq!(present.submission_plan(), plan.id());
}

#[test]
fn a_second_present_for_the_same_frame_is_refused() {
    let mock = Mock::new();
    let lane_ids = lanes(&mock);
    let frame = mock.frame();
    // Fault injection: the backend hands out a second token for the same
    // drawable. The honest second `present_after` is impossible, because the
    // first one consumes the frame and moves its state past `Acquired`.
    let duplicate = mock
        .duplicate_outstanding_frame()
        .expect("one frame is outstanding");

    let work = frame_work(&mock, &frame);
    let mut builder = SubmissionPlanBuilder::new(mock.device());
    let point = builder
        .add_batch(lane_ids[0], vec![work])
        .expect("the batch is accepted");
    builder
        .present_after(frame, point)
        .expect("the first present is accepted");

    let error = builder
        .present_after(duplicate, point)
        .expect_err("a frame may be presented at most once");
    assert_eq!(error.kind(), RhiErrorKind::InvalidUsage);
    assert!(
        error.message().contains("at most once"),
        "the refusal names the rule: {}",
        error.message()
    );
}

#[test]
fn the_same_frame_consumed_twice_is_refused_by_its_own_state() {
    let mock = Mock::new();
    let lane_ids = lanes(&mock);
    let frame = mock.frame();
    let work = frame_work(&mock, &frame);

    let mut builder = SubmissionPlanBuilder::new(mock.device());
    let point = builder
        .add_batch(lane_ids[0], vec![work])
        .expect("the batch is accepted");
    builder
        .present_after(frame, point)
        .expect("the first present is accepted");

    // A second acquisition is refused while the first frame is still
    // outstanding, which is the presentation contract the plan relies on.
    let error = mock
        .try_frame()
        .expect_err("one configured presentation holds one outstanding frame");
    assert_eq!(error.kind(), crate::rhi::presentation::AcquireErrorKind::FrameOutstanding);
}

#[test]
fn an_acquire_that_the_backend_refuses_yields_no_frame_to_plan() {
    let mock = Mock::new();
    let lane_ids = lanes(&mock);
    let src = mock.buffer(64, BufferUsage::COPY_SRC);
    let dst = mock.buffer(64, BufferUsage::COPY_DST);

    mock.fail_next_acquire();
    let error = mock
        .try_frame()
        .expect_err("an injected acquire failure is reported");
    assert_eq!(
        error.kind(),
        crate::rhi::presentation::AcquireErrorKind::NotReady
    );
    assert!(
        mock.saw("presentation:acquire"),
        "the backend was asked before it refused"
    );

    // A plan with no frame still builds, which is what makes the acquire failure
    // a presentation problem rather than a plan problem.
    let mut builder = SubmissionPlanBuilder::new(mock.device());
    builder
        .add_batch(lane_ids[0], vec![copy_work(&mock, &src, &dst, 64)])
        .expect("the batch is accepted");
    builder
        .build()
        .expect("a plan with no frame use needs no present");
}

#[test]
fn an_outstanding_frame_can_be_abandoned_without_submitting() {
    let mock = Mock::new();
    let frame = mock.frame();
    let abandoned = frame.id();

    frame.abandon().expect("an acquired frame can be abandoned");

    // With the slot released, the next acquisition succeeds, which is what
    // "abandon" has to mean for a no-submit recovery path to work.
    let next = mock
        .try_frame()
        .expect("the abandoned frame released the outstanding slot");
    assert_ne!(
        next.id(),
        abandoned,
        "the replacement frame is a different acquisition, not the abandoned one"
    );
}

#[test]
fn a_presentation_target_is_scoped_to_the_backend_that_produced_it() {
    let mock = Mock::new();
    let configuration = PresentationConfiguration::new(TextureFormat::Bgra8UnormSrgb);

    let configured = mock
        .device()
        .configure_presentation(&mock.target(), &configuration)
        .expect("the mock configures its own target");
    assert_eq!(configured.device_identity(), mock.identity());

    // A target is a host-owned payload, so a target from another provider is
    // refused rather than configured with a drawable this backend does not own.
    // This is a property of the fixture itself: the mock must refuse what a real
    // backend refuses, or a plan built on it proves nothing about presentation.
    let other = Mock::new();
    let error = mock
        .device()
        .configure_presentation(&other.target(), &configuration)
        .expect_err("a foreign target is refused");
    assert_eq!(error.kind(), RhiErrorKind::WrongDevice);
}

#[test]
fn a_frame_from_another_device_is_refused() {
    let mock = Mock::new();
    let other = Mock::new();
    let lane_ids = lanes(&mock);
    let src = mock.buffer(64, BufferUsage::COPY_SRC);
    let dst = mock.buffer(64, BufferUsage::COPY_DST);

    let mut builder = SubmissionPlanBuilder::new(mock.device());
    let point = builder
        .add_batch(lane_ids[0], vec![copy_work(&mock, &src, &dst, 64)])
        .expect("the batch is accepted");

    let error = builder
        .present_after(other.frame(), point)
        .expect_err("a frame from another device cannot be presented here");
    assert_eq!(error.kind(), RhiErrorKind::WrongDevice);
}
