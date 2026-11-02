//! Focused state-fact tests for the renderer's private retirement rule.
//!
//! These use a deliberately inert RHI backend. They establish classification
//! behavior only; DX12 and WGL execution are covered by the integration
//! consumer and retained evidence.

use std::{
    future::{Ready, ready},
    sync::Arc,
};

use fluxel_rhi::{api::*, backend::*};

use super::*;

struct TestBackend;
struct TestImage;

// SAFETY: the test image has no native state and is always current.
unsafe impl NativeImageLease for TestImage {
    fn try_handoff(&mut self, _: u64) -> Result<()> {
        Ok(())
    }
    fn is_current(&self, _: u64) -> bool {
        true
    }
}

impl Backend for TestBackend {
    type NativeDevice = Arc<()>;
    type NativeBuffer = ();
    type NativeTexture = ();
    type NativeTextureView = ();
    type NativeVideoFrame = ();
    type NativeSampler = ();
    type NativeBindGroupLayout = ();
    type NativePipelineLayout = ();
    type NativeBindGroup = ();
    type NativeShader = ();
    type NativeRasterPipeline = ();
    type NativeComputePipeline = ();
    type NativeMeshPipeline = ();
    type NativeRayPipeline = ();
    type NativePipelineCache = ();
    type NativeAcceleration = ();
    type NativeSparseAllocation = ();
    type NativeQuerySet = ();
    type NativeEncoder = ();
    type NativeCompletion = ();
    type NativeSurface = Arc<()>;
    type NativeSurfaceImage = TestImage;
    type NativeUncertainty = ();
    type MemoryObservation = Unavailable;
    type Compute = Unavailable;
    type Mesh = Unavailable;
    type Ray = Unavailable;
    type Extras = Unavailable;
}

impl CommandBackend for TestBackend {
    fn begin_encoder(_: &Arc<()>, _: LaneId) -> Result<()> {
        Ok(())
    }
    fn transition(_: &mut (), _: &[ResourceTransition<'_, Self>]) -> Result<()> {
        Ok(())
    }
    fn copy_buffer(_: &mut (), _: &(), _: u64, _: &(), _: u64, _: u64) -> Result<()> {
        Ok(())
    }
    fn copy_texture(
        _: &mut (),
        _: TextureRegion<'_, Self>,
        _: TextureRegion<'_, Self>,
    ) -> Result<()> {
        Ok(())
    }
    fn buffer_to_texture(
        _: &mut (),
        _: &(),
        _: TexelCopyBufferLayout,
        _: TextureRegion<'_, Self>,
    ) -> Result<()> {
        Ok(())
    }
    fn texture_to_buffer(
        _: &mut (),
        _: TextureRegion<'_, Self>,
        _: &(),
        _: TexelCopyBufferLayout,
    ) -> Result<()> {
        Ok(())
    }
    fn write_buffer(_: &mut (), _: &(), _: u64, _: &[u8]) -> Result<()> {
        Ok(())
    }
    fn write_texture(
        _: &mut (),
        _: TextureRegion<'_, Self>,
        _: TexelCopyBufferLayout,
        _: &[u8],
    ) -> Result<()> {
        Ok(())
    }
}

impl CompletionBackend for TestBackend {
    type WaitFuture<'a> = Ready<Result<()>>;

    fn completion_status(_: &Arc<()>, _: &()) -> CompletionState {
        CompletionState::Complete
    }

    fn wait_completion<'a>(_: &'a Arc<()>, _: &'a ()) -> Self::WaitFuture<'a> {
        ready(Ok(()))
    }

    fn wait_idle<'a>(_: &'a Arc<()>) -> Self::WaitFuture<'a> {
        ready(Ok(()))
    }
}

struct TestMode;
struct TestOutcome {
    status: ExecutionStatus,
    failed_step: Option<FailedStep>,
    accepted: Option<Vec<StepId>>,
    affected: Option<Vec<StepId>>,
    presents: Option<Vec<PresentReceipt>>,
}

impl ExecutionOutcome<TestBackend> for TestOutcome {
    fn facts(&self) -> ExecutionFacts<'_, TestBackend> {
        ExecutionFacts {
            status: self.status,
            failed_step: self.failed_step,
            accepted: self.accepted.as_deref(),
            affected: self.affected.as_deref(),
            checkpoints: None,
            presents: self.presents.as_deref(),
        }
    }
    fn error(&self) -> Option<&Error> {
        None
    }
}

impl ExecutionBackend for TestBackend {
    type Mode = TestMode;
}
impl ExecutionMode<TestBackend> for TestMode {
    type State = ();
    type Outcome = TestOutcome;
    unsafe fn begin(
        _: &Device<TestBackend>,
        _: SubmissionPlan<TestBackend>,
        _: CompletionRequest,
    ) -> Result<Self::State> {
        unreachable!("fact tests never execute RHI work")
    }
    fn commands<F: FnOnce(&mut CommandEncoder<TestBackend>) -> Result<()>>(
        _: &mut Self::State,
        _: StepId,
        _: F,
    ) -> Result<()> {
        unreachable!("fact tests never execute RHI work")
    }
    fn present(
        _: &mut Self::State,
        _: StepId,
        _: PresentBinding<TestBackend>,
    ) -> std::result::Result<(), PresentError<TestBackend>> {
        unreachable!("fact tests never execute RHI work")
    }
    unsafe fn finish(_: Self::State) -> Self::Outcome {
        unreachable!("fact tests never execute RHI work")
    }
}

fn facts(
    status: ExecutionStatus,
    effect: Option<StepEffect>,
    accepted: Option<Vec<StepId>>,
    affected: Option<Vec<StepId>>,
    presents: Option<Vec<PresentReceipt>>,
) -> TestOutcome {
    TestOutcome {
        status,
        failed_step: effect.map(|effect| FailedStep {
            step: StepId {
                plan: PlanId {
                    device: DeviceId::allocate().unwrap(),
                    ordinal: 1,
                },
                index: 0,
            },
            effect,
        }),
        accepted,
        affected,
        presents,
    }
}

#[test]
fn explicit_clean_rejection_can_release_its_owners() {
    let outcome = facts(
        ExecutionStatus::Rejected,
        None,
        Some(Vec::new()),
        Some(Vec::new()),
        None,
    );
    assert!(releases_without_effect::<TestBackend>(&outcome));
}

#[test]
fn only_explicit_no_effect_failure_facts_release_owners() {
    let empty = || Some(Vec::new());
    let cases = [
        facts(
            ExecutionStatus::Failed,
            Some(StepEffect::None),
            empty(),
            empty(),
            None,
        ),
        facts(
            ExecutionStatus::Failed,
            Some(StepEffect::Partial),
            empty(),
            empty(),
            None,
        ),
        facts(
            ExecutionStatus::Failed,
            Some(StepEffect::Unknown),
            empty(),
            empty(),
            None,
        ),
        facts(ExecutionStatus::Accepted, None, empty(), empty(), None),
        facts(
            ExecutionStatus::PartiallyAccepted,
            None,
            empty(),
            empty(),
            None,
        ),
        facts(ExecutionStatus::Indeterminate, None, empty(), empty(), None),
        facts(ExecutionStatus::Unknown, None, None, Some(Vec::new()), None),
        facts(
            ExecutionStatus::Rejected,
            None,
            Some(Vec::new()),
            Some(Vec::new()),
            Some(vec![PresentReceipt {
                step: None,
                accepted: true,
            }]),
        ),
    ];
    assert!(releases_without_effect::<TestBackend>(&cases[0]));
    for outcome in &cases[1..] {
        assert!(!releases_without_effect::<TestBackend>(outcome));
    }
}

#[test]
fn fixed_raster_fixture_refuses_invalid_extent_vertex_count_and_non_finite_values() {
    let valid = [0u8; 72];
    assert!(validate_raster_bytes(&valid, [8, 8]).is_ok());
    assert!(matches!(
        validate_raster_bytes(&valid, [0, 8]),
        Err(RendererError::InvalidRasterInput)
    ));
    assert!(matches!(
        validate_raster_bytes(&valid[..48], [8, 8]),
        Err(RendererError::InvalidRasterInput)
    ));
    let mut non_finite = valid;
    non_finite[..4].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(matches!(
        validate_raster_bytes(&non_finite, [8, 8]),
        Err(RendererError::InvalidRasterInput)
    ));
}

#[test]
fn raster_readback_strips_alignment_padding_without_accepting_short_rows() {
    let bytes = [1u8; 512];
    let tight = strip_raster_padding(bytes.to_vec(), [1, 2], 256).unwrap();
    assert_eq!(tight, vec![1; 8]);
    assert!(matches!(
        strip_raster_padding(vec![0; 511], [1, 2], 256),
        Err(RendererError::InvalidRasterInput)
    ));
}

#[test]
fn fixed_compute_fixture_refuses_empty_or_unaligned_u32_payloads() {
    assert_eq!(validate_compute_bytes(&[0; 4]).unwrap(), 1);
    assert!(matches!(
        validate_compute_bytes(&[]),
        Err(RendererError::InvalidComputeInput)
    ));
    assert!(matches!(
        validate_compute_bytes(&[0; 3]),
        Err(RendererError::InvalidComputeInput)
    ));
    assert_eq!(compute_dispatch_groups(64).unwrap(), 1);
    assert_eq!(compute_dispatch_groups(65).unwrap(), 2);
}

#[test]
fn compute_fixture_reports_unsupported_before_resource_work() {
    let id = DeviceId::allocate().unwrap();
    // SAFETY: this inert fixture pairs every native handle with one device ID.
    let device = unsafe {
        Device::<TestBackend>::from_native(id, Arc::new(()), DeviceCapabilities::default())
    };
    // SAFETY: these inert handles are associated with the same fixture device.
    let pipeline: ComputePipeline<TestBackend> = unsafe { ComputePipeline::from_native(id, ()) };
    // SAFETY: this inert layout has no native state and belongs to the fixture device.
    let layout: BindGroupLayout<TestBackend> =
        unsafe { BindGroupLayout::from_native(id, (), Arc::from([])) };
    assert!(matches!(
        validate_compute_input(&[0; 4], pipeline.device(), layout.device(), &device),
        Err(RendererError::UnsupportedCompute)
    ));
}
