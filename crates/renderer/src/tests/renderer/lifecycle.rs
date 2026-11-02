//! Deterministic ownership tests for renderer submissions.
//!
//! The backend below reports only scripted logical execution facts.  It is not
//! a GPU conformance backend; DX12 and WGL evidence remains the integration
//! consumer's responsibility.

use std::{
    future::{Future, Ready, ready},
    ops::Range,
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use fluxel_assets::{Acquire, AssetKind, AssetSnapshot, AssetStore, ResidentBytes};
use fluxel_rhi::{api::*, backend::*};

use super::*;

struct Bytes;
impl AssetKind for Bytes {}

#[derive(Clone, Copy)]
enum Script {
    CleanRejected,
    AllocationDeviceLost,
    MapDeviceLost,
    FlushDeviceLost,
    BeginDeviceLost,
    RecordingDeviceLost,
    FinishDeviceLost,
    PartialAccepted,
    PartialIssued,
    RecordingFailureAfterPrefix,
    Indeterminate,
    AcceptedExact,
    AcceptedWithError,
    AcceptedWrongFrontier,
    AcceptedWaitPending,
    AcceptedIdleFailure,
    AcceptedIdlePending,
    AcceptedIdleDeviceLost,
    AcceptedCompletionDeviceLost,
    AcceptedReadbackDeviceLost,
    SurfacePresentRejected,
    SurfacePresentDeviceLost,
}

struct Native {
    script: Script,
    fault_enabled: AtomicBool,
    finishes: AtomicUsize,
    next_completion: AtomicUsize,
    command_calls: AtomicUsize,
    copied_bytes: Mutex<Vec<u64>>,
    copy_destinations: Mutex<Vec<Range<u64>>>,
    uniform_offsets: Mutex<Vec<u32>>,
    texture_uploads: Mutex<Vec<(u64, u32)>>,
    mapped_bytes: Mutex<Vec<Vec<u8>>>,
    readback_ranges: Mutex<Vec<Range<u64>>>,
    waits: AtomicUsize,
    idle_waits: AtomicUsize,
    waited_completions: Mutex<Vec<usize>>,
    created_buffers: AtomicUsize,
    dropped_buffers: AtomicUsize,
    outcome: Mutex<Weak<OutcomeGuard>>,
}

struct NativeBuffer {
    native: Arc<Native>,
}

struct NativeTexture;

impl Drop for NativeBuffer {
    fn drop(&mut self) {
        self.native.dropped_buffers.fetch_add(1, Ordering::SeqCst);
    }
}

struct OutcomeGuard;

struct CompletionMarker {
    id: usize,
}

struct WitnessBackend;
#[derive(Clone)]
struct WitnessImage {
    generation: Arc<std::sync::atomic::AtomicU64>,
}

// SAFETY: this test-only image has no native state or handoff side effects.
unsafe impl NativeImageLease for WitnessImage {
    fn try_handoff(&mut self, generation: u64) -> Result<()> {
        if self.generation.load(Ordering::SeqCst) != generation {
            return Err(Error::SurfaceOutdated);
        }
        Ok(())
    }
    fn is_current(&self, generation: u64) -> bool {
        self.generation.load(Ordering::SeqCst) == generation
    }
}

#[derive(Clone)]
struct WitnessSurface {
    generation: Arc<std::sync::atomic::AtomicU64>,
}

impl Backend for WitnessBackend {
    type NativeDevice = Arc<Native>;
    type NativeBuffer = Arc<NativeBuffer>;
    type NativeTexture = Arc<NativeTexture>;
    type NativeTextureView = ();
    type NativeVideoFrame = ();
    type NativeSampler = ();
    type NativeBindGroupLayout = ();
    type NativePipelineLayout = ();
    type NativeBindGroup = Arc<Native>;
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
    type NativeCompletion = Arc<CompletionMarker>;
    type NativeSurface = WitnessSurface;
    type NativeSurfaceImage = WitnessImage;
    type NativeUncertainty = ();
    type MemoryObservation = Unavailable;
    type Compute = Unavailable;
    type Mesh = Unavailable;
    type Ray = Unavailable;
    type Extras = Unavailable;
}

impl SurfaceBackend for WitnessBackend {
    type ConfigureFuture<'a> = Ready<Result<()>>;
    type AcquireFuture<'a> = Ready<Result<WitnessImage>>;
    fn generation(surface: &WitnessSurface) -> u64 {
        surface.generation.load(Ordering::SeqCst)
    }
    fn capabilities(_: &WitnessSurface) -> Result<SurfaceCapabilities> {
        Err(Error::Unsupported("test surface capabilities"))
    }
    fn supports_config(_: &WitnessSurface, _: &SurfaceConfig) -> Support<()> {
        Support::Supported(())
    }
    fn configure<'a>(
        surface: &'a mut WitnessSurface,
        _: &'a SurfaceConfig,
    ) -> Self::ConfigureFuture<'a> {
        surface.generation.fetch_add(1, Ordering::SeqCst);
        ready(Ok(()))
    }
    fn acquire<'a>(surface: &'a mut WitnessSurface) -> Self::AcquireFuture<'a> {
        ready(Ok(WitnessImage {
            generation: Arc::clone(&surface.generation),
        }))
    }
}

impl ResourceBackend for WitnessBackend {
    fn buffer_support(_: &Arc<Native>, _: &BufferDesc) -> Support<BufferSupport> {
        Support::Supported(BufferSupport {
            alignment: 1,
            max_size: u64::MAX,
        })
    }
    fn texture_support(_: &Arc<Native>, _: &TextureDesc) -> Support<TextureSupport> {
        Support::Supported(TextureSupport {
            max_extent: [u32::MAX; 3],
            max_array_layers: u32::MAX,
            max_mips: u32::MAX,
            sample_counts: vec![1],
            compatible_view_formats: Vec::new(),
            filterable: true,
            blendable: true,
            storage: true,
        })
    }
    fn create_buffer(device: &Arc<Native>, _: &BufferDesc) -> Result<Arc<NativeBuffer>> {
        if device.fault_enabled.load(Ordering::SeqCst)
            && matches!(device.script, Script::AllocationDeviceLost)
        {
            return Err(Error::DeviceLost);
        }
        device.created_buffers.fetch_add(1, Ordering::SeqCst);
        Ok(Arc::new(NativeBuffer {
            native: Arc::clone(device),
        }))
    }
    fn create_texture(_: &Arc<Native>, _: &TextureDesc) -> Result<Arc<NativeTexture>> {
        Ok(Arc::new(NativeTexture))
    }
    fn create_view(_: &Arc<Native>, _: &Arc<NativeTexture>, _: &TextureViewDesc) -> Result<()> {
        Ok(())
    }
    fn create_sampler(_: &Arc<Native>, _: &SamplerDesc) -> Result<()> {
        Err(Error::Unsupported("sampler"))
    }
    fn create_bind_group_layout(_: &Arc<Native>, _: &BindGroupLayoutDesc) -> Result<()> {
        Err(Error::Unsupported("layout"))
    }
    fn create_bind_group(device: &Arc<Native>, _: &BindGroupDesc<'_, Self>) -> Result<Arc<Native>> {
        Ok(Arc::clone(device))
    }
    fn create_pipeline_layout(_: &Arc<Native>, _: &PipelineLayoutDesc<'_, Self>) -> Result<()> {
        Err(Error::Unsupported("pipeline layout"))
    }
}

struct WitnessMap {
    bytes: Vec<u8>,
    native: Arc<Native>,
}
impl MappingBackend for WitnessBackend {
    type NativeMap = WitnessMap;
    type MapFuture<'a> = Ready<Result<WitnessMap>>;
    fn map<'a>(
        device: &'a Arc<Native>,
        _: &'a Arc<NativeBuffer>,
        range: Range<u64>,
        _: MapMode,
    ) -> Self::MapFuture<'a> {
        if device.fault_enabled.load(Ordering::SeqCst)
            && matches!(device.script, Script::MapDeviceLost)
        {
            return ready(Err(Error::DeviceLost));
        }
        ready(Ok(WitnessMap {
            bytes: vec![0; usize::try_from(range.end - range.start).unwrap()],
            native: Arc::clone(device),
        }))
    }
    fn bytes(map: &WitnessMap) -> &[u8] {
        &map.bytes
    }
    fn bytes_mut(map: &mut WitnessMap) -> &mut [u8] {
        &mut map.bytes
    }
    fn flush(map: &mut WitnessMap, _: Range<u64>) -> Result<()> {
        map.native
            .mapped_bytes
            .lock()
            .unwrap()
            .push(map.bytes.clone());
        if map.native.fault_enabled.load(Ordering::SeqCst)
            && matches!(map.native.script, Script::FlushDeviceLost)
        {
            return Err(Error::DeviceLost);
        }
        Ok(())
    }
    fn invalidate(_: &mut WitnessMap, _: Range<u64>) -> Result<()> {
        Ok(())
    }
    fn unmap(_: WitnessMap) {}
}

impl CommandBackend for WitnessBackend {
    fn begin_encoder(_: &Arc<Native>, _: LaneId) -> Result<()> {
        Ok(())
    }
    fn transition(_: &mut (), _: &[ResourceTransition<'_, Self>]) -> Result<()> {
        Ok(())
    }
    fn copy_buffer(
        _: &mut (),
        source: &Arc<NativeBuffer>,
        _: u64,
        _: &Arc<NativeBuffer>,
        destination: u64,
        bytes: u64,
    ) -> Result<()> {
        source.native.copied_bytes.lock().unwrap().push(bytes);
        source
            .native
            .copy_destinations
            .lock()
            .unwrap()
            .push(destination..destination + bytes);
        Ok(())
    }
    fn copy_texture(
        _: &mut (),
        _: TextureRegion<'_, Self>,
        _: TextureRegion<'_, Self>,
    ) -> Result<()> {
        Err(Error::Unsupported("texture"))
    }
    fn buffer_to_texture(
        _: &mut (),
        source: &Arc<NativeBuffer>,
        layout: TexelCopyBufferLayout,
        _: TextureRegion<'_, Self>,
    ) -> Result<()> {
        source
            .native
            .texture_uploads
            .lock()
            .unwrap()
            .push((layout.offset, layout.bytes_per_row));
        Ok(())
    }
    fn texture_to_buffer(
        _: &mut (),
        _: TextureRegion<'_, Self>,
        _: &Arc<NativeBuffer>,
        _: TexelCopyBufferLayout,
    ) -> Result<()> {
        Ok(())
    }
    fn write_buffer(_: &mut (), _: &Arc<NativeBuffer>, _: u64, _: &[u8]) -> Result<()> {
        Ok(())
    }
    fn write_texture(
        _: &mut (),
        _: TextureRegion<'_, Self>,
        _: TexelCopyBufferLayout,
        _: &[u8],
    ) -> Result<()> {
        Err(Error::Unsupported("texture"))
    }
}

impl RasterBackend for WitnessBackend {
    fn begin_raster(_: &mut (), _: &RasterPassDesc<'_, Self>) -> Result<()> {
        Ok(())
    }
    fn set_pipeline(_: &mut (), _: &()) -> Result<()> {
        Ok(())
    }
    fn set_bind_group(_: &mut (), _: u32, group: &Arc<Native>, offsets: &[u32]) -> Result<()> {
        group
            .uniform_offsets
            .lock()
            .unwrap()
            .extend_from_slice(offsets);
        Ok(())
    }
    fn set_vertex_buffer(_: &mut (), _: u32, _: &Arc<NativeBuffer>, _: u64, _: u64) -> Result<()> {
        Ok(())
    }
    fn set_index_buffer(
        _: &mut (),
        _: &Arc<NativeBuffer>,
        _: u64,
        _: u64,
        _: IndexFormat,
    ) -> Result<()> {
        Ok(())
    }
    fn viewport(_: &mut (), _: Viewport) -> Result<()> {
        Ok(())
    }
    fn scissor(_: &mut (), _: Rect) -> Result<()> {
        Ok(())
    }
    fn blend_constant(_: &mut (), _: [f32; 4]) -> Result<()> {
        Ok(())
    }
    fn stencil_reference(_: &mut (), _: u32) -> Result<()> {
        Ok(())
    }
    fn draw(_: &mut (), _: Range<u32>, _: Range<u32>) -> Result<()> {
        Ok(())
    }
    fn draw_indexed(_: &mut (), _: Range<u32>, _: i32, _: Range<u32>) -> Result<()> {
        Ok(())
    }
    fn end_raster(_: &mut ()) -> Result<()> {
        Ok(())
    }
    fn abort_raster(_: &mut ()) {}
}

impl CompletionBackend for WitnessBackend {
    type WaitFuture<'a> = WitnessWait;
    fn completion_status(_: &Arc<Native>, _: &Arc<CompletionMarker>) -> CompletionState {
        CompletionState::Complete
    }
    fn wait_completion<'a>(
        device: &'a Arc<Native>,
        point: &'a Arc<CompletionMarker>,
    ) -> Self::WaitFuture<'a> {
        device.waited_completions.lock().unwrap().push(point.id);
        WitnessWait {
            native: Arc::clone(device),
            idle: false,
        }
    }
    fn wait_idle<'a>(device: &'a Arc<Native>) -> Self::WaitFuture<'a> {
        WitnessWait {
            native: Arc::clone(device),
            idle: true,
        }
    }
}

struct WitnessWait {
    native: Arc<Native>,
    idle: bool,
}
impl Future for WitnessWait {
    type Output = Result<()>;
    fn poll(
        self: std::pin::Pin<&mut Self>,
        _: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        self.native.waits.fetch_add(1, Ordering::SeqCst);
        if self.idle {
            self.native.idle_waits.fetch_add(1, Ordering::SeqCst);
        }
        if (matches!(self.native.script, Script::AcceptedWaitPending) && !self.idle)
            || (matches!(self.native.script, Script::AcceptedIdlePending) && self.idle)
        {
            std::task::Poll::Pending
        } else if self.idle && matches!(self.native.script, Script::AcceptedIdleFailure) {
            std::task::Poll::Ready(Err(Error::Backend("scripted idle failure".to_owned())))
        } else if (self.idle && matches!(self.native.script, Script::AcceptedIdleDeviceLost))
            || (!self.idle && matches!(self.native.script, Script::AcceptedCompletionDeviceLost))
        {
            std::task::Poll::Ready(Err(Error::DeviceLost))
        } else {
            std::task::Poll::Ready(Ok(()))
        }
    }
}

impl ReadbackBackend for WitnessBackend {
    type NativeReadback = Range<u64>;
    type ReadbackFuture<'a> = Ready<Result<ReadbackData>>;
    fn encode_readback(
        _: &mut (),
        buffer: &Arc<NativeBuffer>,
        range: Range<u64>,
    ) -> Result<Range<u64>> {
        buffer
            .native
            .readback_ranges
            .lock()
            .unwrap()
            .push(range.clone());
        Ok(range)
    }
    fn map_readback<'a>(device: &'a Arc<Native>, range: Range<u64>) -> Self::ReadbackFuture<'a> {
        if matches!(device.script, Script::AcceptedReadbackDeviceLost) {
            return ready(Err(Error::DeviceLost));
        }
        ready(Ok(ReadbackData {
            bytes: if range == (0..2048) {
                vec![0; 2048]
            } else {
                vec![]
            },
            row_pitch: 0,
            slice_pitch: 0,
        }))
    }
}

struct WitnessMode;
struct WitnessState {
    native: Arc<Native>,
    device: Device<WitnessBackend>,
    steps: Vec<StepId>,
    lanes: Vec<LaneId>,
    calls: usize,
    present: Option<PresentReceipt>,
}
struct WitnessOutcome {
    guard: Arc<OutcomeGuard>,
    error: Option<Error>,
    status: ExecutionStatus,
    failed_step: Option<FailedStep>,
    accepted: Option<Vec<StepId>>,
    affected: Option<Vec<StepId>>,
    checkpoints: Option<Vec<Checkpoint<WitnessBackend>>>,
    presents: Option<Vec<PresentReceipt>>,
}

impl ExecutionOutcome<WitnessBackend> for WitnessOutcome {
    fn facts(&self) -> ExecutionFacts<'_, WitnessBackend> {
        // The original outcome owns this guard; keep that ownership observable.
        let _ = &self.guard;
        ExecutionFacts {
            status: self.status,
            failed_step: self.failed_step,
            accepted: self.accepted.as_deref(),
            affected: self.affected.as_deref(),
            checkpoints: self.checkpoints.as_deref(),
            presents: self.presents.as_deref(),
        }
    }
    fn error(&self) -> Option<&Error> {
        self.error.as_ref()
    }
}
impl ExecutionBackend for WitnessBackend {
    type Mode = WitnessMode;
}
impl DeferredBackend for WitnessBackend {
    type NativeRecordedWork = ();
    fn finish_encoder(_: (), _: LaneId) -> Result<()> {
        Ok(())
    }
    unsafe fn submit(_: &Arc<Native>, _: SubmitPacket<Self>) -> SubmitOutcome<Self> {
        unreachable!()
    }
    unsafe fn resume(_: &Arc<Native>, _: DeferredTail<Self>) -> SubmitOutcome<Self> {
        unreachable!()
    }
}
impl ExecutionMode<WitnessBackend> for WitnessMode {
    type State = WitnessState;
    type Outcome = WitnessOutcome;
    unsafe fn begin(
        device: &Device<WitnessBackend>,
        plan: SubmissionPlan<WitnessBackend>,
        _: CompletionRequest,
    ) -> Result<Self::State> {
        if unsafe { device.native_ref() }
            .fault_enabled
            .load(Ordering::SeqCst)
            && matches!(
                unsafe { device.native_ref() }.script,
                Script::BeginDeviceLost
            )
        {
            return Err(Error::DeviceLost);
        }
        // SAFETY: the test fixture constructed this Device from the same Arc.
        Ok(WitnessState {
            native: unsafe { Arc::clone(device.native_ref()) },
            device: device.clone(),
            steps: plan
                .steps()
                .iter()
                .enumerate()
                .map(|(index, _)| plan.step(index))
                .collect(),
            lanes: plan.steps().iter().map(|step| step.lane).collect(),
            calls: 0,
            present: None,
        })
    }
    fn commands<F: FnOnce(&mut CommandEncoder<WitnessBackend>) -> Result<()>>(
        state: &mut Self::State,
        step: StepId,
        encode: F,
    ) -> Result<()> {
        let call = state.calls;
        state.calls += 1;
        state.native.command_calls.fetch_add(1, Ordering::SeqCst);
        if state.native.fault_enabled.load(Ordering::SeqCst)
            && matches!(state.native.script, Script::RecordingDeviceLost)
            && call == 1
        {
            Err(Error::DeviceLost)
        } else if matches!(state.native.script, Script::RecordingFailureAfterPrefix) && call == 1 {
            Err(Error::Backend("scripted recording failure".to_owned()))
        } else {
            if state.steps.get(call).copied() != Some(step) {
                return Err(Error::InvalidState);
            }
            let mut encoder = state.device.encoder(state.lanes[call])?;
            encode(&mut encoder)?;
            let _ = encoder.finish()?;
            Ok(())
        }
    }
    fn present(
        state: &mut Self::State,
        step: StepId,
        mut binding: PresentBinding<WitnessBackend>,
    ) -> std::result::Result<(), PresentError<WitnessBackend>> {
        match state.native.script {
            Script::SurfacePresentRejected => Err(PresentError {
                error: Error::SurfaceOutdated,
                binding: Some(binding),
            }),
            Script::SurfacePresentDeviceLost => Err(PresentError {
                error: Error::DeviceLost,
                binding: None,
            }),
            _ => {
                unsafe { binding.try_handoff() }.map_err(|error| PresentError {
                    error,
                    binding: Some(binding),
                })?;
                state.present = Some(PresentReceipt {
                    step: Some(step),
                    accepted: true,
                });
                Ok(())
            }
        }
    }
    unsafe fn finish(state: Self::State) -> Self::Outcome {
        state.native.finishes.fetch_add(1, Ordering::SeqCst);
        let guard = Arc::new(OutcomeGuard);
        *state.native.outcome.lock().unwrap() = Arc::downgrade(&guard);
        let script = if state.native.fault_enabled.load(Ordering::SeqCst) {
            state.native.script
        } else {
            Script::AcceptedExact
        };
        let failed_step = matches!(script, Script::PartialIssued).then_some(FailedStep {
            step: state.steps[1],
            effect: StepEffect::Partial,
        });
        let (status, accepted, affected, checkpoints) = match script {
            Script::CleanRejected => (
                ExecutionStatus::Rejected,
                Some(Vec::new()),
                Some(Vec::new()),
                None,
            ),
            Script::RecordingFailureAfterPrefix
            | Script::RecordingDeviceLost
            | Script::PartialAccepted => (
                ExecutionStatus::PartiallyAccepted,
                Some(vec![state.steps[0]]),
                Some(vec![state.steps[1]]),
                None,
            ),
            Script::PartialIssued => (
                ExecutionStatus::Failed,
                Some(vec![state.steps[0]]),
                Some(vec![state.steps[1]]),
                None,
            ),
            Script::Indeterminate | Script::FinishDeviceLost | Script::SurfacePresentDeviceLost => {
                (ExecutionStatus::Indeterminate, None, None, None)
            }
            Script::SurfacePresentRejected => (
                ExecutionStatus::Rejected,
                Some(Vec::new()),
                Some(Vec::new()),
                None,
            ),
            Script::AcceptedExact
            | Script::AcceptedWithError
            | Script::AcceptedWaitPending
            | Script::AcceptedWrongFrontier
            | Script::AcceptedIdleFailure
            | Script::AcceptedIdlePending
            | Script::AcceptedIdleDeviceLost
            | Script::AcceptedCompletionDeviceLost
            | Script::AcceptedReadbackDeviceLost
            | Script::AllocationDeviceLost
            | Script::MapDeviceLost
            | Script::FlushDeviceLost
            | Script::BeginDeviceLost => {
                let frontier = if matches!(state.native.script, Script::AcceptedWrongFrontier) {
                    state.steps[0]
                } else {
                    *state.steps.last().expect("copy plan has readback")
                };
                let completion_id = state.native.next_completion.fetch_add(1, Ordering::SeqCst) + 1;
                let point = unsafe {
                    CompletionPoint::from_native(
                        state.device.id(),
                        Arc::new(CompletionMarker { id: completion_id }),
                    )
                };
                (
                    ExecutionStatus::Accepted,
                    Some(state.steps),
                    Some(Vec::new()),
                    Some(vec![Checkpoint {
                        point,
                        covers: CompletionCoverage {
                            lane_frontiers: vec![(state.lanes[0], frontier)],
                        },
                    }]),
                )
            }
        };
        WitnessOutcome {
            guard,
            error: if matches!(script, Script::AcceptedWithError) {
                Some(Error::Backend("accepted with error".to_owned()))
            } else {
                matches!(
                    script,
                    Script::FinishDeviceLost | Script::SurfacePresentDeviceLost
                )
                .then_some(Error::DeviceLost)
            },
            status,
            failed_step,
            accepted,
            affected,
            checkpoints,
            presents: state.present.map(|receipt| vec![receipt]),
        }
    }
}

fn fixture(script: Script) -> (Device<WitnessBackend>, Arc<Native>) {
    fixture_with_caps(script, |_| {})
}

fn fixture_with_caps(
    script: Script,
    adjust: impl FnOnce(&mut DeviceCapabilities),
) -> (Device<WitnessBackend>, Arc<Native>) {
    let id = DeviceId::allocate().unwrap();
    let native = Arc::new(Native {
        script,
        fault_enabled: AtomicBool::new(true),
        finishes: AtomicUsize::new(0),
        next_completion: AtomicUsize::new(0),
        command_calls: AtomicUsize::new(0),
        copied_bytes: Mutex::new(Vec::new()),
        copy_destinations: Mutex::new(Vec::new()),
        uniform_offsets: Mutex::new(Vec::new()),
        texture_uploads: Mutex::new(Vec::new()),
        mapped_bytes: Mutex::new(Vec::new()),
        readback_ranges: Mutex::new(Vec::new()),
        waits: AtomicUsize::new(0),
        idle_waits: AtomicUsize::new(0),
        waited_completions: Mutex::new(Vec::new()),
        created_buffers: AtomicUsize::new(0),
        dropped_buffers: AtomicUsize::new(0),
        outcome: Mutex::new(Weak::new()),
    });
    let mut caps = DeviceCapabilities {
        limits: DeviceLimits {
            max_buffer_bytes: 1 << 20,
            max_texture_dimension_2d: 4096,
            max_texture_array_layers: 1,
            max_color_attachments: 1,
            max_vertex_buffers: 1,
            max_vertex_attributes: 2,
            max_bind_groups: 1,
            max_bindings_per_group: 2,
            uniform_offset_alignment: 256,
            ..DeviceLimits::default()
        },
        lanes: Arc::from([LaneInfo {
            id: LaneId {
                device: id,
                ordinal: 0,
            },
            class: LaneClass::Graphics,
            routes: Vec::new(),
        }]),
        ..DeviceCapabilities::default()
    };
    adjust(&mut caps);
    // SAFETY: native state and capabilities were created together for this ID.
    (
        unsafe { Device::from_native(id, Arc::clone(&native), caps) },
        native,
    )
}

fn snapshot() -> (AssetSnapshot<Bytes, Vec<u8>>, Weak<Vec<u8>>) {
    snapshot_bytes(vec![1, 2, 3, 4])
}

fn snapshot_bytes(bytes: Vec<u8>) -> (AssetSnapshot<Bytes, Vec<u8>>, Weak<Vec<u8>>) {
    let store = AssetStore::<Bytes, Vec<u8>, ()>::new();
    let handle = store.create().unwrap();
    let size = u64::try_from(bytes.len()).unwrap();
    let snapshot = match store.acquire(&handle).unwrap() {
        Acquire::Producer(producer) => producer.commit(bytes, ResidentBytes::new(size)).unwrap(),
        _ => unreachable!(),
    };
    let weak = Arc::downgrade(&snapshot.value_arc());
    (snapshot, weak)
}

fn submit(
    renderer: &mut Renderer<WitnessBackend>,
    snapshot: AssetSnapshot<Bytes, Vec<u8>>,
) -> std::result::Result<Submission, RendererError> {
    block_on(renderer.submit(snapshot))
}

fn block_on<F: Future>(future: F) -> F::Output {
    let waker = std::task::Waker::noop();
    let mut context = std::task::Context::from_waker(waker);
    let mut future = std::pin::pin!(future);
    match future.as_mut().poll(&mut context) {
        std::task::Poll::Ready(output) => output,
        std::task::Poll::Pending => panic!("the deterministic fixture must be ready"),
    }
}

fn poll_once_pending<F: Future>(future: F) {
    let waker = std::task::Waker::noop();
    let mut context = std::task::Context::from_waker(waker);
    let mut future = std::pin::pin!(future);
    assert!(matches!(
        future.as_mut().poll(&mut context),
        std::task::Poll::Pending
    ));
}

#[path = "raster.rs"]
mod raster_tests;
#[path = "shutdown.rs"]
mod shutdown_tests;
#[path = "surface.rs"]
mod surface_tests;

#[test]
fn accepted_exact_completion_retires_actual_owners_after_readback() {
    let (device, native) = fixture(Script::AcceptedExact);
    let (snapshot, weak) = snapshot();
    let mut renderer = Renderer::new(device).unwrap();
    let submission = submit(&mut renderer, snapshot).unwrap();
    assert_eq!(renderer.pending(), 1);
    assert_eq!(
        block_on(renderer.readback(&submission)).unwrap(),
        Vec::<u8>::new()
    );
    assert_eq!(native.waits.load(Ordering::SeqCst), 1);
    assert_eq!(renderer.pending(), 0);
    assert!(weak.upgrade().is_none());
    assert_eq!(native.dropped_buffers.load(Ordering::SeqCst), 2);
    assert!(native.outcome.lock().unwrap().upgrade().is_none());
}

#[test]
fn dynamic_copy_lengths_refresh_bindings_and_record_current_frame_ranges() {
    let (device, native) = fixture(Script::AcceptedExact);
    let mut renderer = Renderer::new(device).unwrap();
    let first = submit(&mut renderer, snapshot_bytes(vec![1; 4]).0).unwrap();
    let plan = renderer.plan_id().unwrap();
    let second = submit(&mut renderer, snapshot_bytes(vec![2; 12]).0).unwrap();
    let third = submit(&mut renderer, snapshot_bytes(vec![3; 8]).0).unwrap();
    assert_eq!(renderer.pending(), 3);
    assert_eq!(renderer.compilations(), 1);
    assert_eq!(renderer.binding_refreshes(), 3);
    assert_eq!(renderer.plan_id(), Some(plan));
    assert_eq!(*native.copied_bytes.lock().unwrap(), [4, 12, 8]);
    assert_eq!(*native.readback_ranges.lock().unwrap(), [0..4, 0..12, 0..8]);
    for submission in [&first, &second, &third] {
        block_on(renderer.readback(submission)).unwrap();
    }
    assert_eq!(renderer.pending(), 0);
    assert_eq!(*native.waited_completions.lock().unwrap(), [1, 2, 3]);
}

#[test]
fn wrong_completion_frontier_refuses_before_wait_or_retirement() {
    let (device, native) = fixture(Script::AcceptedWrongFrontier);
    let (snapshot, weak) = snapshot();
    let mut renderer = Renderer::new(device).unwrap();
    let submission = submit(&mut renderer, snapshot).unwrap();
    assert!(matches!(
        block_on(renderer.readback(&submission)),
        Err(RendererError::MissingCompletion)
    ));
    assert_eq!(native.waits.load(Ordering::SeqCst), 0);
    assert_eq!(renderer.pending(), 1);
    assert!(weak.upgrade().is_some());
    assert_eq!(native.dropped_buffers.load(Ordering::SeqCst), 0);
}

#[test]
fn cancelled_wait_future_does_not_retire_pending_owners() {
    let (device, native) = fixture(Script::AcceptedWaitPending);
    let (snapshot, weak) = snapshot();
    let mut renderer = Renderer::new(device).unwrap();
    let submission = submit(&mut renderer, snapshot).unwrap();
    poll_once_pending(renderer.readback(&submission));
    assert_eq!(native.waits.load(Ordering::SeqCst), 1);
    assert_eq!(renderer.pending(), 1);
    assert!(weak.upgrade().is_some());
    assert_eq!(native.dropped_buffers.load(Ordering::SeqCst), 0);
}

#[test]
fn submissions_are_bounded_and_foreign_or_stale_handles_never_wait() {
    let (device, native) = fixture(Script::AcceptedExact);
    let mut renderer = Renderer::new(device.clone()).unwrap();
    let (first_snapshot, first_value) = snapshot();
    let (second_snapshot, second_value) = snapshot();
    let (third_snapshot, third_value) = snapshot();
    let first = submit(&mut renderer, first_snapshot).unwrap();
    let second = submit(&mut renderer, second_snapshot).unwrap();
    let third = submit(&mut renderer, third_snapshot).unwrap();
    assert_eq!(renderer.pending(), 3);
    assert_eq!(renderer.compilations(), 1);
    let plan = renderer.plan_id().unwrap();
    assert_eq!(renderer.plan_id(), Some(plan));
    assert_ne!(first.serial, second.serial);
    assert_ne!(second.serial, third.serial);
    assert_ne!(first.serial, third.serial);
    assert!(first_value.upgrade().is_some());
    assert!(second_value.upgrade().is_some());
    assert!(third_value.upgrade().is_some());
    assert!(matches!(
        submit(&mut renderer, snapshot().0),
        Err(RendererError::Backpressure)
    ));

    let mut other = Renderer::new(device).unwrap();
    assert!(matches!(
        block_on(other.readback(&first)),
        Err(RendererError::ForeignSubmission)
    ));
    assert_eq!(native.waits.load(Ordering::SeqCst), 0);

    block_on(renderer.readback(&first)).unwrap();
    assert_eq!(*native.waited_completions.lock().unwrap(), [1]);
    assert!(first_value.upgrade().is_none());
    assert!(second_value.upgrade().is_some());
    assert!(third_value.upgrade().is_some());
    assert!(matches!(
        block_on(renderer.readback(&first)),
        Err(RendererError::StaleSubmission)
    ));
    assert_eq!(native.waits.load(Ordering::SeqCst), 1);
    block_on(renderer.readback(&second)).unwrap();
    assert_eq!(*native.waited_completions.lock().unwrap(), [1, 2]);
    assert!(second_value.upgrade().is_none());
    assert!(third_value.upgrade().is_some());
    block_on(renderer.readback(&third)).unwrap();
    assert_eq!(*native.waited_completions.lock().unwrap(), [1, 2, 3]);
    assert!(third_value.upgrade().is_none());
}

#[test]
fn unsafe_outcomes_retain_actual_snapshot_native_buffers_and_original_outcome() {
    for script in [Script::RecordingFailureAfterPrefix, Script::Indeterminate] {
        let (device, native) = fixture(script);
        let (snapshot, weak) = snapshot();
        let mut renderer = Renderer::new(device).unwrap();
        match submit(&mut renderer, snapshot) {
            Err(RendererError::RetainedExecution { .. }) => {}
            Err(error) => panic!("unexpected renderer error: {error:?}"),
            Ok(_) => panic!("unsafe execution must not return a submission"),
        }
        assert_eq!(native.finishes.load(Ordering::SeqCst), 1);
        if matches!(script, Script::RecordingFailureAfterPrefix) {
            assert_eq!(native.command_calls.load(Ordering::SeqCst), 2);
        }
        assert_eq!(renderer.pending(), 1);
        assert!(weak.upgrade().is_some());
        assert_eq!(native.created_buffers.load(Ordering::SeqCst), 2);
        assert_eq!(native.dropped_buffers.load(Ordering::SeqCst), 0);
        assert!(native.outcome.lock().unwrap().upgrade().is_some());
        if matches!(script, Script::RecordingFailureAfterPrefix) {
            let pending = renderer.slots[0]
                .as_ref()
                .expect("partial work stays quarantined");
            let facts = pending.outcome.facts();
            assert_eq!(facts.status, ExecutionStatus::PartiallyAccepted);
            assert_eq!(facts.accepted.map(|steps| steps.len()), Some(1));
            assert_eq!(facts.affected.map(|steps| steps.len()), Some(1));
            assert!(facts.checkpoints.is_none());
        }
        drop(renderer);
        // Retained uncertain work intentionally leaks native ownership at drop.
        assert_eq!(native.dropped_buffers.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn clean_rejection_finishes_and_releases_actual_owners() {
    let (device, native) = fixture(Script::CleanRejected);
    let (snapshot, weak) = snapshot();
    let mut renderer = Renderer::new(device).unwrap();
    assert!(matches!(
        submit(&mut renderer, snapshot),
        Err(RendererError::Rhi(_))
    ));
    assert_eq!(native.finishes.load(Ordering::SeqCst), 1);
    assert_eq!(renderer.pending(), 0);
    assert!(weak.upgrade().is_none());
    assert_eq!(native.created_buffers.load(Ordering::SeqCst), 2);
    assert_eq!(native.dropped_buffers.load(Ordering::SeqCst), 2);
    assert!(native.outcome.lock().unwrap().upgrade().is_none());
}

#[test]
fn submit_stage_loss_quarantines_previous_owners_and_blocks_backend_reentry() {
    for script in [
        Script::AllocationDeviceLost,
        Script::MapDeviceLost,
        Script::FlushDeviceLost,
        Script::BeginDeviceLost,
        Script::RecordingDeviceLost,
        Script::FinishDeviceLost,
    ] {
        let (device, native) = fixture(script);
        native.fault_enabled.store(false, Ordering::SeqCst);
        let mut renderer = Renderer::new(device).unwrap();
        let (first, first_weak) = snapshot();
        let submission = submit(&mut renderer, first).unwrap();
        native.fault_enabled.store(true, Ordering::SeqCst);
        let (second, second_weak) = snapshot();
        assert!(submit(&mut renderer, second).is_err());
        let retained = matches!(
            script,
            Script::RecordingDeviceLost | Script::FinishDeviceLost
        );
        assert_eq!(renderer.pending(), if retained { 2 } else { 1 });
        assert_eq!(renderer.compilations(), 0);
        assert!(first_weak.upgrade().is_some());
        assert_eq!(second_weak.upgrade().is_some(), retained);
        assert_eq!(
            native.finishes.load(Ordering::SeqCst),
            if retained { 2 } else { 1 }
        );
        let counters = (
            native.created_buffers.load(Ordering::SeqCst),
            native.command_calls.load(Ordering::SeqCst),
            native.finishes.load(Ordering::SeqCst),
        );
        assert!(matches!(
            submit(&mut renderer, snapshot().0),
            Err(RendererError::RendererClosed)
        ));
        assert!(matches!(
            block_on(renderer.readback(&submission)),
            Err(RendererError::Rhi(Error::DeviceLost))
        ));
        assert!(matches!(
            block_on(renderer.shutdown()),
            Err(RendererError::Rhi(Error::DeviceLost))
        ));
        assert_eq!(
            counters,
            (
                native.created_buffers.load(Ordering::SeqCst),
                native.command_calls.load(Ordering::SeqCst),
                native.finishes.load(Ordering::SeqCst)
            )
        );
        assert!(native.waited_completions.lock().unwrap().is_empty());
        assert_eq!(native.idle_waits.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn nonaccepted_outcomes_without_recording_error_retain_until_successful_idle() {
    for (script, status) in [
        (Script::PartialAccepted, ExecutionStatus::PartiallyAccepted),
        (Script::PartialIssued, ExecutionStatus::Failed),
        (Script::Indeterminate, ExecutionStatus::Indeterminate),
    ] {
        let (device, native) = fixture(script);
        let mut renderer = Renderer::new(device).unwrap();
        let (source, weak) = snapshot();
        assert!(
            matches!(submit(&mut renderer, source), Err(RendererError::RetainedExecution { status: actual, .. }) if actual == status)
        );
        assert_eq!(native.finishes.load(Ordering::SeqCst), 1);
        assert_eq!(renderer.pending(), 1);
        assert!(weak.upgrade().is_some());
        assert!(native.outcome.lock().unwrap().upgrade().is_some());
        let pending = renderer.slots[0].as_ref().unwrap();
        assert_eq!(pending.outcome.facts().status, status);
        if matches!(script, Script::PartialIssued) {
            assert_eq!(
                pending.outcome.facts().failed_step.unwrap().effect,
                StepEffect::Partial
            );
        }
        let report = block_on(renderer.shutdown()).unwrap();
        assert_eq!(report.retired_submissions, 1);
        assert_eq!(native.idle_waits.load(Ordering::SeqCst), 1);
        assert!(weak.upgrade().is_none());
        assert!(native.outcome.lock().unwrap().upgrade().is_none());
        assert_eq!(native.dropped_buffers.load(Ordering::SeqCst), 2);
    }
}
#[test]
fn contradictory_accepted_outcome_with_error_is_observable_and_retained() {
    let (device, native) = fixture(Script::AcceptedWithError);
    let mut renderer = Renderer::new(device).unwrap();
    let (source, weak) = snapshot();
    assert!(matches!(
        submit(&mut renderer, source),
        Err(RendererError::RetainedExecution {
            status: ExecutionStatus::Accepted,
            execution_error: Some(Error::Backend(_)),
            ..
        })
    ));
    assert_eq!(renderer.pending(), 1);
    assert!(weak.upgrade().is_some());
    assert!(native.outcome.lock().unwrap().upgrade().is_some());
    block_on(renderer.shutdown()).unwrap();
    assert!(weak.upgrade().is_none());
}

#[test]
fn copy_limit_refusal_happens_before_plan_allocation_mapping_or_recording() {
    let (device, native) = fixture_with_caps(Script::AcceptedExact, |caps| {
        caps.limits.max_buffer_bytes = 3;
    });
    let mut renderer = Renderer::new(device).unwrap();
    assert!(matches!(
        submit(&mut renderer, snapshot_bytes(vec![1, 2, 3, 4]).0),
        Err(RendererError::Rhi(Error::InvalidRange))
    ));
    assert_eq!(renderer.compilations(), 0);
    assert_eq!(native.created_buffers.load(Ordering::SeqCst), 0);
    assert_eq!(native.command_calls.load(Ordering::SeqCst), 0);
}

#[test]
fn compute_storage_capability_is_rejected_before_fixture_resources_exist() {
    for case in [
        "storage-buffers",
        "buffer-bytes",
        "bind-groups",
        "bindings-per-group",
        "workgroups-x",
        "workgroups-y",
        "workgroups-z",
        "workgroup-size-x",
        "workgroup-size-y",
        "workgroup-size-z",
    ] {
        let (device, native) = fixture_with_caps(Script::AcceptedExact, |caps| {
            caps.storage_buffers = true;
            caps.limits.max_compute_workgroups = [1, 1, 1];
            caps.limits.max_compute_workgroup_size = [64, 1, 1];
            match case {
                "storage-buffers" => caps.storage_buffers = false,
                "buffer-bytes" => caps.limits.max_buffer_bytes = 3,
                "bind-groups" => caps.limits.max_bind_groups = 0,
                "bindings-per-group" => caps.limits.max_bindings_per_group = 0,
                "workgroups-x" => caps.limits.max_compute_workgroups[0] = 0,
                "workgroups-y" => caps.limits.max_compute_workgroups[1] = 0,
                "workgroups-z" => caps.limits.max_compute_workgroups[2] = 0,
                "workgroup-size-x" => caps.limits.max_compute_workgroup_size[0] = 63,
                "workgroup-size-y" => caps.limits.max_compute_workgroup_size[1] = 0,
                "workgroup-size-z" => caps.limits.max_compute_workgroup_size[2] = 0,
                _ => unreachable!(),
            }
        });
        assert!(
            validate_compute_requirements(&device, 4, 1).is_err(),
            "{case}"
        );
        assert_eq!(native.created_buffers.load(Ordering::SeqCst), 0, "{case}");
        assert_eq!(native.command_calls.load(Ordering::SeqCst), 0, "{case}");
    }
}

#[path = "read_set_raster.rs"]
mod read_set_raster_tests;
