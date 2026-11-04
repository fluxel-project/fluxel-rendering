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

use fluxel_assets::AssetKind;
use fluxel_rhi::{api::*, backend::*};

pub(super) struct Bytes;
impl AssetKind for Bytes {}

#[derive(Clone, Copy)]
pub(super) enum Script {
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

pub(super) struct Native {
    pub(super) script: Script,
    pub(super) fault_enabled: AtomicBool,
    pub(super) finishes: AtomicUsize,
    pub(super) next_completion: AtomicUsize,
    pub(super) command_calls: AtomicUsize,
    pub(super) copied_bytes: Mutex<Vec<u64>>,
    pub(super) copy_destinations: Mutex<Vec<Range<u64>>>,
    pub(super) uniform_offsets: Mutex<Vec<u32>>,
    pub(super) texture_uploads: Mutex<Vec<(u64, u32)>>,
    pub(super) mapped_bytes: Mutex<Vec<Vec<u8>>>,
    pub(super) map_calls: AtomicUsize,
    pub(super) readback_ranges: Mutex<Vec<Range<u64>>>,
    pub(super) waits: AtomicUsize,
    pub(super) idle_waits: AtomicUsize,
    pub(super) waited_completions: Mutex<Vec<usize>>,
    pub(super) created_buffers: AtomicUsize,
    pub(super) dropped_buffers: AtomicUsize,
    pub(super) outcome: Mutex<Weak<OutcomeGuard>>,
}

pub(super) struct NativeBuffer {
    native: Arc<Native>,
}

pub(super) struct NativeTexture;

impl Drop for NativeBuffer {
    fn drop(&mut self) {
        self.native.dropped_buffers.fetch_add(1, Ordering::SeqCst);
    }
}

pub(super) struct OutcomeGuard;

pub(super) struct CompletionMarker {
    pub(super) id: usize,
}

pub(super) struct WitnessBackend;
#[derive(Clone)]
pub(super) struct WitnessImage {
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
pub(super) struct WitnessSurface {
    pub(super) generation: Arc<std::sync::atomic::AtomicU64>,
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

pub(super) struct WitnessMap {
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
        device.map_calls.fetch_add(1, Ordering::SeqCst);
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

pub(super) struct WitnessWait {
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
