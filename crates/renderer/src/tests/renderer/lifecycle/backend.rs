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

mod commands;
mod compute;

pub(super) struct Bytes;
impl AssetKind for Bytes {}

#[derive(Clone, Copy)]
pub(super) enum Script {
    CleanRejected,
    AllocationDeviceLost,
    SamplerBindGroupRejected,
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
    SurfacePresentRejectedAfterScopes,
    SurfacePresentDeviceLost,
}

pub(super) struct Native {
    pub(super) script: Script,
    pub(super) fault_enabled: AtomicBool,
    pub(super) texture_filterable: AtomicBool,
    pub(super) depth_supported: AtomicBool,
    pub(super) blit_supported: AtomicBool,
    pub(super) texture_max_mips: AtomicUsize,
    pub(super) blits: Mutex<Vec<(u32, u32, [u32; 3], [u32; 3], FilterMode)>>,
    pub(super) transitions: AtomicUsize,
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
    pub(super) created_textures: AtomicUsize,
    pub(super) sampler_anisotropy_limit: AtomicUsize,
    pub(super) bound_sampler_descs: Mutex<Vec<(u32, SamplerDesc)>>,
    pub(super) created_samplers: AtomicUsize,
    pub(super) dropped_samplers: AtomicUsize,
    pub(super) pipeline_color_formats: Mutex<Vec<Vec<TextureFormat>>>,
    pub(super) pipeline_depth_states: Mutex<Vec<Option<(TextureFormat, bool, CompareFunction)>>>,
    pub(super) dropped_buffers: AtomicUsize,
    pub(super) outcome: Mutex<Weak<OutcomeGuard>>,
}

pub(super) struct NativeBuffer {
    native: Arc<Native>,
}

pub(super) struct NativeTexture;

pub(super) struct NativeSampler {
    native: Arc<Native>,
    desc: SamplerDesc,
}

impl Drop for NativeBuffer {
    fn drop(&mut self) {
        self.native.dropped_buffers.fetch_add(1, Ordering::SeqCst);
    }
}

impl Drop for NativeSampler {
    fn drop(&mut self) {
        self.native.dropped_samplers.fetch_add(1, Ordering::SeqCst);
    }
}

pub(super) struct OutcomeGuard;

pub(super) struct CompletionMarker {
    pub(super) id: usize,
}

pub(super) struct WitnessBackend;
pub(super) struct WitnessCompute;
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
    type NativeSampler = Arc<NativeSampler>;
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
    type NativeEncoder = Arc<Native>;
    type NativeCompletion = Arc<CompletionMarker>;
    type NativeSurface = WitnessSurface;
    type NativeSurfaceImage = WitnessImage;
    type NativeUncertainty = ();
    type MemoryObservation = Unavailable;
    type Compute = WitnessCompute;
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
    fn texture_support(device: &Arc<Native>, desc: &TextureDesc) -> Support<TextureSupport> {
        if desc.format == TextureFormat::Depth32Float
            && !device.depth_supported.load(Ordering::SeqCst)
        {
            return Support::Unsupported("witness depth attachment");
        }
        Support::Supported(TextureSupport {
            max_extent: [u32::MAX; 3],
            max_array_layers: u32::MAX,
            max_mips: device.texture_max_mips.load(Ordering::SeqCst) as u32,
            sample_counts: vec![1],
            compatible_view_formats: Vec::new(),
            filterable: device.texture_filterable.load(Ordering::SeqCst),
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
    fn create_texture(device: &Arc<Native>, _: &TextureDesc) -> Result<Arc<NativeTexture>> {
        device.created_textures.fetch_add(1, Ordering::SeqCst);
        Ok(Arc::new(NativeTexture))
    }
    fn create_view(_: &Arc<Native>, _: &Arc<NativeTexture>, _: &TextureViewDesc) -> Result<()> {
        Ok(())
    }
    fn create_sampler(device: &Arc<Native>, desc: &SamplerDesc) -> Result<Arc<NativeSampler>> {
        if usize::from(desc.anisotropy) > device.sampler_anisotropy_limit.load(Ordering::SeqCst) {
            return Err(Error::Unsupported("witness sampler anisotropy"));
        }
        device.created_samplers.fetch_add(1, Ordering::SeqCst);
        Ok(Arc::new(NativeSampler {
            native: Arc::clone(device),
            desc: desc.clone(),
        }))
    }
    fn create_bind_group_layout(_: &Arc<Native>, _: &BindGroupLayoutDesc) -> Result<()> {
        Ok(())
    }
    fn create_bind_group(
        device: &Arc<Native>,
        desc: &BindGroupDesc<'_, Self>,
    ) -> Result<Arc<Native>> {
        if matches!(device.script, Script::SamplerBindGroupRejected)
            && desc.entries.iter().any(|entry| {
                entry.binding == 2 && matches!(entry.resource, BindingResource::Sampler(_))
            })
        {
            return Err(Error::InvalidState);
        }
        for entry in desc.entries {
            if let BindingResource::Sampler(sampler) = entry.resource {
                // SAFETY: Core admits only same-device resources before the
                // witness backend receives this concrete bind group.
                let actual = unsafe { sampler.native_ref() };
                device
                    .bound_sampler_descs
                    .lock()
                    .unwrap()
                    .push((entry.binding, actual.desc.clone()));
            }
        }
        Ok(Arc::clone(device))
    }
    fn create_pipeline_layout(_: &Arc<Native>, _: &PipelineLayoutDesc<'_, Self>) -> Result<()> {
        Ok(())
    }
}

impl ShaderBackend for WitnessBackend {
    type DirectFuture<'a> = Ready<Result<()>>;
    type NagaFuture<'a> = Ready<Result<()>>;

    fn shader_route(_: &Arc<Native>, _: &ShaderDesc) -> ShaderRoute {
        ShaderRoute::Direct(ShaderTarget {
            format: ShaderFormat::Wgsl,
            environment: "witness".into(),
            abi_key: "witness".into(),
        })
    }

    fn compile_direct<'a>(
        _: &'a Arc<Native>,
        _: &'a ShaderDesc,
        _: &'a ShaderTarget,
    ) -> Self::DirectFuture<'a> {
        ready(Ok(()))
    }

    fn compile_naga<'a>(
        _: &'a Arc<Native>,
        _: &'a ShaderDesc,
        _: &'a ShaderTarget,
    ) -> Self::NagaFuture<'a> {
        ready(Ok(()))
    }
}

impl PipelineBackend for WitnessBackend {
    type RasterPipelineFuture<'a> = Ready<Result<()>>;

    fn create_raster_pipeline<'a>(
        device: &'a Arc<Native>,
        descriptor: &'a RasterPipelineDesc<Self>,
        _: Option<DynamicDepthBias>,
    ) -> Self::RasterPipelineFuture<'a> {
        device.pipeline_color_formats.lock().unwrap().push(
            descriptor
                .color_targets
                .iter()
                .flatten()
                .map(|target| target.format)
                .collect(),
        );
        device.pipeline_depth_states.lock().unwrap().push(
            descriptor
                .depth_stencil
                .as_ref()
                .map(|state| (state.format, state.depth_write_enabled, state.depth_compare)),
        );
        ready(Ok(()))
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

impl RasterBackend for WitnessBackend {
    fn begin_raster(_: &mut Arc<Native>, _: &RasterPassDesc<'_, Self>) -> Result<()> {
        Ok(())
    }
    fn set_pipeline(_: &mut Arc<Native>, _: &()) -> Result<()> {
        Ok(())
    }
    fn set_bind_group(
        _: &mut Arc<Native>,
        _: u32,
        group: &Arc<Native>,
        offsets: &[u32],
    ) -> Result<()> {
        group
            .uniform_offsets
            .lock()
            .unwrap()
            .extend_from_slice(offsets);
        Ok(())
    }
    fn set_vertex_buffer(
        _: &mut Arc<Native>,
        _: u32,
        _: &Arc<NativeBuffer>,
        _: u64,
        _: u64,
    ) -> Result<()> {
        Ok(())
    }
    fn set_index_buffer(
        _: &mut Arc<Native>,
        _: &Arc<NativeBuffer>,
        _: u64,
        _: u64,
        _: IndexFormat,
    ) -> Result<()> {
        Ok(())
    }
    fn viewport(_: &mut Arc<Native>, _: Viewport) -> Result<()> {
        Ok(())
    }
    fn scissor(_: &mut Arc<Native>, _: Rect) -> Result<()> {
        Ok(())
    }
    fn blend_constant(_: &mut Arc<Native>, _: [f32; 4]) -> Result<()> {
        Ok(())
    }
    fn stencil_reference(_: &mut Arc<Native>, _: u32) -> Result<()> {
        Ok(())
    }
    fn draw(_: &mut Arc<Native>, _: Range<u32>, _: Range<u32>) -> Result<()> {
        Ok(())
    }
    fn draw_indexed(_: &mut Arc<Native>, _: Range<u32>, _: i32, _: Range<u32>) -> Result<()> {
        Ok(())
    }
    fn end_raster(_: &mut Arc<Native>) -> Result<()> {
        Ok(())
    }
    fn abort_raster(_: &mut Arc<Native>) {}
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
            || (!self.idle
                && self.native.fault_enabled.load(Ordering::SeqCst)
                && matches!(self.native.script, Script::AcceptedCompletionDeviceLost))
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
        _: &mut Arc<Native>,
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
