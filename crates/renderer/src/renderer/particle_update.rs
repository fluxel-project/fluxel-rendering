//! Allocation and frame ownership for Babylon-style GPU particle ping-pong.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use fluxel_rhi::{
    api::{
        AccessMask, BindGroup, BindGroupDesc, BindGroupEntry, BindGroupLayout, BindGroupLayoutDesc,
        BindGroupLayoutEntry, BindingCount, BindingResource, BindingType, Buffer, BufferDesc,
        BufferUsage, BufferUses, ComputePipeline, ComputePipelineDesc, Device, DeviceId, Error,
        MemoryClass, PipelineLayoutDesc, PipelineStage, ShaderDesc, ShaderEntryPoint, ShaderFormat,
        ShaderRoute, ShaderStage, StageMask, StorageAccess, StorageBufferView, Support,
    },
    backend::{
        ExecutionBackend, MappingBackend, PipelineBackend, ReadbackBackend, ResourceBackend,
        ShaderBackend,
    },
};
use fluxel_shader::{
    BindingDeclaration, BindingKind, EntryPoint, ShaderComposer, ShaderCompositionRequest,
    ShaderStage as LogicalShaderStage, SourceUnit,
};

use crate::{
    RenderId,
    feature::particles::{
        GPU_PARTICLE_STATE_STRIDE, GpuParticle, GpuParticleUpdate, GpuParticleView,
        PARTICLE_STATE_WGSL, PARTICLE_UPDATE_WGSL,
    },
    graph::ResourceInstanceId,
};

use super::{
    Renderer, RendererError, Submission,
    admission::{self, PipelineCapabilityFacts},
};

pub(crate) const PARTICLE_PARAMETER_BYTES: u64 = 32;
const WORKGROUP_SIZE: u32 = 64;
const COMPUTE_ENTRY: &str = "fluxel_particle_update";

const EMPTY: BufferUses = BufferUses {
    stages: StageMask::EMPTY,
    accesses: AccessMask::EMPTY,
};
const HOST_WRITE: BufferUses = BufferUses {
    stages: StageMask::HOST,
    accesses: AccessMask::HOST_WRITE,
};
const COMPUTE_READ: BufferUses = BufferUses {
    stages: StageMask::COMPUTE,
    accesses: AccessMask::SHADER_READ,
};
const VERTEX_READ: BufferUses = BufferUses {
    stages: StageMask::VERTEX_INPUT,
    accesses: AccessMask::VERTEX_READ,
};

/// Persistent GPU state for one externally seeded, fixed-count particle system.
///
/// A submission retains a reference to both buffers. The next update may use
/// them after the previous submission has retired from this renderer.
pub struct GpuParticleSystem<B: ResourceBackend> {
    renderer: Arc<()>,
    device: DeviceId,
    states: Arc<[ParticleBuffer<B>; 2]>,
    kernel: Arc<ParticleKernel<B>>,
    seeds: Option<Vec<u8>>,
    current: Option<usize>,
    submitted: Option<Submission>,
    recording_attempt: Option<Arc<AtomicBool>>,
    count: u32,
}

/// Selects a scene record and its GPU particle update for this submission.
///
/// The record supplies the quad, material, visibility and transparent sort
/// position. Particle positions are already in world space. The selected view
/// provides projection, and `camera` supplies the matching billboard camera.
pub struct GpuParticleDraw<'a, B: ResourceBackend> {
    /// Scene record used to resolve and order the particle material draw.
    pub record: RenderId,
    /// Persistent particle buffers advanced by the accepted frame.
    pub system: &'a mut GpuParticleSystem<B>,
    /// Values passed to the compute update.
    pub update: GpuParticleUpdate,
    /// View-space rotation and fade parameters.
    pub camera: GpuParticleView,
}

impl<B: ResourceBackend> GpuParticleSystem<B> {
    /// Returns the initialized particle count and maximum instance footprint.
    pub fn count(&self) -> u32 {
        self.count
    }

    pub(super) fn track_recording(&mut self, marker: Arc<AtomicBool>) {
        self.recording_attempt = Some(marker);
    }

    /// Advances the ping-pong front after accepted execution has a checkpoint.
    pub(super) fn finish(&mut self, submission: Option<&Submission>) {
        if let Some(submission) = submission {
            self.current = Some(self.current.unwrap_or(0) ^ 1);
            self.submitted = Some(submission.clone());
            self.seeds = None;
            self.recording_attempt = None;
        }
    }
}

pub(crate) struct ParticleBuffer<B: ResourceBackend> {
    pub(crate) buffer: Buffer<B>,
    pub(crate) desc: BufferDesc,
    pub(crate) incoming: BufferUses,
    pub(crate) instance: ResourceInstanceId,
}

struct ParticleKernel<B: ResourceBackend> {
    pipeline: ComputePipeline<B>,
    layout: BindGroupLayout<B>,
}

pub(crate) struct ParticleStep<B: ResourceBackend> {
    states: Arc<[ParticleBuffer<B>; 2]>,
    kernel: Arc<ParticleKernel<B>>,
    source: usize,
    pub(crate) source_incoming: Option<BufferUses>,
    pub(crate) product_incoming: Option<BufferUses>,
    pub(crate) upload: ParticleBuffer<B>,
    pub(crate) uniform: ParticleBuffer<B>,
    pub(crate) upload_bytes: Vec<u8>,
    pub(crate) group: BindGroup<B>,
    pub(crate) groups: u32,
    use_marker: Arc<AtomicBool>,
}

impl<B: ResourceBackend> ParticleStep<B> {
    pub(crate) fn source(&self) -> &ParticleBuffer<B> {
        &self.states[self.source]
    }

    pub(crate) fn product(&self) -> &ParticleBuffer<B> {
        &self.states[self.source ^ 1]
    }

    pub(crate) fn pipeline(&self) -> &ComputePipeline<B> {
        &self.kernel.pipeline
    }

    pub(in crate::renderer) fn use_marker(&self) -> Arc<AtomicBool> {
        self.use_marker.clone()
    }

    pub(in crate::renderer) fn mark_recording(&self) {
        self.use_marker.store(true, Ordering::Release);
    }
}

impl<B> Renderer<B>
where
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + PipelineBackend
        + ShaderBackend,
{
    /// Creates a GPU particle system from finite, world-space seed states.
    ///
    /// The system uses a fixed particle count. The first submitted update
    /// initializes its source, and later submissions alternate the buffers.
    pub async fn create_gpu_particle_system(
        &mut self,
        particles: &[GpuParticle],
    ) -> Result<GpuParticleSystem<B>, RendererError> {
        self.ensure_open()?;
        let count = u32::try_from(particles.len()).map_err(|_| Error::InvalidRange)?;
        let bytes = u64::from(count) * GPU_PARTICLE_STATE_STRIDE;
        let seed_bytes = PARTICLE_PARAMETER_BYTES
            .checked_add(bytes)
            .ok_or(Error::InvalidRange)?;
        let groups = count.div_ceil(WORKGROUP_SIZE);
        let capabilities = self.device.capabilities();
        let limits = capabilities.limits;
        if count == 0 || seed_bytes > limits.max_buffer_bytes {
            return Err(Error::InvalidRange.into());
        }
        if !capabilities.compute
            || !capabilities.storage_buffers
            || self.device.compute().is_none()
            || limits.max_compute_workgroup_size[0] < WORKGROUP_SIZE
            || limits.max_compute_workgroup_size[1..].contains(&0)
            || limits.max_compute_workgroups[1..].contains(&0)
            || limits.max_compute_workgroups[0] < groups
            || limits.max_bind_groups < 1
            || limits.max_bindings_per_group < 3
        {
            return Err(Error::Unsupported("GPU particle update and vertex storage").into());
        }
        let state_desc = buffer_desc(
            "fluxel particle state",
            bytes,
            BufferUsage::COPY_DST | BufferUsage::STORAGE | BufferUsage::VERTEX,
            MemoryClass::Device,
        );
        let uniform = uniform_desc();
        let upload = upload_desc(seed_bytes);
        for desc in [&state_desc, &uniform, &upload] {
            require_buffer(&self.device, desc)?;
        }
        let seed_capacity = usize::try_from(bytes).map_err(|_| Error::InvalidRange)?;
        let mut seeds = Vec::with_capacity(seed_capacity);
        for particle in particles {
            seeds.extend_from_slice(&particle.pack().map_err(RendererError::ParticleState)?);
        }
        let kernel = create_kernel(&self.device, bytes)
            .await
            .map_err(|error| self.observe_renderer_error(error))?;
        let states = [
            allocate(&self.device, state_desc.clone(), EMPTY)
                .map_err(|error| self.observe_renderer_error(error))?,
            allocate(&self.device, state_desc, EMPTY)
                .map_err(|error| self.observe_renderer_error(error))?,
        ];
        Ok(GpuParticleSystem {
            renderer: self.owner.clone(),
            device: self.device.id(),
            states: Arc::new(states),
            kernel: Arc::new(kernel),
            seeds: Some(seeds),
            current: None,
            submitted: None,
            recording_attempt: None,
            count,
        })
    }

    pub(super) fn prepare_particle_step(
        &self,
        system: &GpuParticleSystem<B>,
        update: GpuParticleUpdate,
    ) -> Result<ParticleStep<B>, RendererError> {
        self.validate_particle_step(system, update)?;
        let params = update.pack().map_err(RendererError::ParticleUpdate)?;
        let mut upload_bytes = Vec::from(params);
        if let Some(seed) = &system.seeds {
            upload_bytes.extend_from_slice(seed);
        }
        let upload_desc =
            upload_desc(u64::try_from(upload_bytes.len()).map_err(|_| Error::InvalidRange)?);
        let uniform_desc = uniform_desc();
        let upload = allocate(&self.device, upload_desc, HOST_WRITE)?;
        let uniform = allocate(&self.device, uniform_desc, EMPTY)?;
        let source = system.current.unwrap_or(0);
        let source_buffer = &system.states[source];
        let product = &system.states[source ^ 1];
        let group = self.device.create_bind_group(&BindGroupDesc {
            label: Some("fluxel particle update states"),
            layout: &system.kernel.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer {
                        buffer: &uniform.buffer,
                        range: 0..PARTICLE_PARAMETER_BYTES,
                    },
                },
                storage_entry(1, source_buffer),
                storage_entry(2, product),
            ],
        })?;
        Ok(ParticleStep {
            states: system.states.clone(),
            kernel: system.kernel.clone(),
            source,
            source_incoming: system.current.map(|_| VERTEX_READ),
            product_incoming: system.current.map(|_| COMPUTE_READ),
            upload,
            uniform,
            upload_bytes,
            group,
            groups: system.count.div_ceil(WORKGROUP_SIZE),
            use_marker: Arc::new(AtomicBool::new(false)),
        })
    }

    pub(super) fn validate_particle_step(
        &self,
        system: &GpuParticleSystem<B>,
        update: GpuParticleUpdate,
    ) -> Result<(), RendererError> {
        if system.device != self.device.id() || !Arc::ptr_eq(&system.renderer, &self.owner) {
            return Err(Error::WrongDevice.into());
        }
        if system
            .recording_attempt
            .as_ref()
            .is_some_and(|used| used.load(Ordering::Acquire))
        {
            return Err(Error::InvalidState.into());
        }
        if system
            .submitted
            .as_ref()
            .is_some_and(|token| self.pending_for(token).is_ok())
        {
            return Err(RendererError::Backpressure);
        }
        if update.active_count != system.count {
            return Err(Error::InvalidRange.into());
        }
        update.validate().map_err(RendererError::ParticleUpdate)
    }
}

async fn create_kernel<B: ResourceBackend + PipelineBackend + ShaderBackend>(
    device: &Device<B>,
    bytes: u64,
) -> Result<ParticleKernel<B>, RendererError> {
    let request = ShaderCompositionRequest::new(vec![
        SourceUnit::new("particle-state", PARTICLE_STATE_WGSL),
        SourceUnit::new("particle-update", PARTICLE_UPDATE_WGSL),
    ])
    .with_entry_point(EntryPoint::new(COMPUTE_ENTRY, LogicalShaderStage::Compute))
    .with_binding(
        BindingDeclaration::new(0, 0, BindingKind::Uniform)
            .with_visibility([LogicalShaderStage::Compute]),
    )
    .with_binding(
        BindingDeclaration::new(0, 1, BindingKind::StorageRead)
            .with_visibility([LogicalShaderStage::Compute]),
    )
    .with_binding(
        BindingDeclaration::new(0, 2, BindingKind::StorageReadWrite)
            .with_visibility([LogicalShaderStage::Compute]),
    );
    let artifact = ShaderComposer::new()
        .compose(&request)
        .map_err(RendererError::ParticleShader)?;
    admission::admit(
        &artifact
            .requirements()
            .naga_capabilities
            .iter()
            .map(|capability| capability.name().to_owned())
            .collect::<Vec<_>>(),
        device.capabilities(),
        PipelineCapabilityFacts {
            multiview_views: None,
            push_constant_bytes: 0,
            multisample_count: 1,
            nonuniform: Default::default(),
        },
    )
    .map_err(RendererError::ShaderAdmission)?;
    let descriptor = ShaderDesc {
        label: Some("fluxel particle compute".into()),
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(artifact.source().as_bytes()),
        entries: vec![ShaderEntryPoint {
            stage: ShaderStage::Compute,
            name: COMPUTE_ENTRY.into(),
            workgroup_size: Some([WORKGROUP_SIZE, 1, 1]),
        }],
        required_features: artifact.requirements().names.iter().cloned().collect(),
    };
    if matches!(device.shader_route(&descriptor), ShaderRoute::Unsupported) {
        return Err(Error::Unsupported("GPU particle update shader route").into());
    }
    let shader = device.shader(descriptor).await?;
    let layout = device.create_bind_group_layout(&BindGroupLayoutDesc {
        label: Some("fluxel particle update states".into()),
        entries: vec![
            binding(
                0,
                BindingType::UniformBuffer {
                    dynamic_offset: false,
                    min_size: PARTICLE_PARAMETER_BYTES,
                },
            ),
            binding(
                1,
                BindingType::StorageBuffer {
                    dynamic_offset: false,
                    min_size: bytes,
                    access: StorageAccess::Read,
                },
            ),
            binding(
                2,
                BindingType::StorageBuffer {
                    dynamic_offset: false,
                    min_size: bytes,
                    access: StorageAccess::ReadWrite,
                },
            ),
        ],
    })?;
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDesc {
        label: Some("fluxel particle update layout".into()),
        groups: &[&layout],
        push_constants: &[],
    })?;
    let pipeline = device
        .compute()
        .ok_or(Error::Unsupported("GPU particle compute"))?
        .create_pipeline(ComputePipelineDesc {
            label: Some("fluxel particle update".into()),
            layout: pipeline_layout,
            compute: PipelineStage {
                module: shader,
                entry: COMPUTE_ENTRY.into(),
                constants: vec![],
            },
            required_subgroup_size: None,
            cache: None,
        })
        .await?;
    Ok(ParticleKernel { pipeline, layout })
}

fn binding(binding: u32, ty: BindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: vec![ShaderStage::Compute],
        ty,
        count: BindingCount::One,
        partially_bound: false,
        nonuniform_indexing: false,
    }
}

fn storage_entry<B: ResourceBackend>(
    binding: u32,
    owner: &ParticleBuffer<B>,
) -> BindGroupEntry<'_, B> {
    BindGroupEntry {
        binding,
        resource: BindingResource::StorageBufferView {
            buffer: &owner.buffer,
            range: 0..owner.desc.size,
            view: StorageBufferView::Raw,
        },
    }
}

fn uniform_desc() -> BufferDesc {
    buffer_desc(
        "fluxel particle update uniforms",
        PARTICLE_PARAMETER_BYTES,
        BufferUsage::UNIFORM | BufferUsage::COPY_DST,
        MemoryClass::Device,
    )
}

fn upload_desc(size: u64) -> BufferDesc {
    buffer_desc(
        "fluxel particle update upload",
        size,
        BufferUsage::MAP_WRITE | BufferUsage::COPY_SRC,
        MemoryClass::Upload,
    )
}

fn buffer_desc(label: &str, size: u64, usage: BufferUsage, memory: MemoryClass) -> BufferDesc {
    BufferDesc {
        label: Some(label.into()),
        size,
        usage,
        memory,
    }
}

fn require_buffer<B: ResourceBackend>(
    device: &Device<B>,
    desc: &BufferDesc,
) -> Result<(), RendererError> {
    if let Support::Unsupported(reason) = device.buffer_support(desc) {
        return Err(Error::Unsupported(reason).into());
    }
    Ok(())
}

fn allocate<B: ResourceBackend>(
    device: &Device<B>,
    desc: BufferDesc,
    incoming: BufferUses,
) -> Result<ParticleBuffer<B>, RendererError> {
    Ok(ParticleBuffer {
        buffer: device.create_buffer(&desc)?,
        desc,
        incoming,
        instance: ResourceInstanceId::fresh()?,
    })
}
