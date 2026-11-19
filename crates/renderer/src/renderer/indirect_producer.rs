//! Shared one-record GPU indirect-draw producer.
//!
//! The producer owns a uniform upload, its device-local uniform copy, and the
//! generated non-indexed indirect argument record. Its caller declares the
//! copy, compute write, and later raster indirect read on the same owners.

use std::{collections::HashMap, ops::Range, sync::Arc};

use fluxel_rhi::{
    api::{
        AccessMask, BindGroup, BindGroupDesc, BindGroupEntry, BindGroupLayoutDesc,
        BindGroupLayoutEntry, BindingCount, BindingResource, BindingType, Buffer, BufferDesc,
        BufferUsage, BufferUses, ComputePipeline, ComputePipelineDesc, Device, Error, MemoryClass,
        PipelineLayoutDesc, PipelineStage, ShaderDesc, ShaderEntryPoint, ShaderFormat, ShaderRoute,
        ShaderStage, StageMask, StorageAccess, StorageBufferView, Support,
    },
    backend::{PipelineBackend, ResourceBackend, ShaderBackend},
};
use fluxel_shader::{
    BindingAccess, BindingDeclaration, BindingKind, CallableDeclaration, EntryPoint, ScalarType,
    ShaderComposer, ShaderCompositionRequest, ShaderStage as LogicalShaderStage, SourceUnit,
    ValueType,
};

use crate::{
    ForwardDrawInput, ForwardResources, FrameSnapshot, GpuIndirectDraw,
    graph::ResourceInstanceId,
    renderer::{
        RendererError,
        admission::{self, PipelineCapabilityFacts},
    },
};

const INDIRECT_ARGUMENT_BYTES: u64 = 16;
const PARAMETER_BYTES: u64 = 16;

/// Handle-free facts for a single non-indexed indirect argument record.
pub(super) struct PreparedIndirect {
    upload_desc: BufferDesc,
    parameter_desc: BufferDesc,
    argument_desc: BufferDesc,
    upload_bytes: [u8; PARAMETER_BYTES as usize],
    descriptor: ShaderDesc,
    entry: String,
    vertex_count: u32,
    first_vertex: u32,
    instance_capacity: u32,
}

/// One concrete producer buffer and the Graph facts that bind it.
pub(crate) struct IndirectBuffer<B: ResourceBackend> {
    /// Native buffer owner retained through submission completion.
    pub(crate) buffer: Buffer<B>,
    /// Descriptor used for allocation and Graph binding validation.
    pub(crate) desc: BufferDesc,
    /// State immediately before this frame's first declared access.
    pub(crate) incoming: BufferUses,
    /// Stable physical-owner identity for the compiled Graph binding.
    pub(crate) instance: ResourceInstanceId,
}

/// Completion-owned buffers and compute objects for one generated draw record.
pub(crate) struct IndirectProduct<B: ResourceBackend> {
    /// Host-visible staging data copied into `parameters` by the Graph.
    pub(crate) upload: IndirectBuffer<B>,
    /// Device-local uniform parameters consumed by the producer compute pass.
    pub(crate) parameters: IndirectBuffer<B>,
    /// One four-word non-indexed indirect record written by compute.
    pub(crate) arguments: IndirectBuffer<B>,
    /// Retained compute pipeline used by the producer Graph pass.
    pub(crate) pipeline: ComputePipeline<B>,
    /// Retained binding group for parameters and the argument record.
    pub(crate) group: BindGroup<B>,
    /// Bytes written into `upload` before the copy pass.
    pub(crate) upload_bytes: [u8; PARAMETER_BYTES as usize],
    /// Fixed vertex count placed in word zero of the indirect record.
    vertex_count: u32,
    /// First vertex placed in word two of the indirect record.
    first_vertex: u32,
    /// Upper bound supplied to `visible_instance_count`.
    instance_capacity: u32,
}

impl<B: ResourceBackend> IndirectProduct<B> {
    /// Maps the producer's immutable command bounds to the raster draw's footprint.
    pub(crate) fn validate_draw(
        &self,
        vertices: &Range<u32>,
        instances: &Range<u32>,
    ) -> fluxel_rhi::api::Result<()> {
        if vertices.start != self.first_vertex
            || vertices.end.checked_sub(vertices.start) != Some(self.vertex_count)
            || *instances != (0..self.instance_capacity)
        {
            return Err(Error::InvalidRange);
        }
        Ok(())
    }
}

/// Validates request ownership and plans the producers of the selected draws.
pub(super) fn plan_draws<B: ResourceBackend + ShaderBackend>(
    device: &Device<B>,
    snapshot: &FrameSnapshot<ForwardResources>,
    inputs: &[ForwardDrawInput],
    requests: &[GpuIndirectDraw],
) -> Result<Vec<(usize, PreparedIndirect)>, RendererError> {
    if requests.is_empty() {
        return Ok(Vec::new());
    }
    let mut by_draw = HashMap::with_capacity(requests.len());
    for request in requests {
        let draw = request.draw();
        snapshot
            .record(draw)
            .map_err(RendererError::IndirectScene)?;
        if by_draw.insert(draw, request).is_some() {
            return Err(RendererError::DuplicateIndirectDraw(draw));
        }
    }
    inputs
        .iter()
        .enumerate()
        .filter_map(|(index, input)| Some((index, input, by_draw.get(&input.record()?)?)))
        .map(|(index, input, request)| {
            plan(
                device,
                input.geometry().value().vertices.clone(),
                input.instance_range(),
                request.source(),
                request.parameters(),
            )
            .map(|prepared| (index, prepared))
        })
        .collect()
}

fn plan<B: ResourceBackend + ShaderBackend>(
    device: &Device<B>,
    vertices: Range<u32>,
    instances: Range<u32>,
    source: &SourceUnit,
    params: [u32; 4],
) -> Result<PreparedIndirect, RendererError> {
    let vertex_count = bounded_count(&vertices)?;
    let instance_capacity = bounded_count(&instances)?;
    if instances.start != 0 {
        return Err(Error::Unsupported(
            "GPU indirect producer requires a zero-based instance range",
        )
        .into());
    }
    let capabilities = device.capabilities();
    let limits = capabilities.limits;
    if !capabilities.compute
        || !capabilities.storage_buffers
        || !capabilities.indirect_draw
        || device.compute().is_none()
    {
        return Err(Error::Unsupported(
            "GPU indirect producer requires compute storage and indirect draw",
        )
        .into());
    }
    if limits.max_compute_workgroup_size.contains(&0)
        || limits.max_compute_workgroups.contains(&0)
        || limits.max_bind_groups < 1
        || limits.max_bindings_per_group < 2
        || INDIRECT_ARGUMENT_BYTES > limits.max_buffer_bytes
        || PARAMETER_BYTES > limits.max_buffer_bytes
    {
        return Err(Error::Unsupported("GPU indirect producer workgroup or buffer limit").into());
    }
    let artifact = compose(source, vertex_count, vertices.start, instance_capacity)?;
    let entry = artifact
        .entry_point(LogicalShaderStage::Compute)
        .ok_or(Error::InvalidState)?
        .name
        .clone();
    let descriptor = ShaderDesc {
        label: Some("Fluxel indirect producer program".into()),
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(artifact.source().as_bytes()),
        entries: vec![ShaderEntryPoint {
            name: entry.clone(),
            stage: ShaderStage::Compute,
            workgroup_size: Some([1, 1, 1]),
        }],
        required_features: artifact.requirements().names.iter().cloned().collect(),
    };
    admit(
        device,
        &descriptor,
        artifact
            .requirements()
            .naga_capabilities
            .iter()
            .map(|capability| capability.name()),
    )?;
    let prepared = PreparedIndirect {
        upload_desc: buffer_desc(
            "fluxel indirect producer upload",
            PARAMETER_BYTES,
            BufferUsage::COPY_SRC | BufferUsage::MAP_WRITE,
            MemoryClass::Upload,
        ),
        parameter_desc: buffer_desc(
            "fluxel indirect producer parameters",
            PARAMETER_BYTES,
            BufferUsage::COPY_DST | BufferUsage::UNIFORM,
            MemoryClass::Device,
        ),
        argument_desc: buffer_desc(
            "fluxel indirect producer arguments",
            INDIRECT_ARGUMENT_BYTES,
            BufferUsage::STORAGE | BufferUsage::INDIRECT,
            MemoryClass::Device,
        ),
        upload_bytes: encode_parameters(params),
        descriptor,
        entry,
        vertex_count,
        first_vertex: vertices.start,
        instance_capacity,
    };
    for desc in [
        &prepared.upload_desc,
        &prepared.parameter_desc,
        &prepared.argument_desc,
    ] {
        if let Support::Unsupported(reason) = device.buffer_support(desc) {
            return Err(Error::Unsupported(reason).into());
        }
    }
    Ok(prepared)
}

/// Realizes admitted producers in the retained draw order.
pub(super) async fn realize_draws<B: ResourceBackend + ShaderBackend + PipelineBackend>(
    device: &Device<B>,
    planned: Vec<(usize, PreparedIndirect)>,
) -> Result<Vec<(usize, IndirectProduct<B>)>, RendererError> {
    let mut products = Vec::with_capacity(planned.len());
    for (draw, producer) in planned {
        products.push((draw, realize(device, producer).await?));
    }
    Ok(products)
}

/// Realizes the admitted compute objects and their completion-owned buffers.
pub(super) async fn realize<B: ResourceBackend + ShaderBackend + PipelineBackend>(
    device: &Device<B>,
    prepared: PreparedIndirect,
) -> Result<IndirectProduct<B>, RendererError> {
    let shader = device.shader(prepared.descriptor).await?;
    let layout = device.create_bind_group_layout(&BindGroupLayoutDesc {
        label: Some("Fluxel indirect producer bindings".into()),
        entries: vec![
            uniform(0, PARAMETER_BYTES),
            storage(1, INDIRECT_ARGUMENT_BYTES),
        ],
    })?;
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDesc {
        label: Some("Fluxel indirect producer pipeline layout"),
        groups: &[&layout],
        push_constants: &[],
    })?;
    let pipeline = device
        .compute()
        .ok_or(Error::Unsupported("compute"))?
        .create_pipeline(ComputePipelineDesc {
            label: Some("Fluxel indirect producer pipeline".into()),
            layout: pipeline_layout,
            compute: PipelineStage {
                module: shader,
                entry: prepared.entry,
                constants: vec![],
            },
            required_subgroup_size: None,
            cache: None,
        })
        .await?;
    let upload = allocate(device, prepared.upload_desc)?;
    let parameters = allocate(device, prepared.parameter_desc)?;
    let arguments = allocate(device, prepared.argument_desc)?;
    let group = device.create_bind_group(&BindGroupDesc {
        label: Some("Fluxel indirect producer bindings"),
        layout: &layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer {
                    buffer: &parameters.buffer,
                    range: 0..PARAMETER_BYTES,
                },
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::StorageBufferView {
                    buffer: &arguments.buffer,
                    range: 0..INDIRECT_ARGUMENT_BYTES,
                    view: StorageBufferView::Raw,
                },
            },
        ],
    })?;
    Ok(IndirectProduct {
        upload,
        parameters,
        arguments,
        pipeline,
        group,
        upload_bytes: prepared.upload_bytes,
        vertex_count: prepared.vertex_count,
        first_vertex: prepared.first_vertex,
        instance_capacity: prepared.instance_capacity,
    })
}

fn bounded_count(range: &Range<u32>) -> Result<u32, RendererError> {
    let count = range
        .end
        .checked_sub(range.start)
        .ok_or(Error::InvalidRange)?;
    if count == 0 {
        return Err(Error::InvalidRange.into());
    }
    Ok(count)
}

fn compose(
    source: &SourceUnit,
    vertex_count: u32,
    first_vertex: u32,
    instance_capacity: u32,
) -> Result<Arc<fluxel_shader::ShaderArtifact>, RendererError> {
    let template = TEMPLATE
        .replace("INDIRECT_VERTEX_COUNT", &format!("{vertex_count}u"))
        .replace("INDIRECT_FIRST_VERTEX", &format!("{first_vertex}u"))
        .replace(
            "INDIRECT_INSTANCE_CAPACITY",
            &format!("{instance_capacity}u"),
        );
    let request = ShaderCompositionRequest::new(vec![
        source.clone(),
        SourceUnit::new("indirect-producer-template", template),
    ])
    .with_entry_point(EntryPoint::new(
        "fluxel_indirect_producer_cs",
        LogicalShaderStage::Compute,
    ))
    .with_callable(CallableDeclaration {
        name: "visible_instance_count".into(),
        arguments: vec![
            ValueType::Vec4(ScalarType::U32),
            ValueType::Scalar(ScalarType::U32),
        ],
        result: Some(ValueType::Scalar(ScalarType::U32)),
    })
    .with_binding(
        BindingDeclaration::new(0, 0, BindingKind::Uniform)
            .with_visibility([LogicalShaderStage::Compute])
            .with_access(BindingAccess {
                read: true,
                write: false,
                query: false,
                atomic: false,
            }),
    )
    .with_binding(
        BindingDeclaration::new(0, 1, BindingKind::StorageReadWrite)
            .with_visibility([LogicalShaderStage::Compute])
            .with_access(BindingAccess {
                read: false,
                write: true,
                query: false,
                atomic: false,
            }),
    );
    ShaderComposer::new()
        .compose(&request)
        .map_err(RendererError::IndirectShader)
}

fn admit<B: ResourceBackend + ShaderBackend>(
    device: &Device<B>,
    descriptor: &ShaderDesc,
    demands: impl IntoIterator<Item = &'static str>,
) -> Result<(), RendererError> {
    admission::admit(
        &demands.into_iter().map(str::to_owned).collect::<Vec<_>>(),
        device.capabilities(),
        PipelineCapabilityFacts {
            multiview_views: None,
            push_constant_bytes: 0,
            multisample_count: 1,
            nonuniform: Default::default(),
        },
    )
    .map_err(RendererError::ShaderAdmission)?;
    if matches!(device.shader_route(descriptor), ShaderRoute::Unsupported) {
        return Err(Error::Unsupported("GPU indirect producer shader route").into());
    }
    Ok(())
}

fn allocate<B: ResourceBackend>(
    device: &Device<B>,
    desc: BufferDesc,
) -> Result<IndirectBuffer<B>, RendererError> {
    let incoming = if desc.usage.contains(BufferUsage::MAP_WRITE) {
        BufferUses {
            stages: StageMask::HOST,
            accesses: AccessMask::HOST_WRITE,
        }
    } else {
        BufferUses {
            stages: StageMask::EMPTY,
            accesses: AccessMask::EMPTY,
        }
    };
    Ok(IndirectBuffer {
        buffer: device.create_buffer(&desc)?,
        desc,
        incoming,
        instance: ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?,
    })
}

fn buffer_desc(label: &str, size: u64, usage: BufferUsage, memory: MemoryClass) -> BufferDesc {
    BufferDesc {
        label: Some(label.into()),
        size,
        usage,
        memory,
    }
}

fn uniform(binding: u32, min_size: u64) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: vec![ShaderStage::Compute],
        ty: BindingType::UniformBuffer {
            dynamic_offset: false,
            min_size,
        },
        count: BindingCount::One,
        partially_bound: false,
        nonuniform_indexing: false,
    }
}

fn storage(binding: u32, min_size: u64) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: vec![ShaderStage::Compute],
        ty: BindingType::StorageBuffer {
            dynamic_offset: false,
            min_size,
            access: StorageAccess::ReadWrite,
        },
        count: BindingCount::One,
        partially_bound: false,
        nonuniform_indexing: false,
    }
}

fn encode_parameters(parameters: [u32; 4]) -> [u8; PARAMETER_BYTES as usize] {
    let mut bytes = [0; PARAMETER_BYTES as usize];
    for (index, value) in parameters.into_iter().enumerate() {
        let start = index * 4;
        bytes[start..start + 4].copy_from_slice(&value.to_le_bytes());
    }
    bytes
}

const TEMPLATE: &str = r#"
@group(0) @binding(0) var<uniform> indirect_parameters: vec4<u32>;
@group(0) @binding(1) var<storage, read_write> indirect_arguments: array<u32>;

@compute @workgroup_size(1)
fn fluxel_indirect_producer_cs() {
    let capacity = INDIRECT_INSTANCE_CAPACITY;
    let visible = min(visible_instance_count(indirect_parameters, capacity), capacity);
    indirect_arguments[0u] = INDIRECT_VERTEX_COUNT;
    indirect_arguments[1u] = visible;
    indirect_arguments[2u] = INDIRECT_FIRST_VERTEX;
    indirect_arguments[3u] = 0u;
}
"#;
