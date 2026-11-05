//! Private position-deformation planning and compute realization.

use std::sync::Arc;

use fluxel_rhi::{
    api::{
        BindGroupDesc, BindGroupEntry, BindGroupLayoutDesc, BindGroupLayoutEntry, BindingCount,
        BindingResource, BindingType, ComputePipelineDesc, Error, PipelineLayoutDesc,
        PipelineStage, ShaderDesc, ShaderEntryPoint, ShaderFormat, ShaderRoute, ShaderStage,
        StorageAccess, StorageBufferView,
    },
    backend::{
        ExecutionBackend, MappingBackend, PipelineBackend, RasterBackend, ReadbackBackend,
        ResourceBackend, ShaderBackend,
    },
};
use fluxel_shader::{
    BindingAccess, BindingDeclaration, BindingKind, CallableDeclaration, EntryPoint, ScalarType,
    ShaderComposer, ShaderCompositionRequest, ShaderStage as LogicalShaderStage, SourceUnit,
    VertexType,
};

use crate::{
    ForwardDrawInput, PositionDeformation,
    renderer::admission::{self, PipelineCapabilityFacts},
    residency::{DeferredDeformation, DeformationAllocationPlan},
};

use super::*;

const WORKGROUP_SIZE: u32 = 64;
const POSITION_BYTES: u64 = 12;

/// CPU-only deformation facts, admitted before Deferred resource allocation.
pub(super) struct PreparedDeformation {
    allocation: DeformationAllocationPlan,
    descriptor: ShaderDesc,
    entry: String,
    vertex_count: u32,
    groups: [u32; 3],
}

impl<B> Renderer<B>
where
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + RasterBackend
        + PipelineBackend
        + ShaderBackend,
{
    /// Plans the fixed packed-position deformation product without allocating.
    pub(super) fn plan_deformation(
        &self,
        input: &ForwardDrawInput,
        spec: &PositionDeformation,
    ) -> std::result::Result<PreparedDeformation, RendererError> {
        self.ensure_open()?;
        if !spec.parameters.into_iter().all(|value| value.is_finite()) {
            return Err(Error::InvalidRange.into());
        }
        let source = position_bytes(input)?;
        let source_bytes = u64::try_from(source.len()).map_err(|_| Error::InvalidRange)?;
        let vertex_count =
            u32::try_from(source_bytes / POSITION_BYTES).map_err(|_| Error::InvalidRange)?;
        let groups = dispatch_groups(
            vertex_count,
            self.device.capabilities().limits.max_compute_workgroups,
        )?;
        let limits = self.device.capabilities().limits;
        if !self.device.capabilities().compute
            || !self.device.capabilities().storage_buffers
            || self.device.compute().is_none()
        {
            return Err(Error::Unsupported(
                "Deferred deformation requires compute storage buffers",
            )
            .into());
        }
        if limits.max_compute_workgroup_size[0] < WORKGROUP_SIZE
            || limits.max_compute_workgroup_size[1] == 0
            || limits.max_compute_workgroup_size[2] == 0
            || limits.max_bind_groups < 1
            || limits.max_bindings_per_group < 3
            || source_bytes > limits.max_buffer_bytes
        {
            return Err(
                Error::Unsupported("Deferred deformation workgroup or buffer limit").into(),
            );
        }
        let allocation = DeformationAllocationPlan::new(self.device.id(), source, spec.parameters)?;
        if [
            &allocation.upload_desc,
            &allocation.source_desc,
            &allocation.product_desc,
            &allocation.parameter_desc,
        ]
        .into_iter()
        .any(|desc| desc.size > limits.max_buffer_bytes)
        {
            return Err(Error::InvalidRange.into());
        }
        allocation.validate(&self.device)?;
        let artifact = compose(spec)?;
        let entry = artifact
            .entry_point(LogicalShaderStage::Compute)
            .ok_or(Error::InvalidState)?
            .name
            .clone();
        let descriptor = ShaderDesc {
            label: Some("Fluxel Deferred deformation program".into()),
            format: ShaderFormat::Wgsl,
            bytes: Arc::from(artifact.source().as_bytes()),
            entries: vec![ShaderEntryPoint {
                name: entry.clone(),
                stage: ShaderStage::Compute,
                workgroup_size: Some([WORKGROUP_SIZE, 1, 1]),
            }],
            required_features: artifact.requirements().names.iter().cloned().collect(),
        };
        admit(
            &self.device,
            &descriptor,
            artifact
                .requirements()
                .naga_capabilities
                .iter()
                .map(|capability| capability.name()),
        )?;
        Ok(PreparedDeformation {
            allocation,
            descriptor,
            entry,
            vertex_count,
            groups,
        })
    }

    /// Realizes the already-admitted compute program and its private owners.
    pub(super) async fn realize_deformation(
        &mut self,
        prepared: PreparedDeformation,
    ) -> std::result::Result<DeferredDeformation<B>, RendererError> {
        self.ensure_open()?;
        let shader = self
            .device
            .shader(prepared.descriptor)
            .await
            .map_err(|error| self.observe_rhi_error(error))?;
        let layout = self
            .device
            .create_bind_group_layout(&BindGroupLayoutDesc {
                label: Some("Deferred deformation bindings".into()),
                entries: vec![
                    storage(0, prepared.allocation.source_bytes, StorageAccess::Read),
                    storage(
                        1,
                        prepared.allocation.source_bytes,
                        StorageAccess::ReadWrite,
                    ),
                    uniform(2, 16),
                ],
            })
            .map_err(|error| self.observe_rhi_error(error))?;
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&PipelineLayoutDesc {
                label: Some("Deferred deformation pipeline layout"),
                groups: &[&layout],
                push_constants: &[],
            })
            .map_err(|error| self.observe_rhi_error(error))?;
        let pipeline = self
            .device
            .compute()
            .ok_or(Error::Unsupported("compute"))?
            .create_pipeline(ComputePipelineDesc {
                label: Some("Deferred deformation pipeline".into()),
                layout: pipeline_layout,
                compute: PipelineStage {
                    module: shader,
                    entry: prepared.entry,
                    constants: vec![],
                },
                required_subgroup_size: None,
                cache: None,
            })
            .await
            .map_err(|error| self.observe_rhi_error(error))?;
        let owners = DeferredDeformation::allocate_buffers(&self.device, prepared.allocation)
            .map_err(|error| self.observe_rhi_error(error))?;
        let group = self
            .device
            .create_bind_group(&BindGroupDesc {
                label: Some("Deferred deformation bindings"),
                layout: &layout,
                entries: &[
                    storage_entry(0, &owners.source.buffer, owners.source_bytes),
                    storage_entry(1, &owners.product.buffer, owners.source_bytes),
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::Buffer {
                            buffer: &owners.parameters.buffer,
                            range: 0..16,
                        },
                    },
                ],
            })
            .map_err(|error| self.observe_rhi_error(error))?;
        Ok(DeferredDeformation {
            source_instance: owners.source.instance,
            product_instance: owners.product.instance,
            parameter_instance: owners.parameters.instance,
            upload: owners.upload,
            source: owners.source,
            product: owners.product,
            parameters: owners.parameters,
            upload_bytes: owners.upload_bytes,
            parameter_source: owners.parameter_source,
            pipeline,
            group,
            groups: prepared.groups,
            vertex_count: prepared.vertex_count,
            source_bytes: owners.source_bytes,
        })
    }
}

fn position_bytes(input: &ForwardDrawInput) -> std::result::Result<Vec<u8>, Error> {
    let geometry = input.geometry().value();
    let [stream] = geometry.layout.streams.as_slice() else {
        return Err(Error::Unsupported(
            "Deferred deformation requires one position stream",
        ));
    };
    let [attribute] = stream.attributes.as_slice() else {
        return Err(Error::Unsupported(
            "Deferred deformation requires POSITION Float32x3",
        ));
    };
    if stream.stride != POSITION_BYTES
        || !matches!(stream.step_mode, fluxel_rhi::api::VertexStepMode::Vertex)
        || attribute.semantic != "POSITION"
        || attribute.offset != 0
        || !matches!(attribute.format, fluxel_rhi::api::VertexFormat::Float32(3))
    {
        return Err(Error::Unsupported(
            "Deferred deformation requires packed POSITION Float32x3",
        ));
    }
    let [bytes] = geometry.streams.as_slice() else {
        return Err(Error::InvalidRange);
    };
    let count = geometry.vertices.end;
    if geometry.vertices.start != 0
        || count == 0
        || bytes.len()
            != usize::try_from(u64::from(count) * POSITION_BYTES)
                .map_err(|_| Error::InvalidRange)?
    {
        return Err(Error::InvalidRange);
    }
    Ok(bytes.clone())
}

fn compose(
    spec: &PositionDeformation,
) -> std::result::Result<Arc<fluxel_shader::ShaderArtifact>, RendererError> {
    let request = ShaderCompositionRequest::new(vec![
        spec.source.clone(),
        SourceUnit::new("deferred-deformation-template", TEMPLATE),
    ])
    .with_entry_point(EntryPoint::new(
        "deferred_deformation_cs",
        LogicalShaderStage::Compute,
    ))
    .with_callable(CallableDeclaration {
        name: "deform_position".into(),
        arguments: vec![
            VertexType::Vec3(ScalarType::F32),
            VertexType::Vec4(ScalarType::F32),
        ],
        result: Some(VertexType::Vec3(ScalarType::F32)),
    })
    .with_binding(
        BindingDeclaration::new(0, 0, BindingKind::StorageRead)
            .with_visibility([LogicalShaderStage::Compute])
            .with_access(BindingAccess {
                read: true,
                write: false,
                query: true,
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
    )
    .with_binding(
        BindingDeclaration::new(0, 2, BindingKind::Uniform)
            .with_visibility([LogicalShaderStage::Compute])
            .with_access(BindingAccess {
                read: true,
                write: false,
                query: false,
                atomic: false,
            }),
    );
    ShaderComposer::new()
        .compose(&request)
        .map_err(RendererError::DeformationShader)
}

fn admit<B: ResourceBackend + ShaderBackend>(
    device: &fluxel_rhi::api::Device<B>,
    descriptor: &ShaderDesc,
    demands: impl IntoIterator<Item = &'static str>,
) -> std::result::Result<(), RendererError> {
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
        return Err(Error::Unsupported("Deferred deformation shader route").into());
    }
    Ok(())
}

fn dispatch_groups(count: u32, maximum: [u32; 3]) -> std::result::Result<[u32; 3], RendererError> {
    let groups = count
        .checked_add(WORKGROUP_SIZE - 1)
        .ok_or(Error::InvalidRange)?
        / WORKGROUP_SIZE;
    if groups == 0 || groups > maximum[0] || maximum[1] == 0 || maximum[2] == 0 {
        return Err(Error::Unsupported("Deferred deformation dispatch limit").into());
    }
    Ok([groups, 1, 1])
}

fn storage(binding: u32, min_size: u64, access: StorageAccess) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: vec![ShaderStage::Compute],
        ty: BindingType::StorageBuffer {
            dynamic_offset: false,
            min_size,
            access,
        },
        count: BindingCount::One,
        partially_bound: false,
        nonuniform_indexing: false,
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

fn storage_entry<'a, B: ResourceBackend>(
    binding: u32,
    buffer: &'a fluxel_rhi::api::Buffer<B>,
    size: u64,
) -> BindGroupEntry<'a, B> {
    BindGroupEntry {
        binding,
        resource: BindingResource::StorageBufferView {
            buffer,
            range: 0..size,
            view: StorageBufferView::Raw,
        },
    }
}

const TEMPLATE: &str = r#"
@group(0) @binding(0) var<storage, read> deformation_source: array<f32>;
@group(0) @binding(1) var<storage, read_write> deformation_product: array<f32>;
@group(0) @binding(2) var<uniform> deformation_parameters: vec4<f32>;

@compute @workgroup_size(64)
fn deferred_deformation_cs(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let vertex = invocation.x;
    if vertex >= arrayLength(&deformation_source) / 3u {
        return;
    }
    let word = vertex * 3u;
    let position = vec3<f32>(
        deformation_source[word],
        deformation_source[word + 1u],
        deformation_source[word + 2u],
    );
    let result = deform_position(position, deformation_parameters);
    deformation_product[word] = result.x;
    deformation_product[word + 1u] = result.y;
    deformation_product[word + 2u] = result.z;
}
"#;
