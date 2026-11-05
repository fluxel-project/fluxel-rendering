//! Private owners for one packed-position compute deformation product.
//!
//! The source and result buffers are deliberately independent from Forward
//! residency. Forward streams are vertex-only, while this bounded product must
//! be admitted as storage before compute can consume it.

use std::ops::Range;

use fluxel_rhi::{
    api::{
        BindGroup, Buffer, BufferDesc, BufferUsage, BufferUses, ComputePipeline, Device, DeviceId,
        Error, MemoryClass, Result, Support,
    },
    backend::ResourceBackend,
};

use crate::graph::ResourceInstanceId;

/// One concrete buffer and the facts Graph binding validation consumes.
pub(crate) struct DeformationBuffer<B: ResourceBackend> {
    pub(crate) buffer: Buffer<B>,
    pub(crate) desc: BufferDesc,
    pub(crate) instance: ResourceInstanceId,
    pub(crate) incoming: BufferUses,
}

/// Handle-free allocation facts validated before the first deformation owner.
pub(crate) struct DeformationAllocationPlan {
    device: DeviceId,
    pub(crate) upload_desc: BufferDesc,
    pub(crate) source_desc: BufferDesc,
    pub(crate) product_desc: BufferDesc,
    pub(crate) parameter_desc: BufferDesc,
    pub(crate) upload_bytes: Vec<u8>,
    pub(crate) parameter_source: Range<u64>,
    pub(crate) source_bytes: u64,
}

impl DeformationAllocationPlan {
    pub(crate) fn new(device: DeviceId, source: Vec<u8>, parameters: [f32; 4]) -> Result<Self> {
        let source_bytes = u64::try_from(source.len()).map_err(|_| Error::InvalidRange)?;
        if source_bytes == 0 || source_bytes % 4 != 0 {
            return Err(Error::InvalidRange);
        }
        let parameter_start = align_up(source_bytes, 256)?;
        let parameter_end = parameter_start.checked_add(16).ok_or(Error::InvalidRange)?;
        let upload_size = parameter_end.max(16);
        let mut upload_bytes = source;
        upload_bytes.resize(
            usize::try_from(upload_size).map_err(|_| Error::InvalidRange)?,
            0,
        );
        for (index, value) in parameters.into_iter().enumerate() {
            let start = usize::try_from(parameter_start)
                .map_err(|_| Error::InvalidRange)?
                .checked_add(index.checked_mul(4).ok_or(Error::InvalidRange)?)
                .ok_or(Error::InvalidRange)?;
            upload_bytes[start..start + 4].copy_from_slice(&value.to_le_bytes());
        }
        Ok(Self {
            device,
            upload_desc: buffer_desc(
                "fluxel deferred deformation upload",
                upload_size,
                BufferUsage::COPY_SRC | BufferUsage::MAP_WRITE,
                MemoryClass::Upload,
            ),
            source_desc: buffer_desc(
                "fluxel deferred deformation source",
                source_bytes,
                BufferUsage::COPY_DST | BufferUsage::STORAGE,
                MemoryClass::Device,
            ),
            product_desc: buffer_desc(
                "fluxel deferred deformation product",
                source_bytes,
                BufferUsage::STORAGE | BufferUsage::VERTEX,
                MemoryClass::Device,
            ),
            parameter_desc: buffer_desc(
                "fluxel deferred deformation parameters",
                16,
                BufferUsage::COPY_DST | BufferUsage::UNIFORM,
                MemoryClass::Device,
            ),
            upload_bytes,
            parameter_source: parameter_start..parameter_end,
            source_bytes,
        })
    }

    pub(crate) fn validate<B: ResourceBackend>(&self, device: &Device<B>) -> Result<()> {
        if self.device != device.id() {
            return Err(Error::WrongDevice);
        }
        for desc in [
            &self.upload_desc,
            &self.source_desc,
            &self.product_desc,
            &self.parameter_desc,
        ] {
            if let Support::Unsupported(reason) = device.buffer_support(desc) {
                return Err(Error::Unsupported(reason));
            }
        }
        Ok(())
    }
}

/// Completion-owned compute pipeline, bindings, and packed position buffers.
pub(crate) struct DeferredDeformation<B: ResourceBackend> {
    pub(crate) upload: DeformationBuffer<B>,
    pub(crate) source: DeformationBuffer<B>,
    pub(crate) product: DeformationBuffer<B>,
    pub(crate) parameters: DeformationBuffer<B>,
    pub(crate) upload_bytes: Vec<u8>,
    pub(crate) parameter_source: Range<u64>,
    pub(crate) pipeline: ComputePipeline<B>,
    pub(crate) group: BindGroup<B>,
    pub(crate) source_instance: ResourceInstanceId,
    pub(crate) product_instance: ResourceInstanceId,
    pub(crate) parameter_instance: ResourceInstanceId,
    pub(crate) groups: [u32; 3],
    pub(crate) vertex_count: u32,
    pub(crate) source_bytes: u64,
}

impl<B: ResourceBackend> DeferredDeformation<B> {
    pub(crate) fn allocate_buffers(
        device: &Device<B>,
        plan: DeformationAllocationPlan,
    ) -> Result<DeformationOwners<B>> {
        plan.validate(device)?;
        Ok(DeformationOwners {
            upload: allocate(device, plan.upload_desc.clone())?,
            source: allocate(device, plan.source_desc.clone())?,
            product: allocate(device, plan.product_desc.clone())?,
            parameters: allocate(device, plan.parameter_desc.clone())?,
            upload_bytes: plan.upload_bytes,
            parameter_source: plan.parameter_source,
            source_bytes: plan.source_bytes,
        })
    }
}

/// Buffer-only intermediate used while the renderer creates the bind group.
pub(crate) struct DeformationOwners<B: ResourceBackend> {
    pub(crate) upload: DeformationBuffer<B>,
    pub(crate) source: DeformationBuffer<B>,
    pub(crate) product: DeformationBuffer<B>,
    pub(crate) parameters: DeformationBuffer<B>,
    pub(crate) upload_bytes: Vec<u8>,
    pub(crate) parameter_source: Range<u64>,
    pub(crate) source_bytes: u64,
}

fn allocate<B: ResourceBackend>(
    device: &Device<B>,
    desc: BufferDesc,
) -> Result<DeformationBuffer<B>> {
    let buffer = device.create_buffer(&desc)?;
    Ok(DeformationBuffer {
        buffer,
        desc,
        instance: ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?,
        incoming: BufferUses {
            stages: fluxel_rhi::api::StageMask::EMPTY,
            accesses: fluxel_rhi::api::AccessMask::EMPTY,
        },
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

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    value
        .checked_add(alignment - 1)
        .map(|next| next / alignment * alignment)
        .ok_or(Error::InvalidRange)
}
