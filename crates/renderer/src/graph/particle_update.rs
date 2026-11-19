//! Same-Graph upload, update, and vertex consumption for GPU particles.

use fluxel_rhi::{
    api::{AccessMask, BufferUsage, BufferUses, CommandEncoder, Device, Error, Result, StageMask},
    backend::{CommandBackend, ResourceBackend},
};

use crate::renderer::particle_update::{PARTICLE_PARAMETER_BYTES, ParticleStep};

use super::{
    binding::BufferSlot,
    definition::{
        AccessKind, BufferAccessContract, BufferHandle, GraphBuilder, GraphError, GraphPass,
        PassBuilder,
    },
    execution::{BoundBuffer, FrameBindings},
};

const fn access(
    stages: StageMask,
    accesses: AccessMask,
    usage: BufferUsage,
) -> BufferAccessContract {
    BufferAccessContract {
        uses: BufferUses { stages, accesses },
        descriptor_usage: usage,
    }
}

const COPY_READ: BufferAccessContract = access(
    StageMask::COPY,
    AccessMask::COPY_READ,
    BufferUsage::COPY_SRC,
);
const COPY_WRITE: BufferAccessContract = access(
    StageMask::COPY,
    AccessMask::COPY_WRITE,
    BufferUsage::COPY_DST,
);
const COMPUTE_UNIFORM: BufferAccessContract = access(
    StageMask::COMPUTE,
    AccessMask::UNIFORM_READ,
    BufferUsage::UNIFORM,
);
const COMPUTE_READ: BufferAccessContract = access(
    StageMask::COMPUTE,
    AccessMask::SHADER_READ,
    BufferUsage::STORAGE,
);
const COMPUTE_WRITE: BufferAccessContract = access(
    StageMask::COMPUTE,
    AccessMask::SHADER_WRITE,
    BufferUsage::STORAGE,
);
const VERTEX_READ: BufferAccessContract = access(
    StageMask::VERTEX_INPUT,
    AccessMask::VERTEX_READ,
    BufferUsage::VERTEX,
);

#[derive(Clone, Copy)]
pub(crate) struct ParticleUpdateGraph {
    pub(crate) upload_pass: GraphPass,
    pub(crate) update_pass: GraphPass,
    upload: BufferSlot,
    uniform: BufferSlot,
    source: BufferSlot,
    product: BufferHandle,
    state_bytes: u64,
    seed_uploaded: bool,
}

impl ParticleUpdateGraph {
    pub(crate) fn declare<B: ResourceBackend>(
        graph: &mut GraphBuilder,
        step: &ParticleStep<B>,
    ) -> std::result::Result<Self, GraphError> {
        let upload = graph.import_buffer_desc(step.upload.desc.clone(), step.upload.incoming)?;
        let uniform = graph.create_buffer_desc(step.uniform.desc.clone())?;
        let source = if let Some(incoming) = step.source_incoming {
            graph.import_buffer_desc(step.source().desc.clone(), incoming)?
        } else {
            graph.create_buffer_desc(step.source().desc.clone())?
        };
        let product = if let Some(incoming) = step.product_incoming {
            graph.import_buffer_desc(step.product().desc.clone(), incoming)?
        } else {
            graph.create_buffer_desc(step.product().desc.clone())?
        };
        let state_bytes = step.source().desc.size;
        let seed_uploaded = step.source_incoming.is_none();
        let (upload_pass, (source, uniform)) =
            graph.add_pass("particle-parameters-and-seed", |pass| {
                pass.read_with_use(
                    upload,
                    0..PARTICLE_PARAMETER_BYTES,
                    COPY_READ.uses,
                    COPY_READ.descriptor_usage,
                )?;
                let uniform = pass.write_with_use(
                    uniform,
                    0..PARTICLE_PARAMETER_BYTES,
                    COPY_WRITE.uses,
                    COPY_WRITE.descriptor_usage,
                )?;
                let source = if seed_uploaded {
                    let end = PARTICLE_PARAMETER_BYTES
                        .checked_add(state_bytes)
                        .ok_or(GraphError::InvalidRange)?;
                    pass.read_with_use(
                        upload,
                        PARTICLE_PARAMETER_BYTES..end,
                        COPY_READ.uses,
                        COPY_READ.descriptor_usage,
                    )?;
                    pass.write_with_use(
                        source,
                        0..state_bytes,
                        COPY_WRITE.uses,
                        COPY_WRITE.descriptor_usage,
                    )?
                } else {
                    source
                };
                Ok((source, uniform))
            })?;
        let (update_pass, product) = graph.add_pass("particle-update", |pass| {
            pass.read_with_use(
                uniform,
                0..PARTICLE_PARAMETER_BYTES,
                COMPUTE_UNIFORM.uses,
                COMPUTE_UNIFORM.descriptor_usage,
            )?;
            pass.read_with_use(
                source,
                0..state_bytes,
                COMPUTE_READ.uses,
                COMPUTE_READ.descriptor_usage,
            )?;
            pass.write_with_use(
                product,
                0..state_bytes,
                COMPUTE_WRITE.uses,
                COMPUTE_WRITE.descriptor_usage,
            )
        })?;
        Ok(Self {
            upload_pass,
            update_pass,
            upload: upload.into(),
            uniform: uniform.into(),
            source: source.into(),
            product,
            state_bytes,
            seed_uploaded,
        })
    }

    pub(crate) fn declare_vertex_read(
        &self,
        pass: &mut PassBuilder<'_>,
    ) -> std::result::Result<(), GraphError> {
        pass.read_with_use(
            self.product,
            0..self.state_bytes,
            VERTEX_READ.uses,
            VERTEX_READ.descriptor_usage,
        )
    }

    pub(crate) fn buffers<'a, B: ResourceBackend>(
        &self,
        step: &'a ParticleStep<B>,
    ) -> [BoundBuffer<'a, B>; 4] {
        [
            (self.upload, &step.upload, Some(step.upload.incoming)),
            (self.uniform, &step.uniform, Some(step.uniform.incoming)),
            (self.source, step.source(), step.source_incoming),
            (self.product.into(), step.product(), step.product_incoming),
        ]
        .map(|(slot, owner, incoming)| {
            BoundBuffer::new(
                slot,
                owner.instance,
                &owner.buffer,
                &owner.desc,
                incoming.unwrap_or(owner.incoming),
            )
        })
    }

    pub(crate) fn record_upload<B: CommandBackend>(
        &self,
        encoder: &mut CommandEncoder<B>,
        bindings: &FrameBindings<'_, B>,
    ) -> Result<()> {
        let pass = bindings.pass(self.upload_pass)?;
        let upload = pass.buffer(
            self.upload,
            0..PARTICLE_PARAMETER_BYTES,
            AccessKind::Read,
            COPY_READ,
        )?;
        let uniform = pass.buffer(
            self.uniform,
            0..PARTICLE_PARAMETER_BYTES,
            AccessKind::Write,
            COPY_WRITE,
        )?;
        let source = if self.seed_uploaded {
            let end = PARTICLE_PARAMETER_BYTES
                .checked_add(self.state_bytes)
                .ok_or(Error::InvalidRange)?;
            pass.buffer(
                self.upload,
                PARTICLE_PARAMETER_BYTES..end,
                AccessKind::Read,
                COPY_READ,
            )?;
            Some(pass.buffer(
                self.source,
                0..self.state_bytes,
                AccessKind::Write,
                COPY_WRITE,
            )?)
        } else {
            None
        };
        pass.transition(encoder)?;
        encoder.copy_buffer(upload, 0, uniform, 0, PARTICLE_PARAMETER_BYTES)?;
        if let Some(source) = source {
            encoder.copy_buffer(
                upload,
                PARTICLE_PARAMETER_BYTES,
                source,
                0,
                self.state_bytes,
            )?;
        }
        Ok(())
    }

    pub(crate) fn record_update<B: CommandBackend + ResourceBackend>(
        &self,
        device: &Device<B>,
        encoder: &mut CommandEncoder<B>,
        bindings: &FrameBindings<'_, B>,
        step: &ParticleStep<B>,
    ) -> Result<()> {
        bindings.require_buffer_owner(self.uniform, step.uniform.instance, &step.uniform.buffer)?;
        bindings.require_buffer_owner(
            self.source,
            step.source().instance,
            &step.source().buffer,
        )?;
        bindings.require_buffer_owner(
            self.product.into(),
            step.product().instance,
            &step.product().buffer,
        )?;
        let pass = bindings.pass(self.update_pass)?;
        pass.buffer(
            self.uniform,
            0..PARTICLE_PARAMETER_BYTES,
            AccessKind::Read,
            COMPUTE_UNIFORM,
        )?;
        pass.buffer(
            self.source,
            0..self.state_bytes,
            AccessKind::Read,
            COMPUTE_READ,
        )?;
        pass.buffer(
            self.product.into(),
            0..self.state_bytes,
            AccessKind::Write,
            COMPUTE_WRITE,
        )?;
        let domain = device
            .compute()
            .ok_or(Error::Unsupported("particle compute update"))?;
        pass.transition(encoder)?;
        let mut compute = domain.begin_labeled(encoder, "fluxel particle update")?;
        compute.set_pipeline(step.pipeline())?;
        compute.set_bind_group(0, &step.group, &[])?;
        compute.dispatch([step.groups, 1, 1])?;
        compute.finish()
    }

    /// Checks the actual particle vertex owner against the active pass.
    pub(crate) fn validate_vertex_read<B: ResourceBackend>(
        &self,
        raster_pass: GraphPass,
        bindings: &FrameBindings<'_, B>,
        step: &ParticleStep<B>,
    ) -> Result<()> {
        let slot = self.product.into();
        bindings.require_buffer_owner(slot, step.product().instance, &step.product().buffer)?;
        bindings.pass(raster_pass)?.buffer(
            slot,
            0..self.state_bytes,
            AccessKind::Read,
            VERTEX_READ,
        )?;
        Ok(())
    }
}
