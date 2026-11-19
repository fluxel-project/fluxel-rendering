//! Graph declaration and recording for one GPU-produced material draw record.

use fluxel_rhi::{
    api::{
        AccessMask, BufferUsage, BufferUses, CommandEncoder, Device, Error, RasterPass, Result,
        StageMask,
    },
    backend::{CommandBackend, RasterBackend, ResourceBackend},
};

use crate::renderer::indirect_producer::IndirectProduct;

use super::{
    binding::BufferSlot,
    definition::{
        AccessKind, BufferAccessContract, BufferHandle, GraphBuilder, GraphError, GraphPass,
        PassBuilder,
    },
    execution::{BoundBuffer, FrameBindings},
};

const RECORD_BYTES: u64 = 16;
const PARAMETER_BYTES: u64 = 16;

const UPLOAD_READ: BufferAccessContract = BufferAccessContract {
    uses: BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    },
    descriptor_usage: BufferUsage::COPY_SRC,
};
const UPLOAD_WRITE: BufferAccessContract = BufferAccessContract {
    uses: BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_WRITE,
    },
    descriptor_usage: BufferUsage::COPY_DST,
};
const PARAMETER_READ: BufferAccessContract = BufferAccessContract {
    uses: BufferUses {
        stages: StageMask::COMPUTE,
        accesses: AccessMask::UNIFORM_READ,
    },
    descriptor_usage: BufferUsage::UNIFORM,
};
const ARGUMENT_WRITE: BufferAccessContract = BufferAccessContract {
    uses: BufferUses {
        stages: StageMask::COMPUTE,
        accesses: AccessMask::SHADER_WRITE,
    },
    descriptor_usage: BufferUsage::STORAGE,
};
const ARGUMENT_READ: BufferAccessContract = BufferAccessContract {
    uses: BufferUses {
        stages: StageMask::INDIRECT,
        accesses: AccessMask::INDIRECT_READ,
    },
    descriptor_usage: BufferUsage::INDIRECT,
};

#[derive(Clone, Copy)]
pub(crate) struct IndirectGraph {
    pub(crate) upload_pass: GraphPass,
    pub(crate) compute_pass: GraphPass,
    upload: BufferSlot,
    parameters: BufferSlot,
    arguments: BufferHandle,
}

impl IndirectGraph {
    pub(crate) fn declare<B: ResourceBackend>(
        graph: &mut GraphBuilder,
        product: &IndirectProduct<B>,
    ) -> std::result::Result<Self, GraphError> {
        let upload =
            graph.import_buffer_desc(product.upload.desc.clone(), product.upload.incoming)?;
        let parameters = graph.create_buffer_desc(product.parameters.desc.clone())?;
        let arguments = graph.create_buffer_desc(product.arguments.desc.clone())?;
        let (upload_pass, parameters) = graph.add_pass("indirect-parameters-upload", |pass| {
            pass.read_with_use(
                upload,
                0..PARAMETER_BYTES,
                UPLOAD_READ.uses,
                UPLOAD_READ.descriptor_usage,
            )?;
            pass.write_with_use(
                parameters,
                0..PARAMETER_BYTES,
                UPLOAD_WRITE.uses,
                UPLOAD_WRITE.descriptor_usage,
            )
        })?;
        let (compute_pass, arguments) = graph.add_pass("indirect-produce-arguments", |pass| {
            pass.read_with_use(
                parameters,
                0..PARAMETER_BYTES,
                PARAMETER_READ.uses,
                PARAMETER_READ.descriptor_usage,
            )?;
            pass.write_with_use(
                arguments,
                0..RECORD_BYTES,
                ARGUMENT_WRITE.uses,
                ARGUMENT_WRITE.descriptor_usage,
            )
        })?;
        Ok(Self {
            upload_pass,
            compute_pass,
            upload: upload.into(),
            parameters: parameters.into(),
            arguments,
        })
    }

    pub(crate) fn declare_raster_read(
        &self,
        pass: &mut PassBuilder<'_>,
    ) -> std::result::Result<(), GraphError> {
        pass.read_with_use(
            self.arguments,
            0..RECORD_BYTES,
            ARGUMENT_READ.uses,
            ARGUMENT_READ.descriptor_usage,
        )
    }

    pub(crate) fn buffers<'a, B: ResourceBackend>(
        &self,
        product: &'a IndirectProduct<B>,
    ) -> [BoundBuffer<'a, B>; 3] {
        [
            (self.upload, &product.upload),
            (self.parameters, &product.parameters),
            (self.arguments.into(), &product.arguments),
        ]
        .map(|(slot, owner)| {
            BoundBuffer::new(
                slot,
                owner.instance,
                &owner.buffer,
                &owner.desc,
                owner.incoming,
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
            0..PARAMETER_BYTES,
            AccessKind::Read,
            UPLOAD_READ,
        )?;
        let parameters = pass.buffer(
            self.parameters,
            0..PARAMETER_BYTES,
            AccessKind::Write,
            UPLOAD_WRITE,
        )?;
        pass.transition(encoder)?;
        encoder.copy_buffer(upload, 0, parameters, 0, PARAMETER_BYTES)
    }

    pub(crate) fn record_compute<B: CommandBackend + ResourceBackend>(
        &self,
        device: &Device<B>,
        encoder: &mut CommandEncoder<B>,
        bindings: &FrameBindings<'_, B>,
        product: &IndirectProduct<B>,
    ) -> Result<()> {
        bindings.require_buffer_owner(
            self.parameters,
            product.parameters.instance,
            &product.parameters.buffer,
        )?;
        bindings.require_buffer_owner(
            self.arguments.into(),
            product.arguments.instance,
            &product.arguments.buffer,
        )?;
        let pass = bindings.pass(self.compute_pass)?;
        pass.buffer(
            self.parameters,
            0..PARAMETER_BYTES,
            AccessKind::Read,
            PARAMETER_READ,
        )?;
        pass.buffer(
            self.arguments.into(),
            0..RECORD_BYTES,
            AccessKind::Write,
            ARGUMENT_WRITE,
        )?;
        let domain = device.compute().ok_or(Error::Unsupported("compute"))?;
        pass.transition(encoder)?;
        let mut compute = domain.begin_labeled(encoder, "fluxel indirect arguments")?;
        compute.set_pipeline(&product.pipeline)?;
        compute.set_bind_group(0, &product.group, &[])?;
        compute.dispatch([1, 1, 1])?;
        compute.finish()
    }

    pub(crate) fn validate_raster_read<B: ResourceBackend>(
        &self,
        raster_pass: GraphPass,
        bindings: &FrameBindings<'_, B>,
        product: &IndirectProduct<B>,
    ) -> Result<()> {
        let slot = self.arguments.into();
        bindings.require_buffer_owner(
            slot,
            product.arguments.instance,
            &product.arguments.buffer,
        )?;
        bindings.pass(raster_pass)?.buffer(
            slot,
            0..RECORD_BYTES,
            AccessKind::Read,
            ARGUMENT_READ,
        )?;
        Ok(())
    }
}

pub(crate) fn draw<B: RasterBackend + ResourceBackend>(
    device: &Device<B>,
    raster: &mut RasterPass<'_, B>,
    product: &IndirectProduct<B>,
) -> Result<()> {
    let domain = device
        .raster_extras()
        .ok_or(Error::Unsupported("indirect draw"))?;
    domain.draw_indirect(raster, &product.arguments.buffer, 0, false)
}
