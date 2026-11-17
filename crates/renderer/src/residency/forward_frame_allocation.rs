//! Offscreen Forward frame allocation.

use super::*;

impl<B: ResourceBackend> ForwardFrameBuffers<B> {
    pub(crate) fn allocate_planned(
        device: &Device<B>,
        plan: ForwardFramePlan,
    ) -> Result<(Self, ForwardUniformUpload)> {
        for desc in [&plan.upload_desc, &plan.uniform_desc, &plan.readback_desc] {
            require_buffer(device, desc)?;
        }
        require_texture(device, &plan.target_desc)?;
        let upload = device.create_buffer(&plan.upload_desc)?;
        let uniform = device.create_buffer(&plan.uniform_desc)?;
        let target = device.create_texture(&plan.target_desc)?;
        let target_view = device.create_view(
            &target,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: fluxel_rhi::api::TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )?;
        let readback = device.create_buffer(&plan.readback_desc)?;
        Ok((
            Self {
                upload,
                upload_desc: plan.upload_desc,
                upload_instance: ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?,
                upload_incoming: empty_buffer_uses(),
                uniform,
                uniform_desc: plan.uniform_desc,
                uniform_instance: ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?,
                uniform_incoming: empty_buffer_uses(),
                target,
                target_desc: plan.target_desc,
                target_instance: ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?,
                target_view,
                target_incoming: empty_texture_uses(),
                target_layout: ImageLayout::Undefined,
                depth: None,
                readback,
                readback_desc: plan.readback_desc,
                readback_instance: ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?,
                readback_incoming: empty_buffer_uses(),
            },
            plan.uniform_upload,
        ))
    }
}
