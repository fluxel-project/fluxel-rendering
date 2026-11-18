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

    /// Allocates only per-overlay buffers and retains actual Deferred color and
    /// depth owners.  The caller has already admitted their descriptor and
    /// provenance; this method deliberately does not create replacement
    /// attachments before installing the borrowed ones.
    pub(crate) fn allocate_with_attachments(
        device: &Device<B>,
        plan: ForwardFramePlan,
        color: &super::ForwardTexture<B>,
        depth: &super::ForwardTexture<B>,
    ) -> Result<(Self, ForwardUniformUpload)> {
        for desc in [&plan.upload_desc, &plan.uniform_desc, &plan.readback_desc] {
            require_buffer(device, desc)?;
        }
        if color.device != device.id()
            || depth.device != device.id()
            || color.texture.device() != device.id()
            || color.view.device() != device.id()
            || depth.texture.device() != device.id()
            || depth.view.device() != device.id()
        {
            return Err(Error::WrongDevice);
        }
        let depth_attachment = super::ForwardDepthAttachment::from_texture(depth);
        Ok((
            Self {
                upload: device.create_buffer(&plan.upload_desc)?,
                upload_desc: plan.upload_desc,
                upload_instance: ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?,
                upload_incoming: empty_buffer_uses(),
                uniform: device.create_buffer(&plan.uniform_desc)?,
                uniform_desc: plan.uniform_desc,
                uniform_instance: ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?,
                uniform_incoming: empty_buffer_uses(),
                target: color.texture.clone(),
                target_desc: color.desc.clone(),
                target_instance: color.instance,
                target_view: color.view.clone(),
                target_incoming: color.incoming,
                target_layout: color.layout,
                depth: Some(depth_attachment),
                readback: device.create_buffer(&plan.readback_desc)?,
                readback_desc: plan.readback_desc,
                readback_instance: ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?,
                readback_incoming: empty_buffer_uses(),
            },
            plan.uniform_upload,
        ))
    }
}
