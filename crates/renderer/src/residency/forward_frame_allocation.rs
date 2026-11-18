//! Offscreen Forward frame allocation.

use super::*;

/// Plans a linear scene target plus an RGBA8 terminal target for Forward HDR.
pub(crate) fn plan_forward_hdr_frame(
    extent: [u32; 2],
    uniform_capacity: u64,
    uniform_len: u64,
    uniform_range: Range<u64>,
    row_pitch: u32,
    alignment: u64,
) -> Result<ForwardFramePlan> {
    let mut plan = plan_forward_frame(
        extent,
        uniform_capacity,
        uniform_len,
        uniform_range,
        row_pitch,
        alignment,
        true,
    )?;
    plan.target_desc.label = Some("fluxel forward HDR scene color".into());
    plan.target_desc.format = TextureFormat::Rgba16Float;
    plan.output_desc = Some(TextureDesc {
        label: Some("fluxel forward display output".into()),
        dimension: TextureDimension::D2,
        extent: [extent[0], extent[1], 1],
        array_layers: 1,
        format: TextureFormat::Rgba8Unorm,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
        view_formats: Vec::new(),
    });
    Ok(plan)
}

impl<B: ResourceBackend> ForwardFrameBuffers<B> {
    pub(crate) fn allocate_planned(
        device: &Device<B>,
        plan: ForwardFramePlan,
    ) -> Result<(Self, ForwardUniformUpload)> {
        for desc in [&plan.upload_desc, &plan.uniform_desc, &plan.readback_desc] {
            require_buffer(device, desc)?;
        }
        require_texture(device, &plan.target_desc)?;
        if let Some(output) = plan.output_desc.as_ref() {
            require_texture(device, output)?;
        }
        let upload = device.create_buffer(&plan.upload_desc)?;
        let uniform = device.create_buffer(&plan.uniform_desc)?;
        let target = device.create_texture(&plan.target_desc)?;
        let target_view = device.create_view(
            &target,
            &TextureViewDesc {
                format: plan.target_desc.format,
                dimension: TextureViewDimension::D2,
                aspects: fluxel_rhi::api::TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )?;
        let output = plan
            .output_desc
            .as_ref()
            .map(|desc| {
                let texture = device.create_texture(desc)?;
                let view = device.create_view(
                    &texture,
                    &TextureViewDesc {
                        format: desc.format,
                        dimension: TextureViewDimension::D2,
                        aspects: fluxel_rhi::api::TextureAspects::COLOR,
                        mips: 0..1,
                        layers: 0..1,
                    },
                )?;
                Ok(ForwardTexture {
                    texture,
                    view,
                    desc: desc.clone(),
                    instance: ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?,
                    incoming: empty_texture_uses(),
                    layout: ImageLayout::Undefined,
                    device: device.id(),
                    filterable: false,
                })
            })
            .transpose()?;
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
                output,
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
                output: None,
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
