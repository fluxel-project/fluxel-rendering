//! Renderer-private fullscreen presentation of frame-local Forward color.

use super::*;

/// Concrete pipeline, binding, and sampler retained until presentation proves
/// that the source target is no longer sampled by the terminal composite.
pub(crate) struct SurfaceCompositeOwner<B: ResourceBackend> {
    pipeline: RasterPipeline<B>,
    group: BindGroup<B>,
    sampler: Sampler<B>,
    target: Texture<B>,
    target_view: TextureView<B>,
    target_desc: TextureDesc,
    target_instance: crate::graph::ResourceInstanceId,
    device: DeviceId,
    extent: [u32; 2],
    target_format: TextureFormat,
}

impl<B: ResourceBackend> SurfaceCompositeOwner<B> {
    pub(crate) fn pipeline(&self) -> &RasterPipeline<B> {
        &self.pipeline
    }

    pub(crate) fn bind_group(&self) -> &BindGroup<B> {
        &self.group
    }

    pub(crate) const fn extent(&self) -> [u32; 2] {
        self.extent
    }

    pub(crate) fn validate_for(
        &self,
        device: &Device<B>,
        image: &SurfaceImage<B>,
    ) -> std::result::Result<(), Error> {
        self.validate_offscreen(device, image.extent(), image.format())?;
        if image.device() != device.id() || !image.is_current() {
            return Err(if image.device() != device.id() {
                Error::WrongDevice
            } else if !image.is_current() {
                Error::SurfaceOutdated
            } else {
                Error::InvalidState
            });
        }
        Ok(())
    }

    /// Validates a terminal composite against an ordinary offscreen target.
    /// This deliberately does not construct a synthetic `SurfaceImage`.
    pub(crate) fn validate_offscreen(
        &self,
        device: &Device<B>,
        extent: [u32; 2],
        target_format: TextureFormat,
    ) -> std::result::Result<(), Error> {
        if self.device != device.id()
            || self.pipeline.device() != device.id()
            || self.group.device() != device.id()
            || self.sampler.device() != device.id()
            || self.target.device() != device.id()
            || self.target_view.device() != device.id()
            || self.extent != extent
            || self.target_format != target_format
            || self.target_desc.dimension != TextureDimension::D2
            || self.target_desc.extent != [extent[0], extent[1], 1]
            || self.target_desc.array_layers != 1
            || self.target_desc.mip_levels != 1
            || self.target_desc.samples != 1
            || !composite_source_format(self.target_desc.format)
            || !self
                .target_desc
                .usage
                .contains(TextureUsage::TEXTURE_BINDING)
        {
            return Err(if self.device != device.id() {
                Error::WrongDevice
            } else {
                Error::InvalidState
            });
        }
        Ok(())
    }

    pub(crate) fn target_instance(&self) -> crate::graph::ResourceInstanceId {
        self.target_instance
    }
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
    pub(in crate::renderer) async fn surface_composite_owner(
        &mut self,
        receiver: &crate::residency::ForwardFrame<B>,
        image: &SurfaceImage<B>,
    ) -> std::result::Result<SurfaceCompositeOwner<B>, RendererError> {
        if receiver.resident.device != self.device.id() {
            return Err(Error::InvalidState.into());
        }
        self.surface_composite_from_target(
            &receiver.frame.target,
            &receiver.frame.target_view,
            &receiver.frame.target_desc,
            receiver.frame.target_instance,
            receiver.extent,
            image,
        )
        .await
    }

    /// Presents the actual Deferred lighting result while preserving its
    /// separate sampled owner and the acquired image's destination format.
    pub(in crate::renderer) async fn deferred_surface_composite_owner(
        &mut self,
        color: &crate::residency::ForwardTexture<B>,
        extent: [u32; 2],
        image: &SurfaceImage<B>,
    ) -> std::result::Result<SurfaceCompositeOwner<B>, RendererError> {
        if color.device != self.device.id() {
            return Err(Error::WrongDevice.into());
        }
        self.surface_composite_from_target(
            &color.texture,
            &color.view,
            &color.desc,
            color.instance,
            extent,
            image,
        )
        .await
    }

    /// Builds the terminal surface composite from OIT's actual final output,
    /// never from the opaque intermediate used by peeling.
    pub(in crate::renderer) async fn oit_surface_composite_owner(
        &mut self,
        output: &crate::residency::OitTextureOwner<B>,
        extent: [u32; 2],
        image: &SurfaceImage<B>,
    ) -> std::result::Result<SurfaceCompositeOwner<B>, RendererError> {
        validate_oit_output(&self.device, output, extent)?;
        self.surface_composite_from_target(
            &output.texture,
            &output.view,
            &output.desc,
            output.instance,
            extent,
            image,
        )
        .await
    }

    /// Builds the same fullscreen color composite for an offscreen terminal.
    /// The source is OIT's completed output, not the opaque input target.
    pub(in crate::renderer) async fn oit_offscreen_composite_owner(
        &mut self,
        output: &crate::residency::OitTextureOwner<B>,
        extent: [u32; 2],
        target_format: TextureFormat,
    ) -> std::result::Result<SurfaceCompositeOwner<B>, RendererError> {
        validate_oit_output(&self.device, output, extent)?;
        self.color_composite_from_target(
            &output.texture,
            &output.view,
            &output.desc,
            output.instance,
            extent,
            target_format,
        )
        .await
    }

    pub(in crate::renderer) async fn surface_composite_from_target(
        &mut self,
        target: &Texture<B>,
        target_view: &TextureView<B>,
        desc: &TextureDesc,
        target_instance: crate::graph::ResourceInstanceId,
        extent: [u32; 2],
        image: &SurfaceImage<B>,
    ) -> std::result::Result<SurfaceCompositeOwner<B>, RendererError> {
        if extent != image.extent() || image.device() != self.device.id() {
            return Err(Error::InvalidRange.into());
        }
        if !image.is_current() {
            return Err(Error::SurfaceOutdated.into());
        }
        self.color_composite_from_target(
            target,
            target_view,
            desc,
            target_instance,
            extent,
            image.format(),
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::renderer) async fn color_composite_from_target(
        &mut self,
        target: &Texture<B>,
        target_view: &TextureView<B>,
        desc: &TextureDesc,
        target_instance: crate::graph::ResourceInstanceId,
        extent: [u32; 2],
        target_format: TextureFormat,
    ) -> std::result::Result<SurfaceCompositeOwner<B>, RendererError> {
        validate_composite_source(&self.device, target, target_view, desc, extent)?;
        let program = self
            .scene_color_composite_program(target_format, desc.format)
            .await?;
        let sampler_desc = default_material_sampler();
        validate_ordinary_sampler(&sampler_desc)
            .map_err(|error| self.observe_forward_error("surface composite sampler", error))?;
        let sampler = self
            .device
            .create_sampler(&sampler_desc)
            .map_err(|error| self.observe_forward_error("surface composite sampler", error))?;
        let group = self
            .device
            .create_bind_group(&BindGroupDesc {
                label: Some("Forward surface composite".into()),
                layout: &program.layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(target_view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::Sampler(&sampler),
                    },
                ],
            })
            .map_err(|error| self.observe_forward_error("surface composite bind group", error))?;
        Ok(SurfaceCompositeOwner {
            pipeline: program.pipeline,
            group,
            sampler,
            target: target.clone(),
            target_view: target_view.clone(),
            target_desc: desc.clone(),
            target_instance,
            device: self.device.id(),
            extent,
            target_format,
        })
    }
}

fn validate_composite_source<B: ResourceBackend>(
    device: &Device<B>,
    target: &Texture<B>,
    target_view: &TextureView<B>,
    desc: &TextureDesc,
    extent: [u32; 2],
) -> std::result::Result<(), RendererError> {
    if target.device() != device.id()
        || target_view.device() != device.id()
        || desc.dimension != TextureDimension::D2
        || desc.extent != [extent[0], extent[1], 1]
        || desc.array_layers != 1
        || desc.mip_levels != 1
        || desc.samples != 1
        || !composite_source_format(desc.format)
        || !desc.usage.contains(TextureUsage::TEXTURE_BINDING)
        || !matches!(device.texture_support(desc), Support::Supported(facts) if facts.filterable)
    {
        return Err(Error::InvalidState.into());
    }
    Ok(())
}

fn composite_source_format(format: TextureFormat) -> bool {
    matches!(
        format,
        TextureFormat::Rgba8Unorm | TextureFormat::Rgba16Float
    )
}

fn validate_oit_output<B: ResourceBackend>(
    device: &Device<B>,
    output: &crate::residency::OitTextureOwner<B>,
    extent: [u32; 2],
) -> std::result::Result<(), RendererError> {
    if output.device != device.id()
        || output.texture.device() != device.id()
        || output.view.device() != device.id()
        || output.view_desc.format != output.desc.format
        || output.view_desc.dimension != TextureViewDimension::D2
        || output.view_desc.aspects != TextureAspects::COLOR
        || output.view_desc.mips != (0..1)
        || output.view_desc.layers != (0..1)
    {
        return Err(Error::InvalidState.into());
    }
    validate_composite_source(device, &output.texture, &output.view, &output.desc, extent)
}
