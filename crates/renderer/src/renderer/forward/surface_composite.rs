//! Renderer-private fullscreen owner for a capture receiver Surface terminal.

use super::*;

/// Concrete pipeline, binding, and sampler retained until presentation proves
/// that a receiver target is no longer sampled by the terminal composite.
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
    color_format: TextureFormat,
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
        if self.device != device.id()
            || self.pipeline.device() != device.id()
            || self.group.device() != device.id()
            || self.sampler.device() != device.id()
            || self.target.device() != device.id()
            || self.target_view.device() != device.id()
            || image.device() != device.id()
            || !image.is_current()
            || image.extent() != self.extent
            || image.format() != self.color_format
            || !self
                .target_desc
                .usage
                .contains(TextureUsage::TEXTURE_BINDING)
        {
            return Err(
                if image.device() != device.id() || self.device != device.id() {
                    Error::WrongDevice
                } else if !image.is_current() {
                    Error::SurfaceOutdated
                } else {
                    Error::InvalidState
                },
            );
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
    pub(super) async fn surface_composite_owner(
        &mut self,
        receiver: &crate::residency::ForwardFrame<B>,
        image: &SurfaceImage<B>,
    ) -> std::result::Result<SurfaceCompositeOwner<B>, RendererError> {
        if receiver.extent != image.extent() || image.device() != self.device.id() {
            return Err(Error::InvalidRange.into());
        }
        if !image.is_current() {
            return Err(Error::SurfaceOutdated.into());
        }
        let target = &receiver.frame.target;
        let target_view = &receiver.frame.target_view;
        let desc = &receiver.frame.target_desc;
        if receiver.resident.device != self.device.id()
            || target.device() != self.device.id()
            || target_view.device() != self.device.id()
            || !desc.usage.contains(TextureUsage::TEXTURE_BINDING)
            || !matches!(self.device.texture_support(desc), Support::Supported(facts) if facts.filterable)
        {
            return Err(Error::InvalidState.into());
        }
        let program = self
            .deferred_scene_color_composite_program(image.format())
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
                label: Some("Forward capture surface composite".into()),
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
            target_instance: receiver.frame.target_instance,
            device: self.device.id(),
            extent: receiver.extent,
            color_format: image.format(),
        })
    }
}
