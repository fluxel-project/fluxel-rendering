//! Mock object domains: buffers, textures, renderbuffers and samplers.
//!
//! These implement the creation/destruction half of the recorder and own
//! the identity tables that every other mock domain validates against.

use super::*;

impl GlResourceApi for MockGlFamilyApi {
    fn create_buffer_resource(&mut self, desc: GlBufferDesc) -> Result<BufferId, GlError> {
        self.ready("create-buffer")?;
        desc.validate().map_err(|error| GlError::Validation {
            operation: "create-buffer",
            message: error.message(),
        })?;
        self.validate_buffer_allocation(desc)?;
        let id = BufferId::new(self.stamp, self.slot()?, 0);
        self.buffers.insert(id, desc);
        self.calls.push(MockCall::CreateBuffer(id));
        Ok(id)
    }
    fn create_texture_resource(&mut self, desc: GlTextureDesc) -> Result<TextureId, GlError> {
        self.ready("create-texture")?;
        desc.validate().map_err(|_| GlError::Validation {
            operation: "create-texture",
            message: "invalid texture descriptor".into(),
        })?;
        self.validate_texture_allocation(desc)?;
        let id = TextureId::new(self.stamp, self.slot()?, 0);
        self.textures.insert(id, desc);
        self.calls.push(MockCall::CreateTexture(id));
        Ok(id)
    }
    fn destroy_buffer_resource(&mut self, id: BufferId) -> Result<(), GlError> {
        self.ready("destroy-buffer")?;
        self.buffer("destroy-buffer", id)?;
        self.buffers.remove(&id);
        self.calls.push(MockCall::DestroyBuffer(id));
        Ok(())
    }
    fn destroy_texture_resource(&mut self, id: TextureId) -> Result<(), GlError> {
        self.ready("destroy-texture")?;
        self.texture("destroy-texture", id)?;
        self.textures.remove(&id);
        self.calls.push(MockCall::DestroyTexture(id));
        Ok(())
    }
    fn create_render_buffer(
        &mut self,
        desc: GlRenderBufferDesc,
    ) -> Result<RenderbufferId, GlError> {
        self.ready("create-render-buffer")?;
        desc.validate().map_err(|_| GlError::Validation {
            operation: "create-render-buffer",
            message: "invalid renderbuffer descriptor".into(),
        })?;
        let limits = self.discovery.limits();
        if desc.samples > limits.max_samples
            || desc.width > limits.max_renderbuffer_size
            || desc.height > limits.max_renderbuffer_size
        {
            return self.invalid(
                "create-render-buffer",
                "renderbuffer samples or extent exceed the discovered limits",
            );
        }
        let facts = self
            .discovery
            .formats()
            .get_for(
                GlFormatResourceKind::Renderbuffer,
                desc.format,
                desc.samples,
            )
            .ok_or_else(|| GlError::Validation {
                operation: "create-render-buffer",
                message: "no exact discovered format fact for this renderbuffer allocation".into(),
            })
            .inspect_err(|error| self.error(error.clone()))?;
        if !facts.renderable {
            return self.invalid(
                "create-render-buffer",
                "format is not renderable for this sample count",
            );
        }
        let id = RenderbufferId::new(self.stamp, self.slot()?, 0);
        self.render_buffers.insert(id, desc);
        self.calls.push(MockCall::CreateRenderBuffer(id));
        Ok(id)
    }
    fn destroy_render_buffer(&mut self, id: RenderbufferId) -> Result<(), GlError> {
        self.ready("destroy-render-buffer")?;
        self.render_buffer("destroy-render-buffer", id)?;
        self.render_buffers.remove(&id);
        self.calls.push(MockCall::DestroyRenderBuffer(id));
        Ok(())
    }
}
impl GlSamplerApi for MockGlFamilyApi {
    fn create_sampler(&mut self, d: GlSamplerDesc) -> Result<SamplerId, GlError> {
        self.ready("create-sampler")?;
        d.validate_for(&self.discovery)
            .map_err(|_| GlError::Validation {
                operation: "create-sampler",
                message: "invalid sampler".into(),
            })?;
        let id = SamplerId::new(self.stamp, self.slot()?, 0);
        self.samplers.insert(id);
        self.calls.push(MockCall::CreateSampler(id));
        Ok(id)
    }
    fn destroy_sampler(&mut self, id: SamplerId) -> Result<(), GlError> {
        self.ready("destroy-sampler")?;
        self.live("destroy-sampler", id, |this| this.samplers.contains(&id))?;
        self.samplers.remove(&id);
        self.calls.push(MockCall::DestroySampler(id));
        Ok(())
    }
}
