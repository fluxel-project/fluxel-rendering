//! Mock program, binding and vertex-array domains.
//!
//! A program and the inputs it consumes change together whenever the
//! shader/binding contract changes, and never with the transfer or sync
//! domains.

use super::*;

impl GlBindingApi for MockGlFamilyApi {
    fn active_texture(&mut self, unit: u32) -> Result<(), GlError> {
        self.ready("active-texture")?;
        if let Err(error) = validate_texture_unit(unit, self.binding_limits()) {
            return Err(self.invalid_binding("active-texture", error));
        }
        self.calls.push(MockCall::ActiveTexture(unit));
        Ok(())
    }
    fn bind_texture(
        &mut self,
        unit: u32,
        target: GlTextureTarget,
        texture: Option<TextureId>,
    ) -> Result<(), GlError> {
        self.ready("bind-texture")?;
        if let Err(error) = validate_texture_unit(unit, self.binding_limits()) {
            return Err(self.invalid_binding("bind-texture", error));
        }
        if let Some(texture) = texture {
            self.texture("bind-texture", texture)?;
        }
        self.calls.push(MockCall::BindTexture {
            unit,
            target,
            texture,
        });
        Ok(())
    }
    fn bind_sampler(&mut self, unit: u32, sampler: Option<SamplerId>) -> Result<(), GlError> {
        self.ready("bind-sampler")?;
        if let Err(error) = validate_texture_unit(unit, self.binding_limits()) {
            return Err(self.invalid_binding("bind-sampler", error));
        }
        if let Some(sampler) = sampler {
            self.live("bind-sampler", sampler, |this| {
                this.samplers.contains(&sampler)
            })?;
        }
        self.calls.push(MockCall::BindSampler { unit, sampler });
        Ok(())
    }
    fn bind_uniform_buffer(
        &mut self,
        index: u32,
        buffer: Option<BufferId>,
        offset: u32,
        size: u32,
    ) -> Result<(), GlError> {
        self.ready("bind-uniform-buffer")?;
        if let Err(error) =
            validate_uniform_buffer_binding(index, buffer, offset, size, self.binding_limits())
        {
            return Err(self.invalid_binding("bind-uniform-buffer", error));
        }
        if let Some(buffer) = buffer {
            let desc = self.buffer("bind-uniform-buffer", buffer)?;
            if !desc.usage.contains(GlBufferUsage::UNIFORM) {
                return self.invalid("bind-uniform-buffer", "buffer lacks uniform usage");
            }
            if let Err(error) = validate_uniform_range(offset, size, desc.size) {
                return Err(self.invalid_binding("bind-uniform-buffer", error));
            }
        }
        self.calls.push(MockCall::BindUniformBuffer {
            index,
            buffer,
            offset,
            size,
        });
        Ok(())
    }
}
impl GlShaderApi for MockGlFamilyApi {
    fn create_shader(&mut self, s: &GlShaderSource) -> Result<ShaderId, GlError> {
        self.ready("create-shader")?;
        s.validate_for(self.profile())
            .map_err(|_| GlError::Validation {
                operation: "create-shader",
                message: "shader source does not target this profile".into(),
            })?;
        let id = ShaderId::new(self.stamp, self.slot()?, 0);
        self.shaders.insert(id);
        self.calls.push(MockCall::CreateShader(id));
        Ok(id)
    }
    fn destroy_shader(&mut self, id: ShaderId) -> Result<(), GlError> {
        self.ready("destroy-shader")?;
        self.live("destroy-shader", id, |this| this.shaders.contains(&id))?;
        self.shaders.remove(&id);
        self.calls.push(MockCall::DestroyShader(id));
        Ok(())
    }
    fn create_program(
        &mut self,
        d: &GlProgramDescriptor,
    ) -> Result<(ProgramId, GlProgramReflection), GlError> {
        self.ready("create-program")?;
        d.validate_for(self.profile())
            .map_err(|_| GlError::Validation {
                operation: "create-program",
                message: "invalid program descriptor".into(),
            })?;
        if matches!(d.kind, GlProgramKind::Compute { .. })
            && !self
                .discovery
                .capabilities()
                .supports(GlCapability::Compute)
        {
            return self.error_result(GlError::Unsupported {
                operation: "create-program",
                reason: "compute program requires proved compute capability",
            });
        }
        // An injected reflection must satisfy the same layout agreement a
        // real provider validates, so differential tests exercise the same
        // failure modes; without injection every program reflects as empty.
        // The check runs before any identity is published so a mismatch
        // leaves no half-initialized object behind.
        let reflection = match self.next_reflection.take() {
            Some(reflection) => {
                reflection
                    .validate_against(&d.layout)
                    .map_err(|_| GlError::Validation {
                        operation: "create-program",
                        message: "injected reflection does not satisfy the layout".into(),
                    })?;
                reflection
            }
            None => GlProgramReflection {
                vertex_inputs: vec![],
                fragment_outputs: vec![],
                assignments: vec![],
            },
        };
        let id = ProgramId::new(self.stamp, self.slot()?, 0);
        self.programs.insert(id);
        self.calls.push(MockCall::CreateProgram(id));
        Ok((id, reflection))
    }
    fn destroy_program(&mut self, id: ProgramId) -> Result<(), GlError> {
        self.ready("destroy-program")?;
        self.live("destroy-program", id, |this| this.programs.contains(&id))?;
        self.programs.remove(&id);
        self.calls.push(MockCall::DestroyProgram(id));
        Ok(())
    }
}
impl GlVertexApi for MockGlFamilyApi {
    fn create_vertex_array(&mut self, l: &GlVertexLayout) -> Result<VertexArrayId, GlError> {
        self.ready("create-vertex-array")?;
        l.validate().map_err(|_| GlError::Validation {
            operation: "create-vertex-array",
            message: "invalid vertex layout".into(),
        })?;
        let id = VertexArrayId::new(self.stamp, self.slot()?, 0);
        self.vaos.insert(id);
        self.calls.push(MockCall::CreateVertexArray(id));
        Ok(id)
    }
    fn destroy_vertex_array(&mut self, id: VertexArrayId) -> Result<(), GlError> {
        self.ready("destroy-vertex-array")?;
        self.live("destroy-vertex-array", id, |this| this.vaos.contains(&id))?;
        self.vaos.remove(&id);
        self.calls.push(MockCall::DestroyVertexArray(id));
        Ok(())
    }
    fn bind_vertex_array(
        &mut self,
        id: VertexArrayId,
        buffers: &[GlVertexBufferBinding],
        index: Option<GlIndexBinding>,
    ) -> Result<(), GlError> {
        self.ready("bind-vertex-array")?;
        self.live("bind-vertex-array", id, |this| this.vaos.contains(&id))?;
        for b in buffers {
            self.buffer("bind-vertex-array", b.buffer)?;
        }
        if let Some(i) = index {
            self.buffer("bind-vertex-array", i.buffer)?;
        }
        self.calls.push(MockCall::BindVertexArray(id));
        Ok(())
    }
}
