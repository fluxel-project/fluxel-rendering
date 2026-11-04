//! Pass-local access declarations and version updates.
use super::*;

impl PassBuilder<'_> {
    /// Returns this pass identity for private dependency authoring.
    #[allow(dead_code)] // Exercised by graph failure tests and later pass composition.
    pub(crate) fn current(&self) -> GraphPass {
        self.pass
    }

    /// Declares a read of a defined logical buffer version.
    pub(crate) fn read(
        &mut self,
        buffer: BufferHandle,
        range: Range<u64>,
    ) -> Result<(), GraphError> {
        let range = self.graph.validate_range(buffer, range)?;
        if self
            .graph
            .buffers
            .get(buffer.resource)
            .ok_or(GraphError::UnknownResource)?
            .current
            != buffer.version
        {
            return Err(GraphError::StaleVersion);
        }
        let version = self.graph.version(buffer)?;
        if !range_is_defined(&version.defined, &range) {
            return Err(GraphError::UndefinedRead);
        }
        if let Some(producer) = version.producer {
            self.graph.add_dependency(self.pass, producer);
        }
        let version = self.graph.version_mut(buffer)?;
        if !version.readers.contains(&self.pass) {
            version.readers.push(self.pass);
        }
        self.graph.passes[self.pass.1].accesses.push(Access {
            buffer,
            range,
            kind: AccessKind::Read,
            contract: None,
        });
        Ok(())
    }
    /// Declares a read with its exact portable RHI access and descriptor use.
    pub(crate) fn read_with_use(
        &mut self,
        buffer: BufferHandle,
        range: Range<u64>,
        uses: BufferUses,
        descriptor_usage: BufferUsage,
    ) -> Result<(), GraphError> {
        self.read(buffer, range)?;
        self.graph.passes[self.pass.1]
            .accesses
            .last_mut()
            .expect("read appends its declared access")
            .contract = Some(BufferAccessContract {
            uses,
            descriptor_usage,
        });
        Ok(())
    }
    /// Declares a read of a fully-defined selected texture subresource range.
    #[allow(dead_code)] // The next Raster graph lowering consumes this declaration.
    pub(crate) fn read_texture(
        &mut self,
        texture: TextureHandle,
        range: TextureRange,
    ) -> Result<(), GraphError> {
        self.validate_texture_current(texture, &range)?;
        let version = self.graph.texture_version(texture)?;
        if !texture_range_defined(&version.defined, &range) {
            return Err(GraphError::UndefinedRead);
        }
        if let Some(producer) = version.producer {
            self.graph.add_dependency(self.pass, producer);
        }
        let version = self.graph.texture_version_mut(texture)?;
        if !version.readers.contains(&self.pass) {
            version.readers.push(self.pass);
        }
        self.graph.passes[self.pass.1]
            .texture_accesses
            .push(TextureAccess {
                texture,
                range,
                kind: AccessKind::Read,
                contract: None,
            });
        Ok(())
    }
    /// Declares a texture read with its portable RHI usage and layout.
    #[allow(dead_code)]
    pub(crate) fn read_texture_with_use(
        &mut self,
        texture: TextureHandle,
        range: TextureRange,
        uses: TextureUses,
        layout: ImageLayout,
        descriptor_usage: TextureUsage,
    ) -> Result<(), GraphError> {
        self.read_texture(texture, range)?;
        self.graph.passes[self.pass.1]
            .texture_accesses
            .last_mut()
            .expect("texture read appends its declared access")
            .contract = Some(TextureAccessContract {
            uses,
            layout,
            descriptor_usage,
        });
        Ok(())
    }
    /// Declares a write and returns the newly defined immutable version.
    pub(crate) fn write(
        &mut self,
        buffer: BufferHandle,
        range: Range<u64>,
    ) -> Result<BufferHandle, GraphError> {
        let range = self.graph.validate_range(buffer, range)?;
        if self
            .graph
            .buffers
            .get(buffer.resource)
            .ok_or(GraphError::UnknownResource)?
            .current
            != buffer.version
        {
            return Err(GraphError::StaleVersion);
        }
        let prior = self.graph.version(buffer)?.clone();
        if let Some(producer) = prior.producer {
            self.graph.add_dependency(self.pass, producer);
        }
        for reader in prior.readers {
            self.graph.add_dependency(self.pass, reader);
        }
        let nonce = self.graph.issue_version_nonce()?;
        let resource = self
            .graph
            .buffers
            .get_mut(buffer.resource)
            .ok_or(GraphError::UnknownResource)?;
        let version =
            u32::try_from(resource.versions.len()).map_err(|_| GraphError::InvalidRange)?;
        let mut defined = prior.defined;
        defined.push(range.bytes());
        normalize_ranges(&mut defined);
        resource.versions.push(VersionNode {
            nonce,
            producer: Some(self.pass),
            defined,
            readers: Vec::new(),
        });
        resource.current = version;
        let output = BufferHandle {
            graph: self.graph.graph,
            resource: buffer.resource,
            version,
            nonce,
        };
        self.graph.passes[self.pass.1].accesses.push(Access {
            buffer: output,
            range,
            kind: AccessKind::Write,
            contract: None,
        });
        Ok(output)
    }
    /// Declares a buffer write with its exact portable RHI access and descriptor use.
    pub(crate) fn write_with_use(
        &mut self,
        buffer: BufferHandle,
        range: Range<u64>,
        uses: BufferUses,
        descriptor_usage: BufferUsage,
    ) -> Result<BufferHandle, GraphError> {
        let output = self.write(buffer, range)?;
        self.graph.passes[self.pass.1]
            .accesses
            .last_mut()
            .expect("write appends its declared access")
            .contract = Some(BufferAccessContract {
            uses,
            descriptor_usage,
        });
        Ok(output)
    }
    /// Declares a write and returns the next immutable logical texture version.
    #[allow(dead_code)] // The next Raster graph lowering consumes this declaration.
    pub(crate) fn write_texture(
        &mut self,
        texture: TextureHandle,
        range: TextureRange,
    ) -> Result<TextureHandle, GraphError> {
        self.validate_texture_current(texture, &range)?;
        let prior = self.graph.texture_version(texture)?.clone();
        if let Some(producer) = prior.producer {
            self.graph.add_dependency(self.pass, producer);
        }
        for reader in prior.readers {
            self.graph.add_dependency(self.pass, reader);
        }
        let nonce = self.graph.issue_version_nonce()?;
        let node = self
            .graph
            .textures
            .get_mut(texture.resource)
            .ok_or(GraphError::UnknownResource)?;
        let version = u32::try_from(node.versions.len()).map_err(|_| GraphError::InvalidRange)?;
        let mut defined = prior.defined;
        merge_texture_range(&mut defined, range.clone());
        node.versions.push(TextureVersionNode {
            nonce,
            producer: Some(self.pass),
            defined,
            readers: Vec::new(),
        });
        node.current = version;
        let output = TextureHandle {
            graph: self.graph.graph,
            resource: texture.resource,
            version,
            nonce,
        };
        self.graph.passes[self.pass.1]
            .texture_accesses
            .push(TextureAccess {
                texture: output,
                range,
                kind: AccessKind::Write,
                contract: None,
            });
        Ok(output)
    }
    /// Declares a texture write with its portable RHI usage and layout.
    pub(crate) fn write_texture_with_use(
        &mut self,
        texture: TextureHandle,
        range: TextureRange,
        uses: TextureUses,
        layout: ImageLayout,
        descriptor_usage: TextureUsage,
    ) -> Result<TextureHandle, GraphError> {
        let output = self.write_texture(texture, range)?;
        self.graph.passes[self.pass.1]
            .texture_accesses
            .last_mut()
            .expect("texture write appends its declared access")
            .contract = Some(TextureAccessContract {
            uses,
            layout,
            descriptor_usage,
        });
        Ok(output)
    }
    /// Adds an explicit ordering edge to an already authored pass.
    #[allow(dead_code)] // Kept for the next multi-pass graph consumer.
    pub(crate) fn depends_on(&mut self, predecessor: GraphPass) -> Result<(), GraphError> {
        if predecessor.0 != self.graph.graph {
            return Err(GraphError::UnknownPass);
        }
        if predecessor == self.pass {
            return Err(GraphError::SelfDependency);
        }
        if predecessor.1 >= self.pass.1
            || self
                .graph
                .passes
                .get(predecessor.1)
                .is_none_or(|node| node.id != predecessor)
        {
            return Err(GraphError::UnknownPass);
        }
        self.graph.add_dependency(self.pass, predecessor);
        Ok(())
    }
    /// Makes this pass a culling root even when it produces no exported resource.
    pub(crate) fn side_effect(&mut self) {
        self.graph.passes[self.pass.1].side_effect = true;
    }

    #[allow(dead_code)]
    fn validate_texture_current(
        &self,
        texture: TextureHandle,
        range: &TextureRange,
    ) -> Result<(), GraphError> {
        if texture.graph != self.graph.graph {
            return Err(GraphError::UnknownResource);
        }
        let node = self
            .graph
            .textures
            .get(texture.resource)
            .ok_or(GraphError::UnknownResource)?;
        if node.current != texture.version {
            return Err(GraphError::StaleVersion);
        }
        validate_texture_range(&node.desc, range)
    }
}
