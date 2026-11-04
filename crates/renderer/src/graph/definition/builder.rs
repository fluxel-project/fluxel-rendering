//! Logical resource creation and transactional pass insertion.
use super::*;

impl GraphBuilder {
    /// Creates an undefined transient logical buffer.
    #[allow(dead_code)] // Compact authoring helper used by graph-only tests.
    pub(crate) fn create_buffer(&mut self, size: u64) -> Result<BufferHandle, GraphError> {
        self.add_buffer(
            BufferDesc {
                label: None,
                size,
                usage: BufferUsage(0),
                memory: MemoryClass::Device,
            },
            false,
            Some(empty_buffer_uses()),
        )
    }
    /// Creates an undefined logical buffer with its portable RHI descriptor.
    pub(crate) fn create_buffer_desc(
        &mut self,
        desc: BufferDesc,
    ) -> Result<BufferHandle, GraphError> {
        self.add_buffer(desc, false, Some(empty_buffer_uses()))
    }
    /// Imports logical contents that are already defined by an external producer.
    #[allow(dead_code)] // Compact authoring helper used by graph-only tests.
    pub(crate) fn import_buffer(&mut self, size: u64) -> Result<BufferHandle, GraphError> {
        self.add_buffer(
            BufferDesc {
                label: None,
                size,
                usage: BufferUsage(0),
                memory: MemoryClass::Device,
            },
            true,
            None,
        )
    }
    /// Imports defined contents together with their RHI descriptor and incoming state.
    pub(crate) fn import_buffer_desc(
        &mut self,
        desc: BufferDesc,
        initial: BufferUses,
    ) -> Result<BufferHandle, GraphError> {
        self.add_buffer(desc, true, Some(initial))
    }
    /// Creates an undefined transient texture from the selected RHI descriptor.
    #[allow(dead_code)] // Used by the next private Raster graph lowering.
    pub(crate) fn create_texture(
        &mut self,
        desc: TextureDesc,
    ) -> Result<TextureHandle, GraphError> {
        self.add_texture(
            desc,
            false,
            Some((empty_texture_uses(), ImageLayout::Undefined)),
        )
    }
    /// Imports a texture whose complete subresource set is defined externally.
    #[allow(dead_code)] // Used by external texture imports in the next Raster fixture.
    pub(crate) fn import_texture(
        &mut self,
        desc: TextureDesc,
    ) -> Result<TextureHandle, GraphError> {
        self.add_texture(desc, true, None)
    }
    /// Imports a fully-defined texture and its exact RHI incoming state.
    #[allow(dead_code)] // Surface graph lowering will supply its acquired-image state here.
    pub(crate) fn import_texture_with_state(
        &mut self,
        desc: TextureDesc,
        initial: TextureUses,
        layout: ImageLayout,
    ) -> Result<TextureHandle, GraphError> {
        self.add_texture(desc, true, Some((initial, layout)))
    }
    /// Declares a pass and records its local accesses.
    pub(crate) fn add_pass<T>(
        &mut self,
        name: impl Into<String>,
        build: impl FnOnce(&mut PassBuilder<'_>) -> Result<T, GraphError>,
    ) -> Result<(GraphPass, T), GraphError> {
        let before = self.clone();
        let pass = GraphPass(self.graph, self.passes.len(), self.issue_pass_nonce()?);
        self.passes.push(PassNode {
            id: pass,
            name: name.into(),
            accesses: Vec::new(),
            texture_accesses: Vec::new(),
            dependencies: Vec::new(),
            side_effect: false,
        });
        match build(&mut PassBuilder { graph: self, pass }) {
            Ok(result) => Ok((pass, result)),
            Err(error) => {
                let next_version_nonce = self.next_version_nonce;
                let next_pass_nonce = self.next_pass_nonce;
                *self = before;
                // Handles and passes that escaped a failed closure must never
                // become aliases for a later successful transaction.
                self.next_version_nonce = next_version_nonce;
                self.next_pass_nonce = next_pass_nonce;
                Err(error)
            }
        }
    }
    /// Marks a fully-defined texture version as externally observed.
    #[allow(dead_code)] // Used when the Raster target becomes an observed graph output.
    pub(crate) fn export_texture(&mut self, texture: TextureHandle) -> Result<(), GraphError> {
        if texture.graph != self.graph {
            return Err(GraphError::UnknownResource);
        }
        let node = self
            .textures
            .get(texture.resource)
            .ok_or(GraphError::UnknownResource)?;
        let version = self.texture_version(texture)?;
        if !texture_range_defined(&version.defined, &full_texture_range(&node.desc)) {
            return Err(GraphError::UndefinedRead);
        }
        self.texture_exports.push(texture);
        Ok(())
    }
    /// Marks one version as an externally observed graph result.
    #[allow(dead_code)] // Used by graph consumers after the first copy fixture.
    pub(crate) fn export(&mut self, buffer: BufferHandle) -> Result<(), GraphError> {
        if buffer.graph != self.graph {
            return Err(GraphError::UnknownResource);
        }
        let resource = self
            .buffers
            .get(buffer.resource)
            .ok_or(GraphError::UnknownResource)?;
        let version = self.version(buffer)?;
        if !range_is_defined(&version.defined, &ResourceRange(0..resource.size)) {
            return Err(GraphError::UndefinedRead);
        }
        self.exports.push(buffer);
        Ok(())
    }
    pub(super) fn add_buffer(
        &mut self,
        desc: BufferDesc,
        defined: bool,
        initial: Option<BufferUses>,
    ) -> Result<BufferHandle, GraphError> {
        if desc.size == 0 {
            return Err(GraphError::InvalidRange);
        }
        let resource = self.buffers.len();
        let nonce = self.issue_version_nonce()?;
        self.buffers.push(BufferNode {
            size: desc.size,
            usage: desc.usage,
            memory: desc.memory,
            initial,
            versions: vec![VersionNode {
                nonce,
                producer: None,
                defined: defined.then_some(0..desc.size).into_iter().collect(),
                readers: Vec::new(),
            }],
            current: 0,
        });
        Ok(BufferHandle {
            graph: self.graph,
            resource,
            version: 0,
            nonce,
        })
    }
    #[allow(dead_code)]
    pub(super) fn add_texture(
        &mut self,
        desc: TextureDesc,
        defined: bool,
        initial: Option<(TextureUses, ImageLayout)>,
    ) -> Result<TextureHandle, GraphError> {
        validate_texture_desc(&desc)?;
        let resource = self.textures.len();
        let defined = if defined {
            vec![full_texture_range(&desc)]
        } else {
            Default::default()
        };
        let nonce = self.issue_version_nonce()?;
        self.textures.push(TextureNode {
            desc,
            initial,
            versions: vec![TextureVersionNode {
                nonce,
                producer: None,
                defined,
                readers: Vec::new(),
            }],
            current: 0,
        });
        Ok(TextureHandle {
            graph: self.graph,
            resource,
            version: 0,
            nonce,
        })
    }
    pub(crate) fn version(&self, handle: BufferHandle) -> Result<&VersionNode, GraphError> {
        if handle.graph != self.graph {
            return Err(GraphError::UnknownResource);
        }
        self.buffers
            .get(handle.resource)
            .ok_or(GraphError::UnknownResource)?
            .versions
            .get(handle.version as usize)
            .filter(|version| version.nonce == handle.nonce)
            .ok_or(GraphError::StaleVersion)
    }
    pub(super) fn version_mut(
        &mut self,
        handle: BufferHandle,
    ) -> Result<&mut VersionNode, GraphError> {
        if handle.graph != self.graph {
            return Err(GraphError::UnknownResource);
        }
        self.buffers
            .get_mut(handle.resource)
            .ok_or(GraphError::UnknownResource)?
            .versions
            .get_mut(handle.version as usize)
            .filter(|version| version.nonce == handle.nonce)
            .ok_or(GraphError::StaleVersion)
    }
    #[allow(dead_code)]
    pub(crate) fn texture_version(
        &self,
        handle: TextureHandle,
    ) -> Result<&TextureVersionNode, GraphError> {
        if handle.graph != self.graph {
            return Err(GraphError::UnknownResource);
        }
        self.textures
            .get(handle.resource)
            .ok_or(GraphError::UnknownResource)?
            .versions
            .get(handle.version as usize)
            .filter(|version| version.nonce == handle.nonce)
            .ok_or(GraphError::StaleVersion)
    }
    #[allow(dead_code)]
    pub(super) fn texture_version_mut(
        &mut self,
        handle: TextureHandle,
    ) -> Result<&mut TextureVersionNode, GraphError> {
        if handle.graph != self.graph {
            return Err(GraphError::UnknownResource);
        }
        self.textures
            .get_mut(handle.resource)
            .ok_or(GraphError::UnknownResource)?
            .versions
            .get_mut(handle.version as usize)
            .filter(|version| version.nonce == handle.nonce)
            .ok_or(GraphError::StaleVersion)
    }
    pub(super) fn validate_range(
        &self,
        buffer: BufferHandle,
        range: Range<u64>,
    ) -> Result<ResourceRange, GraphError> {
        if buffer.graph != self.graph {
            return Err(GraphError::UnknownResource);
        }
        let resource = self
            .buffers
            .get(buffer.resource)
            .ok_or(GraphError::UnknownResource)?;
        if range.start >= range.end || range.end > resource.size {
            return Err(GraphError::InvalidRange);
        }
        Ok(ResourceRange(range))
    }
    pub(super) fn add_dependency(&mut self, pass: GraphPass, predecessor: GraphPass) {
        let dependencies = &mut self.passes[pass.1].dependencies;
        if predecessor != pass && !dependencies.contains(&predecessor) {
            dependencies.push(predecessor);
        }
    }
    pub(super) fn issue_version_nonce(&mut self) -> Result<u64, GraphError> {
        let nonce = self.next_version_nonce;
        self.next_version_nonce = nonce.checked_add(1).ok_or(GraphError::InvalidRange)?;
        Ok(nonce)
    }
    pub(super) fn issue_pass_nonce(&mut self) -> Result<u64, GraphError> {
        let nonce = self.next_pass_nonce;
        self.next_pass_nonce = nonce.checked_add(1).ok_or(GraphError::InvalidRange)?;
        Ok(nonce)
    }
}
