//! Private logical graph authoring with typed buffer versions.

use std::{
    ops::Range,
    sync::atomic::{AtomicU64, Ordering},
};

use fluxel_rhi::api::{
    BufferDesc, BufferUsage, BufferUses, ImageLayout, MemoryClass, TextureAspects, TextureDesc,
    TextureFormat, TextureUsage, TextureUses,
};

static NEXT_GRAPH: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct GraphId(u64);

/// A structured graph authoring failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum GraphError {
    UnknownResource,
    StaleVersion,
    InvalidRange,
    UndefinedRead,
    SelfDependency,
    UnknownPass,
    Cycle,
}

/// An opaque logical buffer version.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct BufferHandle {
    pub(crate) graph: GraphId,
    pub(crate) resource: usize,
    pub(crate) version: u32,
    nonce: u64,
}

/// An opaque logical texture version. It carries the private graph provenance
/// and has no relationship to a backend texture handle.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[allow(dead_code)] // First consumed by the pending Raster graph lowering.
pub(crate) struct TextureHandle {
    pub(crate) graph: GraphId,
    pub(crate) resource: usize,
    pub(crate) version: u32,
    nonce: u64,
}

/// A subresource selection expressed in the selected RHI vocabulary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TextureRange {
    pub(crate) aspects: TextureAspects,
    pub(crate) mips: Range<u32>,
    pub(crate) layers: Range<u32>,
}

#[allow(dead_code)] // Private authoring vocabulary, exercised by graph tests now.
impl TextureRange {
    /// Makes one selection after validating it against the logical descriptor.
    pub(crate) fn new(aspects: TextureAspects, mips: Range<u32>, layers: Range<u32>) -> Self {
        Self {
            aspects,
            mips,
            layers,
        }
    }
}

/// A validated byte range used by a graph buffer access.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ResourceRange(Range<u64>);

impl ResourceRange {
    /// Returns the validated byte range.
    pub(crate) fn bytes(&self) -> Range<u64> {
        self.0.clone()
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct GraphPass(pub(crate) GraphId, pub(crate) usize, u64);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AccessKind {
    Read,
    Write,
}

#[derive(Clone, Debug)]
pub(crate) struct Access {
    pub(crate) buffer: BufferHandle,
    pub(crate) range: ResourceRange,
    pub(crate) kind: AccessKind,
    pub(crate) contract: Option<BufferAccessContract>,
}

/// RHI access and creation usage declared by one buffer pass use.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BufferAccessContract {
    pub(crate) uses: BufferUses,
    pub(crate) descriptor_usage: BufferUsage,
}

/// A declared use of a logical texture version.
#[derive(Clone, Debug)]
#[allow(dead_code)] // Retained in the unified graph plan until Raster lowering consumes it.
pub(crate) struct TextureAccess {
    pub(crate) texture: TextureHandle,
    pub(crate) range: TextureRange,
    pub(crate) kind: AccessKind,
    pub(crate) contract: Option<TextureAccessContract>,
}

/// RHI access, layout, and creation usage declared by one texture pass use.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TextureAccessContract {
    pub(crate) uses: TextureUses,
    pub(crate) layout: ImageLayout,
    pub(crate) descriptor_usage: TextureUsage,
}

#[derive(Clone, Debug)]
pub(crate) struct PassNode {
    pub(crate) id: GraphPass,
    pub(crate) name: String,
    pub(crate) accesses: Vec<Access>,
    pub(crate) texture_accesses: Vec<TextureAccess>,
    pub(crate) dependencies: Vec<GraphPass>,
    pub(crate) side_effect: bool,
}

#[derive(Clone, Debug)]
#[allow(dead_code)] // Private texture version state for the forthcoming Raster path.
pub(crate) struct TextureVersionNode {
    nonce: u64,
    pub(crate) producer: Option<GraphPass>,
    pub(crate) defined: Vec<TextureRange>,
    pub(crate) readers: Vec<GraphPass>,
}

#[derive(Clone, Debug)]
#[allow(dead_code)] // Private texture description and version state.
pub(crate) struct TextureNode {
    pub(crate) desc: TextureDesc,
    pub(crate) initial: Option<(TextureUses, ImageLayout)>,
    pub(crate) versions: Vec<TextureVersionNode>,
    pub(crate) current: u32,
}

#[derive(Clone, Debug)]
pub(crate) struct VersionNode {
    nonce: u64,
    pub(crate) producer: Option<GraphPass>,
    pub(crate) defined: Vec<Range<u64>>,
    pub(crate) readers: Vec<GraphPass>,
}

#[derive(Clone, Debug)]
pub(crate) struct BufferNode {
    pub(crate) size: u64,
    pub(crate) usage: BufferUsage,
    pub(crate) memory: MemoryClass,
    pub(crate) initial: Option<BufferUses>,
    pub(crate) versions: Vec<VersionNode>,
    pub(crate) current: u32,
}

/// The private, backend-independent graph definition.
#[derive(Clone)]
pub(crate) struct GraphBuilder {
    pub(crate) graph: GraphId,
    pub(crate) buffers: Vec<BufferNode>,
    pub(crate) textures: Vec<TextureNode>,
    pub(crate) passes: Vec<PassNode>,
    pub(crate) exports: Vec<BufferHandle>,
    pub(crate) texture_exports: Vec<TextureHandle>,
    next_version_nonce: u64,
    next_pass_nonce: u64,
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self {
            graph: GraphId(
                NEXT_GRAPH
                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
                    .expect("private graph identity space exhausted"),
            ),
            buffers: Vec::new(),
            textures: Vec::new(),
            passes: Vec::new(),
            exports: Vec::new(),
            texture_exports: Vec::new(),
            next_version_nonce: 0,
            next_pass_nonce: 0,
        }
    }
}

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
    fn add_buffer(
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
    fn add_texture(
        &mut self,
        desc: TextureDesc,
        defined: bool,
        initial: Option<(TextureUses, ImageLayout)>,
    ) -> Result<TextureHandle, GraphError> {
        validate_texture_desc(&desc)?;
        let resource = self.textures.len();
        let defined = defined
            .then(|| vec![full_texture_range(&desc)])
            .unwrap_or_default();
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
    fn version_mut(&mut self, handle: BufferHandle) -> Result<&mut VersionNode, GraphError> {
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
    fn texture_version_mut(
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
    fn validate_range(
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
    fn add_dependency(&mut self, pass: GraphPass, predecessor: GraphPass) {
        let dependencies = &mut self.passes[pass.1].dependencies;
        if predecessor != pass && !dependencies.contains(&predecessor) {
            dependencies.push(predecessor);
        }
    }
    fn issue_version_nonce(&mut self) -> Result<u64, GraphError> {
        let nonce = self.next_version_nonce;
        self.next_version_nonce = nonce.checked_add(1).ok_or(GraphError::InvalidRange)?;
        Ok(nonce)
    }
    fn issue_pass_nonce(&mut self) -> Result<u64, GraphError> {
        let nonce = self.next_pass_nonce;
        self.next_pass_nonce = nonce.checked_add(1).ok_or(GraphError::InvalidRange)?;
        Ok(nonce)
    }
}

/// The pass-local authoring authority. It cannot record RHI work directly.
pub(crate) struct PassBuilder<'a> {
    graph: &'a mut GraphBuilder,
    pass: GraphPass,
}

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

fn range_is_defined(defined: &[Range<u64>], wanted: &ResourceRange) -> bool {
    defined
        .iter()
        .any(|range| range.start <= wanted.0.start && wanted.0.end <= range.end)
}

fn empty_buffer_uses() -> BufferUses {
    BufferUses {
        stages: fluxel_rhi::api::StageMask::EMPTY,
        accesses: fluxel_rhi::api::AccessMask::EMPTY,
    }
}

fn empty_texture_uses() -> TextureUses {
    TextureUses {
        stages: fluxel_rhi::api::StageMask::EMPTY,
        accesses: fluxel_rhi::api::AccessMask::EMPTY,
    }
}

fn normalize_ranges(ranges: &mut Vec<Range<u64>>) {
    ranges.sort_unstable_by_key(|range| range.start);
    let mut merged = Vec::<Range<u64>>::with_capacity(ranges.len());
    for range in ranges.drain(..) {
        if let Some(last) = merged.last_mut().filter(|last| range.start <= last.end) {
            last.end = last.end.max(range.end);
        } else {
            merged.push(range);
        }
    }
    *ranges = merged;
}

#[allow(dead_code)]
fn validate_texture_desc(desc: &TextureDesc) -> Result<(), GraphError> {
    if desc.extent.contains(&0)
        || desc.array_layers == 0
        || desc.mip_levels == 0
        || desc.samples == 0
    {
        return Err(GraphError::InvalidRange);
    }
    Ok(())
}

#[allow(dead_code)]
fn full_texture_range(desc: &TextureDesc) -> TextureRange {
    TextureRange::new(
        format_aspects(desc.format),
        0..desc.mip_levels,
        0..desc.array_layers,
    )
}

#[allow(dead_code)]
fn format_aspects(format: TextureFormat) -> TextureAspects {
    match format {
        TextureFormat::Depth16Unorm | TextureFormat::Depth24Plus | TextureFormat::Depth32Float => {
            TextureAspects::DEPTH
        }
        TextureFormat::Depth24PlusStencil8 | TextureFormat::Depth32FloatStencil8 => {
            TextureAspects::DEPTH | TextureAspects::STENCIL
        }
        TextureFormat::Stencil8 => TextureAspects::STENCIL,
        _ => TextureAspects::COLOR,
    }
}

#[allow(dead_code)]
fn validate_texture_range(desc: &TextureDesc, range: &TextureRange) -> Result<(), GraphError> {
    let valid_aspects = format_aspects(desc.format);
    if range.aspects.0 == 0
        || range.aspects.0 & !valid_aspects.0 != 0
        || range.mips.start >= range.mips.end
        || range.mips.end > desc.mip_levels
        || range.layers.start >= range.layers.end
        || range.layers.end > desc.array_layers
    {
        return Err(GraphError::InvalidRange);
    }
    Ok(())
}

#[allow(dead_code)]
fn texture_range_defined(defined: &[TextureRange], wanted: &TextureRange) -> bool {
    // Texture versions retain individual written subresources. A requested
    // range can consequently be fully defined by several adjacent writes; it
    // need not be enclosed by one original declaration.
    selected_texture_aspects(wanted.aspects).all(|aspect| {
        wanted.mips.clone().all(|mip| {
            wanted.layers.clone().all(|layer| {
                defined.iter().any(|range| {
                    range.aspects.0 & aspect != 0
                        && range.mips.start <= mip
                        && mip < range.mips.end
                        && range.layers.start <= layer
                        && layer < range.layers.end
                })
            })
        })
    })
}

fn selected_texture_aspects(aspects: TextureAspects) -> impl Iterator<Item = u8> {
    [
        TextureAspects::COLOR.0,
        TextureAspects::DEPTH.0,
        TextureAspects::STENCIL.0,
    ]
    .into_iter()
    .filter(move |aspect| aspects.0 & aspect != 0)
}

#[allow(dead_code)]
fn merge_texture_range(defined: &mut Vec<TextureRange>, range: TextureRange) {
    // A new logical version preserves all prior defined pieces. Coalescing is
    // intentionally conservative: only identical aspect/layer spans with
    // adjacent mip spans merge. It cannot invent defined subresources.
    if let Some(existing) = defined.iter_mut().find(|existing| {
        existing.aspects == range.aspects
            && existing.layers == range.layers
            && range.mips.start <= existing.mips.end
            && existing.mips.start <= range.mips.end
    }) {
        existing.mips.start = existing.mips.start.min(range.mips.start);
        existing.mips.end = existing.mips.end.max(range.mips.end);
    } else {
        defined.push(range);
    }
}
