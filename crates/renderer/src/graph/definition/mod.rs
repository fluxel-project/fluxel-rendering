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

/// The pass-local authoring authority. It cannot record RHI work directly.
pub(crate) struct PassBuilder<'a> {
    graph: &'a mut GraphBuilder,
    pass: GraphPass,
}

mod builder;
mod contract;
mod pass;
mod ranges;
use ranges::*;

#[cfg(test)]
#[path = "../../tests/graph/declaration.rs"]
mod declaration_tests;
