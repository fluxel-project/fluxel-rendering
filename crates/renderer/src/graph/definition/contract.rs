//! Exact declaration equivalence without compiling another execution plan.
use super::*;

impl GraphBuilder {
    /// Compares deterministic authoring order, independently of graph identity.
    ///
    /// This is deliberately conservative: every resource fact, version, access,
    /// dependency and output must match. The caller also validates its recording
    /// shape and concrete owners. No scheduling or barrier compilation occurs.
    pub(crate) fn same_declaration(&self, candidate: &Self) -> bool {
        self.valid_references()
            && candidate.valid_references()
            && same_items(&self.buffers, &candidate.buffers, |a, b| {
                a.size == b.size
                    && a.usage == b.usage
                    && a.memory == b.memory
                    && a.initial == b.initial
                    && a.current == b.current
                    && same_items(&a.versions, &b.versions, |a, b| {
                        a.producer.map(pass_index) == b.producer.map(pass_index)
                            && a.defined == b.defined
                            && same_passes(&a.readers, &b.readers)
                    })
            })
            && same_items(&self.textures, &candidate.textures, |a, b| {
                same_texture_desc(&a.desc, &b.desc)
                    && a.initial == b.initial
                    && a.current == b.current
                    && same_items(&a.versions, &b.versions, |a, b| {
                        a.producer.map(pass_index) == b.producer.map(pass_index)
                            && a.defined == b.defined
                            && same_passes(&a.readers, &b.readers)
                    })
            })
            && same_items(&self.passes, &candidate.passes, |a, b| {
                a.side_effect == b.side_effect
                    && same_passes(&a.dependencies, &b.dependencies)
                    && same_items(&a.accesses, &b.accesses, |a, b| {
                        buffer_key(a.buffer) == buffer_key(b.buffer)
                            && a.range == b.range
                            && a.kind == b.kind
                            && a.contract == b.contract
                    })
                    && same_items(&a.texture_accesses, &b.texture_accesses, |a, b| {
                        texture_key(a.texture) == texture_key(b.texture)
                            && a.range == b.range
                            && a.kind == b.kind
                            && a.contract == b.contract
                    })
            })
            && same_items(&self.exports, &candidate.exports, |a, b| {
                buffer_key(*a) == buffer_key(*b)
            })
            && same_items(&self.texture_exports, &candidate.texture_exports, |a, b| {
                texture_key(*a) == texture_key(*b)
            })
    }

    fn valid_references(&self) -> bool {
        let valid_pass = |pass: GraphPass| {
            pass.0 == self.graph && self.passes.get(pass.1).is_some_and(|node| node.id == pass)
        };
        self.passes.iter().enumerate().all(|(index, pass)| {
            pass.id.1 == index
                && valid_pass(pass.id)
                && pass.dependencies.iter().all(|id| valid_pass(*id))
                && pass
                    .accesses
                    .iter()
                    .all(|access| self.version(access.buffer).is_ok())
                && pass
                    .texture_accesses
                    .iter()
                    .all(|access| self.texture_version(access.texture).is_ok())
        }) && self.buffers.iter().all(|node| {
            (node.current as usize) < node.versions.len()
                && node.versions.iter().all(|version| {
                    version.producer.is_none_or(valid_pass)
                        && version.readers.iter().all(|id| valid_pass(*id))
                })
        }) && self.textures.iter().all(|node| {
            (node.current as usize) < node.versions.len()
                && node.versions.iter().all(|version| {
                    version.producer.is_none_or(valid_pass)
                        && version.readers.iter().all(|id| valid_pass(*id))
                })
        }) && self
            .exports
            .iter()
            .all(|handle| self.version(*handle).is_ok())
            && self
                .texture_exports
                .iter()
                .all(|handle| self.texture_version(*handle).is_ok())
    }
}

fn same_items<T>(a: &[T], b: &[T], equal: impl Fn(&T, &T) -> bool) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| equal(a, b))
}

fn pass_index(pass: GraphPass) -> usize {
    pass.1
}
fn buffer_key(handle: BufferHandle) -> (usize, u32) {
    (handle.resource, handle.version)
}
fn texture_key(handle: TextureHandle) -> (usize, u32) {
    (handle.resource, handle.version)
}
fn same_passes(a: &[GraphPass], b: &[GraphPass]) -> bool {
    same_items(a, b, |a, b| pass_index(*a) == pass_index(*b))
}

fn same_texture_desc(a: &TextureDesc, b: &TextureDesc) -> bool {
    // Exhaustive destructuring makes a new RHI descriptor field a compile error.
    let TextureDesc {
        label: _,
        dimension,
        extent,
        array_layers,
        format,
        mip_levels,
        samples,
        usage,
        view_formats,
    } = a;
    *dimension == b.dimension
        && *extent == b.extent
        && *array_layers == b.array_layers
        && *format == b.format
        && *mip_levels == b.mip_levels
        && *samples == b.samples
        && *usage == b.usage
        && *view_formats == b.view_formats
}
