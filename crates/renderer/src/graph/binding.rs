//! Pre-record validation for concrete buffer instances bound to a compiled graph.

use std::{
    collections::{HashMap, HashSet},
    ops::Range,
    sync::atomic::{AtomicU64, Ordering},
};

use fluxel_rhi::api::{
    Buffer, BufferDesc, BufferUsage, BufferUses, DeviceId, ImageLayout, MemoryClass, Texture,
    TextureDesc, TextureUses,
};

use super::definition::{
    AccessKind, BufferAccessContract, BufferHandle, GraphId, TextureAccessContract, TextureHandle,
    TextureRange,
};

static NEXT_RESOURCE_INSTANCE: AtomicU64 = AtomicU64::new(1);

/// Identity for one Renderer-created resource realization.
///
/// The identity is assigned alongside the concrete RHI handle and is never
/// derived from a backend-native object. It is private to one Renderer process.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct ResourceInstanceId(u64);

impl ResourceInstanceId {
    /// Issues a distinct identity for a newly allocated concrete resource.
    pub(crate) fn fresh() -> Result<Self, BindingError> {
        NEXT_RESOURCE_INSTANCE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map(Self)
            .map_err(|_| BindingError::IdentityExhausted)
    }
}

/// Stable graph-local slot for one logical buffer independent of its versions.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct BufferSlot {
    graph: GraphId,
    pub(crate) resource: usize,
}

impl From<BufferHandle> for BufferSlot {
    fn from(handle: BufferHandle) -> Self {
        Self {
            graph: handle.graph,
            resource: handle.resource,
        }
    }
}

/// Stable graph-local slot for one logical texture independent of its versions.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct TextureSlot {
    graph: GraphId,
    pub(crate) resource: usize,
}

impl From<TextureHandle> for TextureSlot {
    fn from(handle: TextureHandle) -> Self {
        Self {
            graph: handle.graph,
            resource: handle.resource,
        }
    }
}

/// A concrete buffer instance and the descriptor/state facts known by its owner.
#[derive(Clone, Debug)]
pub(crate) struct BufferBinding {
    pub(crate) slot: BufferSlot,
    pub(crate) instance: ResourceInstanceId,
    pub(crate) device: DeviceId,
    pub(crate) size: u64,
    pub(crate) usage: BufferUsage,
    pub(crate) memory: MemoryClass,
    pub(crate) initial: BufferUses,
}

impl BufferBinding {
    /// Builds facts from the exact RHI handle and descriptor used to create it.
    pub(crate) fn new<B: fluxel_rhi::backend::Backend>(
        slot: BufferSlot,
        instance: ResourceInstanceId,
        buffer: &Buffer<B>,
        desc: &BufferDesc,
        initial: BufferUses,
    ) -> Self {
        Self {
            slot,
            instance,
            device: buffer.device(),
            size: desc.size,
            usage: desc.usage,
            memory: desc.memory,
            initial,
        }
    }
}

/// A concrete texture instance and its owner-known descriptor/incoming state.
#[derive(Clone, Debug)]
pub(crate) struct TextureBinding {
    pub(crate) slot: TextureSlot,
    pub(crate) instance: ResourceInstanceId,
    pub(crate) device: DeviceId,
    pub(crate) desc: TextureDesc,
    pub(crate) initial: TextureUses,
    pub(crate) layout: ImageLayout,
}

impl TextureBinding {
    pub(crate) fn new<B: fluxel_rhi::backend::Backend>(
        slot: TextureSlot,
        instance: ResourceInstanceId,
        texture: &Texture<B>,
        desc: &TextureDesc,
        initial: TextureUses,
        layout: ImageLayout,
    ) -> Self {
        Self {
            slot,
            instance,
            device: texture.device(),
            desc: desc.clone(),
            initial,
            layout,
        }
    }
}

/// Buffer requirements compiled from live graph accesses.
#[derive(Clone, Debug)]
pub(crate) struct BufferRequirement {
    pub(crate) slot: BufferSlot,
    pub(crate) size: u64,
    pub(crate) usage: BufferUsage,
    pub(crate) memory: MemoryClass,
    pub(crate) initial: Option<BufferUses>,
    pub(crate) ranges: Vec<Range<u64>>,
    pub(crate) accesses: Vec<(Range<u64>, AccessKind, Option<BufferAccessContract>)>,
    pub(crate) first_pass: usize,
    pub(crate) last_pass: usize,
}

/// The immutable buffer contract of one compiled graph.
#[derive(Clone, Debug, Default)]
pub(crate) struct BufferBindingManifest {
    requirements: Vec<BufferRequirement>,
}

/// Requirements for one live compiled-graph texture resource.
#[derive(Clone, Debug)]
pub(crate) struct TextureRequirement {
    pub(crate) slot: TextureSlot,
    pub(crate) desc: TextureDesc,
    pub(crate) initial: Option<(TextureUses, ImageLayout)>,
    pub(crate) ranges: Vec<TextureRange>,
    pub(crate) accesses: Vec<(TextureRange, AccessKind, Option<TextureAccessContract>)>,
    pub(crate) first_pass: usize,
    pub(crate) last_pass: usize,
}

/// Immutable texture contract of one compiled graph.
#[derive(Clone, Debug, Default)]
pub(crate) struct TextureBindingManifest {
    requirements: Vec<TextureRequirement>,
}

impl TextureBindingManifest {
    pub(crate) fn new(requirements: Vec<TextureRequirement>) -> Self {
        Self { requirements }
    }

    pub(crate) fn requirements(&self) -> &[TextureRequirement] {
        &self.requirements
    }

    pub(crate) fn validate(
        &self,
        device: DeviceId,
        bindings: &[TextureBinding],
    ) -> Result<(), BindingError> {
        let requirement_indices: HashMap<_, _> = self
            .requirements
            .iter()
            .enumerate()
            .map(|(index, requirement)| (requirement.slot, index))
            .collect();
        let mut slots = HashSet::with_capacity(bindings.len());
        let mut binding_indices = vec![None; self.requirements.len()];
        for (binding_index, binding) in bindings.iter().enumerate() {
            if !slots.insert(binding.slot) {
                return Err(BindingError::DuplicateSlot);
            }
            let requirement_index = requirement_indices
                .get(&binding.slot)
                .ok_or(BindingError::UndeclaredResource)?;
            binding_indices[*requirement_index] = Some(binding_index);
        }
        if bindings.len() != self.requirements.len() {
            return Err(BindingError::MissingResource);
        }
        for (requirement_index, requirement) in self.requirements.iter().enumerate() {
            let binding = binding_indices[requirement_index]
                .map(|binding_index| &bindings[binding_index])
                .ok_or(BindingError::MissingResource)?;
            if binding.device != device {
                return Err(BindingError::WrongDevice);
            }
            if !texture_desc_compatible(&binding.desc, &requirement.desc)
                || requirement.ranges.iter().any(|range| {
                    range.aspects.0 == 0
                        || range.mips.start >= range.mips.end
                        || range.mips.end > binding.desc.mip_levels
                        || range.layers.start >= range.layers.end
                        || range.layers.end > binding.desc.array_layers
                })
            {
                return Err(BindingError::IncompatibleDescriptor);
            }
            for (_, _, contract) in &requirement.accesses {
                let contract = contract.as_ref().ok_or(BindingError::UndeclaredUse)?;
                if !binding.desc.usage.contains(contract.descriptor_usage)
                    || contract.uses.stages.0 == 0
                    || contract.uses.accesses.0 == 0
                    || contract.layout == ImageLayout::Undefined
                {
                    return Err(BindingError::IncompatibleDescriptor);
                }
            }
            if requirement.initial != Some((binding.initial, binding.layout)) {
                return Err(BindingError::IncompatibleIncomingState);
            }
        }
        let mut first_requirement_for_instance = HashMap::with_capacity(self.requirements.len());
        let mut first_alias = None;
        for (requirement_index, binding_index) in binding_indices.iter().enumerate() {
            let binding_index = binding_index.ok_or(BindingError::MissingResource)?;
            let instance = bindings[binding_index].instance;
            if let Some(left) = first_requirement_for_instance.get(&instance) {
                let candidate = (*left, requirement_index);
                if first_alias.is_none_or(|first| candidate < first) {
                    first_alias = Some(candidate);
                }
            } else {
                first_requirement_for_instance.insert(instance, requirement_index);
            }
        }
        if let Some((left, right)) = first_alias {
            let left_requirement = &self.requirements[left];
            let right_requirement = &self.requirements[right];
            let lifetimes_overlap = left_requirement.first_pass <= right_requirement.last_pass
                && right_requirement.first_pass <= left_requirement.last_pass;
            if lifetimes_overlap {
                return Err(BindingError::AliasedLiveResources);
            }
            return Err(BindingError::AliasedResourcesRequireRecompile);
        }
        Ok(())
    }
}

fn texture_desc_compatible(actual: &TextureDesc, expected: &TextureDesc) -> bool {
    actual.dimension == expected.dimension
        && actual.extent == expected.extent
        && actual.array_layers == expected.array_layers
        && actual.format == expected.format
        && actual.mip_levels == expected.mip_levels
        && actual.samples == expected.samples
        && actual.usage.contains(expected.usage)
        && actual.view_formats == expected.view_formats
}

impl BufferBindingManifest {
    pub(crate) fn new(requirements: Vec<BufferRequirement>) -> Self {
        Self { requirements }
    }

    pub(crate) fn requirements(&self) -> &[BufferRequirement] {
        &self.requirements
    }

    /// Validates a complete frame binding set before mapping or command recording.
    pub(crate) fn validate(
        &self,
        device: DeviceId,
        bindings: &[BufferBinding],
    ) -> Result<(), BindingError> {
        let mut slots = HashSet::with_capacity(bindings.len());
        for binding in bindings {
            if !slots.insert(binding.slot) {
                return Err(BindingError::DuplicateSlot);
            }
            if !self
                .requirements
                .iter()
                .any(|requirement| requirement.slot == binding.slot)
            {
                return Err(BindingError::UndeclaredResource);
            }
        }
        if bindings.len() != self.requirements.len() {
            return Err(BindingError::MissingResource);
        }

        for requirement in &self.requirements {
            let binding = bindings
                .iter()
                .find(|binding| binding.slot == requirement.slot)
                .ok_or(BindingError::MissingResource)?;
            if binding.device != device {
                return Err(BindingError::WrongDevice);
            }
            if binding.size < requirement.size
                || !binding.usage.contains(requirement.usage)
                || binding.memory != requirement.memory
                || requirement
                    .ranges
                    .iter()
                    .any(|range| range.start >= range.end || range.end > binding.size)
            {
                return Err(BindingError::IncompatibleDescriptor);
            }
            for (_, _, contract) in &requirement.accesses {
                let contract = contract.as_ref().ok_or(BindingError::UndeclaredUse)?;
                if !binding.usage.contains(contract.descriptor_usage)
                    || contract.uses.stages.0 == 0
                    || contract.uses.accesses.0 == 0
                {
                    return Err(BindingError::IncompatibleDescriptor);
                }
            }
            if requirement.initial != Some(binding.initial) {
                return Err(BindingError::IncompatibleIncomingState);
            }
        }

        for left in 0..self.requirements.len() {
            for right in left + 1..self.requirements.len() {
                let left_requirement = &self.requirements[left];
                let right_requirement = &self.requirements[right];
                let left_binding = bindings
                    .iter()
                    .find(|binding| binding.slot == left_requirement.slot)
                    .ok_or(BindingError::MissingResource)?;
                let right_binding = bindings
                    .iter()
                    .find(|binding| binding.slot == right_requirement.slot)
                    .ok_or(BindingError::MissingResource)?;
                if left_binding.instance != right_binding.instance {
                    continue;
                }
                if left_binding.size != right_binding.size
                    || left_binding.usage != right_binding.usage
                    || left_binding.memory != right_binding.memory
                {
                    return Err(BindingError::IncompatibleAliasDescriptor);
                }
                let lifetimes_overlap = left_requirement.first_pass <= right_requirement.last_pass
                    && right_requirement.first_pass <= left_requirement.last_pass;
                if lifetimes_overlap {
                    return Err(BindingError::AliasedLiveResources);
                }
                return Err(BindingError::AliasedResourcesRequireRecompile);
            }
        }
        Ok(())
    }
}

/// A structured pre-record binding refusal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BindingError {
    IdentityExhausted,
    MissingResource,
    UndeclaredResource,
    DuplicateSlot,
    WrongDevice,
    IncompatibleDescriptor,
    IncompatibleIncomingState,
    UndeclaredUse,
    IncompatibleAliasDescriptor,
    AliasedLiveResources,
    AliasedResourcesRequireRecompile,
}

#[cfg(test)]
#[path = "../tests/graph/binding.rs"]
mod tests;
