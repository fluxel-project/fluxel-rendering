//! Identity checks for command owners; these grant no pass access.

use super::*;

impl<B: Backend> FrameBindings<'_, B> {
    /// Resolves a checked alias to the original owner installed in this Graph.
    /// This preserves pointer-exact command guards when immutable residents
    /// are shared by independently assembled frame namespaces.
    pub(in crate::graph) fn canonical_buffer(
        &self,
        slot: BufferSlot,
        instance: ResourceInstanceId,
        owner: &Buffer<B>,
        desc: &BufferDesc,
        incoming: BufferUses,
    ) -> Result<&Buffer<B>> {
        let candidate = BufferBinding::new(slot, instance, owner, desc, incoming);
        let bound = self
            .buffers
            .iter()
            .find(|bound| bound.facts.slot == slot)
            .ok_or(Error::InvalidState)?;
        if !same_buffer_facts(&bound.facts, &candidate) {
            return Err(Error::InvalidState);
        }
        self.require_buffer_owner(slot, instance, bound.resource)?;
        Ok(bound.resource)
    }

    pub(in crate::graph) fn canonical_texture(
        &self,
        slot: TextureSlot,
        instance: ResourceInstanceId,
        owner: &Texture<B>,
        desc: &TextureDesc,
        incoming: TextureUses,
        layout: ImageLayout,
    ) -> Result<&Texture<B>> {
        let candidate = TextureBinding::new(slot, instance, owner, desc, incoming, layout);
        let bound = self
            .textures
            .iter()
            .find(|bound| bound.facts.slot == slot)
            .ok_or(Error::InvalidState)?;
        if !same_texture_facts(&bound.facts, &candidate) {
            return Err(Error::InvalidState);
        }
        self.require_texture_owner(slot, instance, bound.resource)?;
        Ok(bound.resource)
    }
    /// Checks that a raw command owner is the exact owner installed in this
    /// frame's validated binding. Callers still need CheckedPass authorization
    /// for every range and contract before using the resource.
    pub(crate) fn require_texture_owner(
        &self,
        slot: TextureSlot,
        instance: ResourceInstanceId,
        owner: &Texture<B>,
    ) -> Result<()> {
        if self.textures.iter().any(|bound| {
            bound.facts.slot == slot
                && bound.facts.instance == instance
                && std::ptr::eq(bound.resource, owner)
        }) {
            Ok(())
        } else {
            Err(Error::InvalidState)
        }
    }

    pub(crate) fn require_buffer_owner(
        &self,
        slot: BufferSlot,
        instance: ResourceInstanceId,
        owner: &Buffer<B>,
    ) -> Result<()> {
        if self.buffers.iter().any(|bound| {
            bound.facts.slot == slot
                && bound.facts.instance == instance
                && std::ptr::eq(bound.resource, owner)
        }) {
            Ok(())
        } else {
            Err(Error::InvalidState)
        }
    }
}

fn same_buffer_facts(left: &BufferBinding, right: &BufferBinding) -> bool {
    left.slot == right.slot
        && left.instance == right.instance
        && left.device == right.device
        && left.size == right.size
        && left.usage == right.usage
        && left.memory == right.memory
        && left.initial == right.initial
}

fn same_texture_facts(left: &TextureBinding, right: &TextureBinding) -> bool {
    let a = &left.desc;
    let b = &right.desc;
    left.slot == right.slot
        && left.instance == right.instance
        && left.device == right.device
        && left.initial == right.initial
        && left.layout == right.layout
        && left.kind == right.kind
        && a.dimension == b.dimension
        && a.extent == b.extent
        && a.array_layers == b.array_layers
        && a.format == b.format
        && a.mip_levels == b.mip_levels
        && a.samples == b.samples
        && a.usage == b.usage
        && a.view_formats == b.view_formats
}

/// Explicit namespace assembly; FrameBindings itself still rejects duplicate slots.
pub(in crate::graph) fn merge_buffer_bindings<B: Backend>(
    bindings: Vec<BoundBuffer<'_, B>>,
) -> Result<Vec<BoundBuffer<'_, B>>> {
    let mut output: Vec<BoundBuffer<'_, B>> = Vec::new();
    for binding in bindings {
        if let Some(previous) = output
            .iter()
            .find(|previous| previous.facts.slot == binding.facts.slot)
        {
            if !same_buffer_facts(&previous.facts, &binding.facts) {
                return Err(Error::InvalidState);
            }
        } else {
            output.push(binding);
        }
    }
    Ok(output)
}

pub(in crate::graph) fn merge_texture_bindings<B: Backend>(
    bindings: Vec<BoundTexture<'_, B>>,
) -> Result<Vec<BoundTexture<'_, B>>> {
    let mut output: Vec<BoundTexture<'_, B>> = Vec::new();
    for binding in bindings {
        if let Some(previous) = output
            .iter()
            .find(|previous| previous.facts.slot == binding.facts.slot)
        {
            if !same_texture_facts(&previous.facts, &binding.facts) {
                return Err(Error::InvalidState);
            }
        } else {
            output.push(binding);
        }
    }
    Ok(output)
}
