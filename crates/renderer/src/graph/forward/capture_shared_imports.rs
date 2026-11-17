//! Cross-namespace reuse for completed resident allocations.

use super::*;
use crate::residency::{ForwardTexture, ForwardVertexStream};

impl SharedResidentImports {
    /// Imports or reuses a completed resident stream from a Deferred namespace.
    /// Upload products must keep their own Graph version.
    pub(in crate::graph) fn deferred_stream<B: ResourceBackend>(
        &mut self,
        graph: &mut GraphBuilder,
        device: fluxel_rhi::api::DeviceId,
        owner: &ForwardVertexStream<B>,
        uploaded: bool,
    ) -> Result<crate::graph::definition::BufferHandle> {
        if owner.buffer.device() != device {
            return Err(Error::WrongDevice);
        }
        if uploaded {
            return graph
                .import_buffer_desc(owner.desc.clone(), owner.incoming)
                .map_err(|_| Error::InvalidState);
        }
        if let Some((handle, saved)) = self.streams.get(&owner.instance) {
            if !same_buffer_owner_desc(&saved.desc, &owner.desc) || saved.incoming != owner.incoming
            {
                return Err(Error::InvalidState);
            }
            return Ok(*handle);
        }
        let handle = graph
            .import_buffer_desc(owner.desc.clone(), owner.incoming)
            .map_err(|_| Error::InvalidState)?;
        self.streams.insert(
            owner.instance,
            (
                handle,
                BufferGraphFact {
                    desc: owner.desc.clone(),
                    incoming: owner.incoming,
                },
            ),
        );
        Ok(handle)
    }

    /// Imports or reuses a completed resident texture from a Deferred namespace.
    /// Upload and fresh-mip products deliberately stay separate versions.
    pub(in crate::graph) fn deferred_texture<B: ResourceBackend>(
        &mut self,
        graph: &mut GraphBuilder,
        device: fluxel_rhi::api::DeviceId,
        owner: &ForwardTexture<B>,
        uploaded: bool,
        fresh_mips: bool,
    ) -> Result<crate::graph::definition::TextureHandle> {
        if owner.device != device
            || owner.texture.device() != device
            || owner.view.device() != device
        {
            return Err(Error::WrongDevice);
        }
        if uploaded || fresh_mips {
            return (if fresh_mips {
                graph.create_texture(owner.desc.clone())
            } else {
                graph.import_texture_with_state(owner.desc.clone(), owner.incoming, owner.layout)
            })
            .map_err(|_| Error::InvalidState);
        }
        if let Some((handle, saved)) = self.textures.get(&owner.instance) {
            if !same_texture_owner_desc(&saved.desc, &owner.desc)
                || saved.incoming != owner.incoming
                || saved.layout != owner.layout
            {
                return Err(Error::InvalidState);
            }
            return Ok(*handle);
        }
        let handle = graph
            .import_texture_with_state(owner.desc.clone(), owner.incoming, owner.layout)
            .map_err(|_| Error::InvalidState)?;
        self.textures.insert(
            owner.instance,
            (
                handle,
                TextureGraphFact {
                    desc: owner.desc.clone(),
                    incoming: owner.incoming,
                    layout: owner.layout,
                },
            ),
        );
        Ok(handle)
    }
}
