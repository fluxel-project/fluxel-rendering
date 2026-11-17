//! Concrete sampled-texture provenance checks shared by Graph routes.

use fluxel_rhi::{
    api::{DeviceId, Error, Result},
    backend::ResourceBackend,
};

/// Checks both the saved allocation provenance and the actual RHI handles.
///
/// A cached Graph declaration contains descriptors, not resource identity.
/// Reusing it therefore must reject an owner whose saved device fact no longer
/// agrees with the selected device even when its unchanged RHI handles still
/// belong to that device.
pub(super) fn validate_resident_texture_devices<B: ResourceBackend>(
    resident: &crate::residency::ForwardResidentBuffers<B>,
    device: DeviceId,
) -> Result<()> {
    if resident.device != device
        || resident.textures.iter().any(|texture| {
            texture.device != device
                || texture.texture.device() != device
                || texture.view.device() != device
        })
    {
        return Err(Error::WrongDevice);
    }
    Ok(())
}
