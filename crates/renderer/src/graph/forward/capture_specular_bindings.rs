//! Concrete specular bindings for each independent capture frame namespace.
use super::*;

pub(super) fn append<'a, B: ResourceBackend>(
    slots: &CaptureFrameSlots,
    owner: Option<&'a crate::residency::SpecularEnvironmentFrameOwner<B>>,
    device: DeviceId,
    buffers: &mut Vec<BoundBuffer<'a, B>>,
    textures: &mut Vec<BoundTexture<'a, B>>,
) -> Result<()> {
    match (
        owner,
        slots.specular_upload,
        slots.specular_view_uniform,
        slots.specular_radiance,
        slots.specular_brdf_lut,
    ) {
        (None, None, None, None, None) => Ok(()),
        (Some(owner), Some(upload), Some(uniform), Some(radiance), Some(lut)) => {
            owner.validate_for_sampling(device)?;
            buffers.push(BoundBuffer::new(
                upload,
                owner.upload_instance,
                &owner.upload,
                &owner.upload_desc,
                owner.upload_incoming,
            ));
            buffers.push(BoundBuffer::new(
                uniform,
                owner.view_uniform_instance,
                &owner.view_uniform,
                &owner.view_uniform_desc,
                owner.view_uniform_incoming,
            ));
            textures.push(BoundTexture::new(
                radiance,
                owner.radiance.instance,
                &owner.radiance.texture,
                &owner.radiance.desc,
                owner.radiance.incoming,
                owner.radiance.layout,
            ));
            textures.push(BoundTexture::new(
                lut,
                owner.brdf_lut.instance,
                &owner.brdf_lut.texture,
                &owner.brdf_lut.desc,
                owner.brdf_lut.incoming,
                owner.brdf_lut.layout,
            ));
            Ok(())
        }
        _ => Err(Error::InvalidState),
    }
}
