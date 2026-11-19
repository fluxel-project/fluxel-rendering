//! Particle billboard matrices and fade parameters in the Forward uniform ABI.

use fluxel_rhi::api::{Error, Result};

use crate::{
    MeshVisibility, RenderView, feature::particles::GpuParticleView,
    preparation::particle_material::PARTICLE_TRANSFORM_BYTES,
};

pub(crate) fn pack(
    view: &RenderView,
    camera: GpuParticleView,
    visibility: MeshVisibility,
) -> Result<[u8; PARTICLE_TRANSFORM_BYTES as usize]> {
    if !camera
        .translation_pivot
        .iter()
        .chain(&camera.dead_color)
        .all(|value| value.is_finite())
    {
        return Err(Error::InvalidRange);
    }
    let view_from_world = camera.view_from_world.columns();
    let world_from_view = super::specular_view::inverse(view_from_world)?;
    let [pivot_x, pivot_y] = camera.translation_pivot;
    let fields = view
        .transform
        .columns()
        .into_iter()
        .flatten()
        .chain(view_from_world.into_iter().flatten())
        .chain(world_from_view.into_iter().flatten())
        .chain([visibility.get(), 0.0, 0.0, 0.0])
        .chain([pivot_x, pivot_y, 0.0, 0.0])
        .chain(camera.dead_color);
    let mut bytes = [0; PARTICLE_TRANSFORM_BYTES as usize];
    for (slot, value) in bytes.chunks_exact_mut(4).zip(fields) {
        slot.copy_from_slice(&value.to_le_bytes());
    }
    Ok(bytes)
}
