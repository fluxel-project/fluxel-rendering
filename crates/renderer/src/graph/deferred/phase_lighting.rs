//! Deferred lighting Graph access declarations.

use super::{phase::*, *};
use crate::frame_data::specular_view::SPECULAR_VIEW_BYTES;

impl DeferredLightingPass<'_, '_> {
    fn mark(&mut self, bit: u8) -> std::result::Result<(), ()> {
        if self.declared & bit == 0 {
            self.declared |= bit;
            Ok(())
        } else {
            Err(())
        }
    }

    pub(crate) fn read_gbuffer(&mut self) -> std::result::Result<(), ()> {
        self.mark(GBUFFER_READ)?;
        for (index, texture) in self.gbuffer.iter().enumerate() {
            self.pass
                .read_texture_with_use(
                    *texture,
                    if index == 2 {
                        whole_depth()
                    } else {
                        whole_color()
                    },
                    sampled_read().uses,
                    ImageLayout::ShaderReadOnly,
                    TextureUsage::TEXTURE_BINDING,
                )
                .map_err(|_| ())?;
        }
        for texture in [self.effects, self.diffuse_roughness, self.ambient_occlusion]
            .into_iter()
            .flatten()
        {
            self.pass
                .read_texture_with_use(
                    texture,
                    whole_color(),
                    sampled_read().uses,
                    ImageLayout::ShaderReadOnly,
                    TextureUsage::TEXTURE_BINDING,
                )
                .map_err(|_| ())?;
        }
        Ok(())
    }

    pub(crate) fn write_main_color(&mut self) -> std::result::Result<(), ()> {
        self.mark(COLOR)?;
        if let Some(target) = self.target {
            self.target = Some(
                self.pass
                    .write_texture_with_use(
                        target,
                        whole_color(),
                        color_write().uses,
                        ImageLayout::ColorAttachment,
                        TextureUsage::RENDER_ATTACHMENT,
                    )
                    .map_err(|_| ())?,
            );
        }
        Ok(())
    }

    pub(super) fn finish(
        &mut self,
    ) -> std::result::Result<Option<super::super::definition::TextureHandle>, ()> {
        if self.declared != GBUFFER_READ | COLOR {
            return Err(());
        }
        if let Some((buffer, range)) = &self.view {
            self.pass
                .read_with_use(
                    *buffer,
                    range.clone(),
                    lighting_uniform_read().uses,
                    BufferUsage::UNIFORM,
                )
                .map_err(|_| ())?;
        }
        if let Some(environment) = self.environment {
            self.pass
                .read_texture_with_use(
                    environment,
                    whole_color(),
                    sampled_read().uses,
                    ImageLayout::ShaderReadOnly,
                    TextureUsage::TEXTURE_BINDING,
                )
                .map_err(|_| ())?;
        }
        match (
            self.specular_radiance,
            self.specular_brdf_lut,
            self.specular_view_uniform,
        ) {
            (Some(radiance), Some(brdf_lut), Some(view_uniform)) => {
                self.pass
                    .read_texture_with_use(
                        radiance,
                        all_color_mips(self.specular_mip_levels.ok_or(())?),
                        sampled_read().uses,
                        ImageLayout::ShaderReadOnly,
                        TextureUsage::TEXTURE_BINDING,
                    )
                    .map_err(|_| ())?;
                self.pass
                    .read_texture_with_use(
                        brdf_lut,
                        one_color_mip(0),
                        sampled_read().uses,
                        ImageLayout::ShaderReadOnly,
                        TextureUsage::TEXTURE_BINDING,
                    )
                    .map_err(|_| ())?;
                self.pass
                    .read_with_use(
                        view_uniform,
                        0..SPECULAR_VIEW_BYTES,
                        uniform_read().uses,
                        BufferUsage::UNIFORM,
                    )
                    .map_err(|_| ())?;
            }
            (None, None, None) => {}
            _ => return Err(()),
        }
        self.pass.side_effect();
        Ok(self.target)
    }
}
