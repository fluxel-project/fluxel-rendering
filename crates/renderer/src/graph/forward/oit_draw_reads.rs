//! The existing Forward read protocol applied to an actual OIT Graph pass.

use crate::frame_data::specular_view::SPECULAR_VIEW_BYTES;
use super::offscreen_prefix::ForwardDeclarationPrefix;
use super::*;
use crate::graph::definition::{GraphError, PassBuilder};

pub(super) struct OitDrawReads<'a> {
    streams: Vec<crate::graph::definition::BufferHandle>,
    uniform: crate::graph::definition::BufferHandle,
    textures: Vec<crate::graph::definition::TextureHandle>,
    environment: Option<crate::graph::definition::TextureHandle>,
    specular_radiance: Option<crate::graph::definition::TextureHandle>,
    specular_brdf_lut: Option<crate::graph::definition::TextureHandle>,
    specular_view_uniform: Option<crate::graph::definition::BufferHandle>,
    spec: &'a ForwardGraphSpec,
}

impl<'a> OitDrawReads<'a> {
    pub(super) fn new(prefix: &ForwardDeclarationPrefix, spec: &'a ForwardGraphSpec) -> Self {
        Self {
            streams: prefix.streams.clone(),
            uniform: prefix.uniform,
            textures: prefix.textures.clone(),
            environment: prefix.environment,
            specular_radiance: prefix.specular_radiance,
            specular_brdf_lut: prefix.specular_brdf_lut,
            specular_view_uniform: prefix.specular_view_uniform,
            spec,
        }
    }
    pub(super) fn from_capture_handles(
        handles: &super::capture_declaration::FrameHandles,
        spec: &'a ForwardGraphSpec,
    ) -> Self {
        Self {
            streams: handles.streams.clone(),
            uniform: handles.uniform,
            textures: handles.textures.clone(),
            environment: handles.environment,
            specular_radiance: handles.specular_radiance,
            specular_brdf_lut: handles.specular_brdf_lut,
            specular_view_uniform: handles.specular_view_uniform,
            spec,
        }
    }

    pub(super) fn declare_with(
        &self,
        pass: &mut PassBuilder<'_>,
        draws: Range<usize>,
        contribute: &mut impl FnMut(
            &mut ForwardOitReadContext<'_, '_>,
        ) -> std::result::Result<(), PipelineError>,
        failure: &mut Option<PipelineError>,
    ) -> std::result::Result<(), GraphError> {
        let has_draws = !draws.is_empty();
        let mut context = ForwardOitReadContext::new(
            pass,
            self.streams.clone(),
            self.uniform,
            self.textures.clone(),
            draws,
            &self.spec.draws,
        )
        .map_err(|_| GraphError::InvalidRange)?;
        if let Err(error) = contribute(&mut context) {
            *failure = Some(error);
            return Err(GraphError::InvalidRange);
        }
        if let Err(error) = context.finish() {
            *failure = Some(error);
            return Err(GraphError::InvalidRange);
        }
        if !has_draws {
            return Ok(());
        }
        if let Some(environment) = self.environment {
            pass.read_texture_with_use(
                environment,
                whole(),
                texture_read_uses().uses,
                ImageLayout::ShaderReadOnly,
                TextureUsage::TEXTURE_BINDING,
            )?;
        }
        match (
            self.specular_radiance,
            self.specular_brdf_lut,
            self.specular_view_uniform,
        ) {
            (Some(radiance), Some(lut), Some(view)) => {
                let levels = self
                    .spec
                    .specular_radiance
                    .as_ref()
                    .ok_or(GraphError::InvalidRange)?
                    .desc
                    .mip_levels;
                pass.read_texture_with_use(
                    radiance,
                    all_color_mips(levels),
                    texture_read_uses().uses,
                    ImageLayout::ShaderReadOnly,
                    TextureUsage::TEXTURE_BINDING,
                )?;
                pass.read_texture_with_use(
                    lut,
                    whole(),
                    texture_read_uses().uses,
                    ImageLayout::ShaderReadOnly,
                    TextureUsage::TEXTURE_BINDING,
                )?;
                pass.read_with_use(view, 0..SPECULAR_VIEW_BYTES, shader_read_uses().uses, BufferUsage::UNIFORM)?;
            }
            (None, None, None) => {}
            _ => return Err(GraphError::InvalidRange),
        }
        Ok(())
    }
}
