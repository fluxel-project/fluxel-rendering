//! Particle specialization of an already prepared Forward draw.

use fluxel_material::{SurfaceAlphaMode, SurfaceOutput};
use fluxel_rhi::api::{VertexFormat, VertexStepMode};
use fluxel_shader::ShaderComposer;

use super::{
    ForwardDrawInput, ForwardDrawSource, ForwardPreparationError, GeometryData,
    ParticleDrawPayload, ResolvedForwardTexture, prepare_material, validate_material_schema,
};

/// Re-specializes one ordinary unlit quad draw for GPU particle instances.
///
/// This consumes the existing material schema and keeps its textures, samplers
/// and dynamic values unchanged. The caller supplies a native particle stream
/// later, so this function validates CPU bytes only for the source quad.
pub(crate) fn prepare_particle_draw(
    input: &mut ForwardDrawInput,
    camera: crate::feature::particles::GpuParticleView,
    count: u32,
    composer: &mut ShaderComposer,
) -> Result<(), ForwardPreparationError> {
    if input.particle.is_some()
        || input.instance_transforms.is_some()
        || input.scene_refraction.is_some()
        || input
            .textures
            .iter()
            .any(|texture| matches!(texture, ResolvedForwardTexture::CaptureSlot(_)))
    {
        return Err(ForwardPreparationError::ParticleDrawUnsupported {
            source: input.source,
        });
    }
    let output = input.material.value().output();
    if !matches!(output, SurfaceOutput::Color | SurfaceOutput::ColorUv0)
        || !matches!(
            input.effective_alpha_mode(),
            SurfaceAlphaMode::Blend | SurfaceAlphaMode::MaskBlend
        )
    {
        return Err(ForwardPreparationError::ParticleDrawUnsupported {
            source: input.source,
        });
    }
    validate_particle_quad(input.geometry.value(), input.source, output)?;
    let (layout, request) =
        super::super::particle_material::request(output, &input.geometry.value().layout).map_err(
            |_| ForwardPreparationError::ParticleDrawUnsupported {
                source: input.source,
            },
        )?;
    let prepared = prepare_material(&input.material, &layout, composer, request)
        .map_err(ForwardPreparationError::Material)?;
    let schema = validate_material_schema(&prepared, input.source)?;
    if input.prepared.binding_schema() != Some(&schema) {
        return Err(ForwardPreparationError::MaterialBindingSchema {
            source: input.source,
        });
    }
    let transform = crate::frame_data::particle::pack(&input.view, camera, input.mesh_visibility)
        .map_err(|_| ForwardPreparationError::InvalidView)?;
    input.prepared = prepared.with_binding_schema(schema);
    input.normal_from_local = None;
    input.particle = Some(ParticleDrawPayload { count, transform });
    Ok(())
}

fn validate_particle_quad(
    geometry: &GeometryData,
    source: ForwardDrawSource,
    output: &SurfaceOutput,
) -> Result<(), ForwardPreparationError> {
    if geometry.layout.streams.len() != geometry.streams.len()
        || geometry.vertices.start > geometry.vertices.end
        || geometry.layout.streams.is_empty()
    {
        return Err(ForwardPreparationError::ParticleQuadLayout { source });
    }
    let mut position = false;
    let mut uv0 = false;
    for (stream, bytes) in geometry.layout.streams.iter().zip(&geometry.streams) {
        if !matches!(stream.step_mode, VertexStepMode::Vertex)
            || stream.stride == 0
            || u64::try_from(bytes.len())
                .ok()
                .is_none_or(|length| length % stream.stride != 0)
            || u64::from(geometry.vertices.end)
                .checked_mul(stream.stride)
                .is_none_or(|end| end > bytes.len() as u64)
        {
            return Err(ForwardPreparationError::ParticleQuadLayout { source });
        }
        for attribute in &stream.attributes {
            let valid_range = match attribute.format {
                VertexFormat::Float32(3) => attribute.offset.checked_add(12),
                VertexFormat::Float32(2) => attribute.offset.checked_add(8),
                _ => attribute.offset.checked_add(1),
            }
            .is_some_and(|end| end <= stream.stride);
            if !valid_range {
                return Err(ForwardPreparationError::ParticleQuadLayout { source });
            }
            match attribute.semantic.as_str() {
                "POSITION" if matches!(attribute.format, VertexFormat::Float32(3)) => {
                    if position {
                        return Err(ForwardPreparationError::ParticleQuadLayout { source });
                    }
                    position = true;
                }
                "UV0" if matches!(attribute.format, VertexFormat::Float32(2)) => {
                    if uv0 {
                        return Err(ForwardPreparationError::ParticleQuadLayout { source });
                    }
                    uv0 = true;
                }
                _ => {}
            }
        }
    }
    if !position || matches!(output, SurfaceOutput::ColorUv0) && !uv0 {
        return Err(ForwardPreparationError::ParticleQuadLayout { source });
    }
    Ok(())
}
