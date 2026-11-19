//! Asset, geometry, and template validation for Forward preparation.

use fluxel_assets::{AssetKind, AssetSnapshot};
use fluxel_rhi::api::{VertexFormat, VertexStepMode};
use fluxel_shader::ShaderCompositionRequest;

use crate::{RenderTransform, SceneAssetRef};

use super::{
    ColorTextureArrayData, ColorTextureData, ColorTextureVolumeData, DepthTextureArrayData,
    DepthTextureData, ForwardDrawSource, ForwardPreparationError, GeometryData, GeometryLayout,
    PreparedMaterial,
};
use crate::preparation::forward_templates;

pub(super) fn resolve_asset<K: AssetKind, V>(
    snapshots: &[AssetSnapshot<K, V>],
    reference: SceneAssetRef<K>,
    asset: &'static str,
    source: ForwardDrawSource,
) -> Result<AssetSnapshot<K, V>, ForwardPreparationError> {
    let mut matches = snapshots.iter().filter(|snapshot| {
        snapshot.id() == reference.id() && snapshot.generation() == reference.generation()
    });
    let Some(snapshot) = matches.next() else {
        return Err(ForwardPreparationError::MissingAsset { asset, source });
    };
    if matches.next().is_some() {
        return Err(ForwardPreparationError::DuplicateAsset { asset, source });
    }
    Ok(snapshot.clone())
}

pub(super) fn validate_texture(
    texture: &ColorTextureData,
    source: ForwardDrawSource,
) -> Result<(), ForwardPreparationError> {
    let [width, height] = texture.extent;
    let mip_levels = texture.mip_levels;
    if width == 0 || height == 0 || mip_levels == 0 || mip_levels > 1 + width.max(height).ilog2() {
        return Err(ForwardPreparationError::InvalidTexture { source });
    }
    let bytes = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|texels| texels.checked_mul(4))
        .and_then(|bytes| usize::try_from(bytes).ok());
    if bytes != Some(texture.rgba8.len()) {
        return Err(ForwardPreparationError::InvalidTexture { source });
    }
    Ok(())
}

pub(super) fn validate_depth_texture(
    texture: &DepthTextureData,
    source: ForwardDrawSource,
) -> Result<(), ForwardPreparationError> {
    texture
        .is_valid()
        .then_some(())
        .ok_or(ForwardPreparationError::InvalidTexture { source })
}

pub(super) fn validate_depth_texture_array(
    texture: &DepthTextureArrayData,
    source: ForwardDrawSource,
) -> Result<(), ForwardPreparationError> {
    texture
        .is_valid()
        .then_some(())
        .ok_or(ForwardPreparationError::InvalidTexture { source })
}

pub(super) fn validate_texture_array(
    texture: &ColorTextureArrayData,
    source: ForwardDrawSource,
) -> Result<(), ForwardPreparationError> {
    texture
        .mip_payloads()
        .is_some()
        .then_some(())
        .ok_or(ForwardPreparationError::InvalidTexture { source })
}

pub(super) fn validate_texture_volume(
    texture: &ColorTextureVolumeData,
    source: ForwardDrawSource,
) -> Result<(), ForwardPreparationError> {
    texture
        .mip_payloads()
        .is_some()
        .then_some(())
        .ok_or(ForwardPreparationError::InvalidTexture { source })
}

/// Validates a base-only array payload and a complete generated mip target.
pub(super) fn validate_generated_texture_array(
    texture: &ColorTextureArrayData,
    mip_levels: u32,
    source: ForwardDrawSource,
) -> Result<(), ForwardPreparationError> {
    validate_texture_array(texture, source)?;
    let maximum_mips = 1 + texture.extent[0].max(texture.extent[1]).ilog2();
    if texture.mip_levels != 1 || !(2..=maximum_mips).contains(&mip_levels) {
        return Err(ForwardPreparationError::InvalidTexture { source });
    }
    Ok(())
}

/// Validates the packed six-face cube convention used by Forward.
pub(super) fn validate_cube_texture_array(
    texture: &ColorTextureArrayData,
    source: ForwardDrawSource,
) -> Result<(), ForwardPreparationError> {
    validate_texture_array(texture, source)?;
    if texture.extent[0] != texture.extent[1] || texture.layers != 6 {
        return Err(ForwardPreparationError::InvalidTexture { source });
    }
    Ok(())
}

/// Validates a base-only packed cube and the requested generated mip count.
pub(super) fn validate_generated_cube_texture_array(
    texture: &ColorTextureArrayData,
    mip_levels: u32,
    source: ForwardDrawSource,
) -> Result<(), ForwardPreparationError> {
    validate_cube_texture_array(texture, source)?;
    let maximum_mips = 1 + texture.extent[0].ilog2();
    if texture.mip_levels != 1 || !(2..=maximum_mips).contains(&mip_levels) {
        return Err(ForwardPreparationError::InvalidTexture { source });
    }
    Ok(())
}

pub(super) fn validate_cpu_geometry(
    geometry: &GeometryData,
    prepared: &crate::PreparedGeometry,
    source: ForwardDrawSource,
) -> Result<(), ForwardPreparationError> {
    if geometry.vertices.start > geometry.vertices.end {
        return Err(ForwardPreparationError::InvalidVertexRange { source });
    }
    if geometry.streams.len() != prepared.streams.len() {
        return Err(ForwardPreparationError::GeometryStreamCount {
            source,
            expected: prepared.streams.len(),
            actual: geometry.streams.len(),
        });
    }
    if geometry.vertices.is_empty() {
        return Ok(());
    }
    for (stream_index, (layout, bytes)) in
        prepared.streams.iter().zip(&geometry.streams).enumerate()
    {
        if !matches!(layout.step_mode, VertexStepMode::Vertex) {
            return Err(ForwardPreparationError::InstanceStreamUnsupported {
                source,
                stream: stream_index,
            });
        }
        let attribute_end = layout
            .attributes
            .iter()
            .map(|attribute| {
                forward_format_size(attribute.format)
                    .and_then(|size| attribute.offset.checked_add(size))
            })
            .collect::<Option<Vec<_>>>()
            .and_then(|ends| ends.into_iter().max());
        let Some(attribute_end) = attribute_end else {
            continue;
        };
        let required_end = u64::from(geometry.vertices.end - 1)
            .checked_mul(layout.stride)
            .and_then(|offset| offset.checked_add(attribute_end))
            .and_then(|end| usize::try_from(end).ok())
            .ok_or(ForwardPreparationError::GeometryPayloadTooShort {
                source,
                stream: stream_index,
            })?;
        if required_end > bytes.len() {
            return Err(ForwardPreparationError::GeometryPayloadTooShort {
                source,
                stream: stream_index,
            });
        }
    }
    Ok(())
}

pub(super) fn validate_material_schema(
    prepared: &PreparedMaterial,
    source: ForwardDrawSource,
) -> Result<crate::preparation::MaterialBindingSchema, ForwardPreparationError> {
    crate::preparation::forward_schema::material_binding_schema(prepared, source)
}

#[derive(Clone, Copy)]
pub(super) enum ForwardTemplate {
    Position,
    PositionUv0,
    PositionUvColor,
}

impl ForwardTemplate {
    pub(super) fn request(self) -> ShaderCompositionRequest {
        match self {
            Self::Position => forward_templates::position_request(),
            Self::PositionUv0 => forward_templates::position_uv0_request(),
            Self::PositionUvColor => forward_templates::position_uv_color_request(),
        }
    }
}

pub(super) fn forward_template(
    layout: &GeometryLayout,
    source: ForwardDrawSource,
) -> Result<ForwardTemplate, ForwardPreparationError> {
    let position = layout
        .streams
        .iter()
        .flat_map(|stream| &stream.attributes)
        .any(|attribute| attribute.semantic == "POSITION");
    let uv = layout
        .streams
        .iter()
        .flat_map(|stream| &stream.attributes)
        .any(|attribute| attribute.semantic == "UV");
    let color = layout
        .streams
        .iter()
        .flat_map(|stream| &stream.attributes)
        .any(|attribute| attribute.semantic == "COLOR");
    if !position {
        return Err(ForwardPreparationError::MissingPosition { source });
    }
    match (uv, color) {
        (false, false) => Ok(ForwardTemplate::Position),
        (true, true) => Ok(ForwardTemplate::PositionUvColor),
        _ => Err(ForwardPreparationError::IncompleteOptionalAttributes { source }),
    }
}

/// Selects the unlit UV0 template without accepting the legacy `UV`/`COLOR`
/// pair. The named semantic is part of the material/geometry contract.
pub(super) fn color_uv0_template(
    layout: &GeometryLayout,
    source: ForwardDrawSource,
) -> Result<ForwardTemplate, ForwardPreparationError> {
    let mut position = false;
    let mut uv0 = false;
    for attribute in layout.streams.iter().flat_map(|stream| &stream.attributes) {
        position |= attribute.semantic == "POSITION";
        uv0 |= attribute.semantic == "UV0";
    }
    if !position {
        return Err(ForwardPreparationError::MissingPosition { source });
    }
    if !uv0 {
        return Err(ForwardPreparationError::MissingUv0 { source });
    }
    Ok(ForwardTemplate::PositionUv0)
}

pub(super) fn finite_transform(transform: RenderTransform) -> bool {
    transform
        .columns()
        .iter()
        .flatten()
        .all(|value| value.is_finite())
}

pub(super) fn multiply_columns(left: RenderTransform, right: RenderTransform) -> [[f32; 4]; 4] {
    let left = left.columns();
    let right = right.columns();
    let mut product = [[0.0; 4]; 4];
    for column in 0..4 {
        for row in 0..4 {
            product[column][row] = (0..4)
                .map(|inner| left[inner][row] * right[column][inner])
                .sum();
        }
    }
    product
}

fn forward_format_size(format: VertexFormat) -> Option<u64> {
    let (bytes, components): (u64, u8) = match format {
        VertexFormat::Float32(n) | VertexFormat::Uint32(n) | VertexFormat::Sint32(n) => (4, n),
        VertexFormat::Float16(n)
        | VertexFormat::Uint16(n)
        | VertexFormat::Sint16(n)
        | VertexFormat::Unorm16(n)
        | VertexFormat::Snorm16(n) => (2, n),
        VertexFormat::Uint8(n)
        | VertexFormat::Sint8(n)
        | VertexFormat::Unorm8(n)
        | VertexFormat::Snorm8(n) => (1, n),
        VertexFormat::Float64(n) | VertexFormat::Uint64(n) => (8, n),
    };
    (1..=4)
        .contains(&components)
        .then_some(bytes * u64::from(components))
}
