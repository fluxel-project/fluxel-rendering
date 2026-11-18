//! Bounded, aligned uniform bytes for the Forward preparation boundary.
use fluxel_material::DynamicValue;
use fluxel_rhi::api::{Error, Result};
use std::ops::Range;

pub(crate) struct ForwardUniformInput {
    pub(crate) transform: [[f32; 4]; 4],
    /// Forward visibility. G-buffer draws carry no fade and retain their own ABI.
    pub(crate) mesh_visibility: Option<crate::MeshVisibility>,
    pub(crate) normal_from_local: Option<[[f32; 4]; 4]>,
    pub(crate) world_from_local: Option<[[f32; 4]; 4]>,
    pub(crate) clip_from_world: Option<[[f32; 4]; 4]>,
    /// Values in the material binding schema's ascending binding order.
    pub(crate) material_uniforms: Vec<DynamicValue>,
    pub(crate) shading_camera: Option<crate::ShadingCamera>,
    pub(crate) directional_light: crate::DirectionalLight,
}

pub(crate) struct ForwardUniformData {
    pub(crate) bytes: Vec<u8>,
    pub(crate) transforms: Vec<Range<u64>>,
    /// Dynamic uniform ranges per draw, in material binding order.
    pub(crate) material_uniforms: Vec<Vec<Range<u64>>>,
}

// `PbrTransforms::normal_from_local` is only ever multiplied by directions
// with w=0.  Its fourth column is consequently outside the normal transform
// itself.  Keep the world orientation fact beside the two matrices so the
// fragment TBN path need not recompute a determinant in f32 (which can
// underflow for an otherwise representable inverse-transpose matrix).
const PBR_WORLD_HANDEDNESS_OFFSET: usize = 64 + 12 * 4;
// The second component of the PBR normal matrix metadata column.
const PBR_MESH_VISIBILITY_OFFSET: usize = PBR_WORLD_HANDEDNESS_OFFSET + 4;
pub(crate) const UNLIT_FORWARD_TRANSFORM_BYTES: u64 = 80;

fn world_handedness(world: [[f32; 4]; 4]) -> Result<f32> {
    let world = world.map(|column| column.map(f64::from));
    let (a, b, c) = (world[0][0], world[1][0], world[2][0]);
    let (d, e, f) = (world[0][1], world[1][1], world[2][1]);
    let (g, h, i) = (world[0][2], world[1][2], world[2][2]);
    let determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if !determinant.is_finite() || determinant == 0.0 {
        return Err(Error::InvalidRange);
    }
    Ok(if determinant.is_sign_negative() {
        -1.0
    } else {
        1.0
    })
}

const PBR_CAMERA_OFFSET: usize = 192;
const PBR_LIGHT_OFFSET: usize = PBR_CAMERA_OFFSET + 16;
const PBR_CLIP_FROM_WORLD_OFFSET: usize = PBR_LIGHT_OFFSET + 32;
pub(crate) const PBR_TRANSFORM_BYTES: u64 = PBR_CLIP_FROM_WORLD_OFFSET as u64;
pub(crate) const PBR_SCENE_REFRACTION_TRANSFORM_BYTES: u64 = PBR_CLIP_FROM_WORLD_OFFSET as u64 + 64;

fn transform_bytes(input: &ForwardUniformInput) -> Result<u64> {
    match (
        input.normal_from_local.is_some(),
        input.world_from_local.is_some(),
        input.clip_from_world.is_some(),
    ) {
        (false, false, false) if input.mesh_visibility.is_some() => {
            Ok(UNLIT_FORWARD_TRANSFORM_BYTES)
        }
        (false, false, false) => Ok(64),
        (true, true, false) => Ok(PBR_TRANSFORM_BYTES),
        (true, true, true) => Ok(PBR_SCENE_REFRACTION_TRANSFORM_BYTES),
        _ => Err(Error::InvalidRange),
    }
}

/// Every dynamic binding uses an individually aligned physical start.
pub(crate) fn pack_forward_uniforms(
    inputs: &[ForwardUniformInput],
    alignment: u64,
    max_bytes: u64,
) -> Result<ForwardUniformData> {
    if alignment == 0 {
        return Err(Error::InvalidRange);
    }
    let align = |value: u64| {
        value
            .checked_add(alignment - 1)
            .map(|value| value / alignment * alignment)
            .ok_or(Error::InvalidRange)
    };
    let mut transforms = Vec::new();
    transforms
        .try_reserve_exact(inputs.len())
        .map_err(|_| Error::OutOfMemory)?;
    let mut material_uniforms = Vec::new();
    material_uniforms
        .try_reserve_exact(inputs.len())
        .map_err(|_| Error::OutOfMemory)?;
    let mut total: u64 = 0;
    for input in inputs {
        let transform_end = total
            .checked_add(transform_bytes(input)?)
            .ok_or(Error::InvalidRange)?;
        let mut ranges = Vec::new();
        ranges
            .try_reserve_exact(input.material_uniforms.len())
            .map_err(|_| Error::OutOfMemory)?;
        let mut uniform_end = transform_end;
        for value in &input.material_uniforms {
            let start = align(uniform_end)?;
            let end = start
                .checked_add(value.ty().uniform_binding_size())
                .ok_or(Error::InvalidRange)?;
            ranges.push(start..end);
            uniform_end = end;
        }
        transforms.push(total..transform_end);
        material_uniforms.push(ranges);
        total = align(uniform_end)?;
    }
    let total = total.max(16);
    if total > max_bytes || total > u64::from(u32::MAX) {
        return Err(Error::InvalidRange);
    }
    if inputs.iter().any(|input| {
        input
            .transform
            .iter()
            .flatten()
            .chain(
                input
                    .normal_from_local
                    .iter()
                    .flat_map(|matrix| matrix.iter().flatten()),
            )
            .chain(
                input
                    .world_from_local
                    .iter()
                    .flat_map(|matrix| matrix.iter().flatten()),
            )
            .chain(
                input
                    .clip_from_world
                    .iter()
                    .flat_map(|matrix| matrix.iter().flatten()),
            )
            .chain(
                input
                    .material_uniforms
                    .iter()
                    .flat_map(DynamicValue::components),
            )
            .any(|value| !value.is_finite())
    }) {
        return Err(Error::InvalidRange);
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(usize::try_from(total).map_err(|_| Error::InvalidRange)?)
        .map_err(|_| Error::OutOfMemory)?;
    bytes.resize(usize::try_from(total).map_err(|_| Error::InvalidRange)?, 0);
    for ((input, transform), uniforms) in inputs.iter().zip(&transforms).zip(&material_uniforms) {
        let start = transform.start;
        for (component, value) in input.transform.iter().flatten().enumerate() {
            let offset = start as usize + component * 4;
            bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        if let Some(normal) = input.normal_from_local {
            for (component, value) in normal.iter().flatten().enumerate() {
                let offset = start as usize + 64 + component * 4;
                bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            }
            let world = input.world_from_local.ok_or(Error::InvalidRange)?;
            let offset = start as usize + PBR_WORLD_HANDEDNESS_OFFSET;
            bytes[offset..offset + 4].copy_from_slice(&world_handedness(world)?.to_le_bytes());
            let camera = input
                .shading_camera
                .map_or([0.0, 0.0, 1.0, 0.0], crate::ShadingCamera::uniform);
            for (component, value) in camera.into_iter().enumerate() {
                let offset = start as usize + PBR_CAMERA_OFFSET + component * 4;
                bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            }
            for (component, value) in input.directional_light.uniform().into_iter().enumerate() {
                let offset = start as usize + PBR_LIGHT_OFFSET + component * 4;
                bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            }
            if let Some(clip_from_world) = input.clip_from_world {
                for (component, value) in clip_from_world.iter().flatten().enumerate() {
                    let offset = start as usize + PBR_CLIP_FROM_WORLD_OFFSET + component * 4;
                    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
                }
            }
        }
        if let Some(world) = input.world_from_local {
            for (component, value) in world.iter().flatten().enumerate() {
                let offset = start as usize + 128 + component * 4;
                bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            }
        }
        if let Some(visibility) = input.mesh_visibility {
            let offset = start as usize
                + if input.normal_from_local.is_some() {
                    PBR_MESH_VISIBILITY_OFFSET
                } else {
                    64
                };
            bytes[offset..offset + 4].copy_from_slice(&visibility.get().to_le_bytes());
        }
        for (value, range) in input.material_uniforms.iter().zip(uniforms) {
            for (component, value) in value.components().iter().enumerate() {
                let offset = range.start as usize + component * 4;
                bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            }
        }
    }
    Ok(ForwardUniformData {
        bytes,
        transforms,
        material_uniforms,
    })
}

#[cfg(test)]
#[path = "../tests/frame_data/forward.rs"]
mod tests;

#[cfg(test)]
#[path = "../tests/frame_data/forward_handedness.rs"]
mod handedness_tests;
