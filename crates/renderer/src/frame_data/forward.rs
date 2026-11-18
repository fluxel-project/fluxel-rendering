//! Bounded, aligned uniform bytes for the Forward preparation boundary.
use fluxel_rhi::api::{Error, Result};
use std::ops::Range;

pub(crate) struct ForwardUniformInput {
    pub(crate) transform: [[f32; 4]; 4],
    pub(crate) normal_from_local: Option<[[f32; 4]; 4]>,
    pub(crate) world_from_local: Option<[[f32; 4]; 4]>,
    pub(crate) clip_from_world: Option<[[f32; 4]; 4]>,
    pub(crate) tint: [f32; 4],
    pub(crate) shading_camera: Option<crate::ShadingCamera>,
    pub(crate) directional_light: crate::DirectionalLight,
}

pub(crate) struct ForwardUniformData {
    pub(crate) bytes: Vec<u8>,
    pub(crate) transforms: Vec<Range<u64>>,
    pub(crate) tints: Vec<Range<u64>>,
}

// `PbrTransforms::normal_from_local` is only ever multiplied by directions
// with w=0.  Its fourth column is consequently outside the normal transform
// itself.  Keep the world orientation fact beside the two matrices so the
// fragment TBN path need not recompute a determinant in f32 (which can
// underflow for an otherwise representable inverse-transpose matrix).
const PBR_WORLD_HANDEDNESS_OFFSET: usize = 64 + 12 * 4;

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
        (false, false, false) => Ok(64),
        (true, true, false) => Ok(PBR_TRANSFORM_BYTES),
        (true, true, true) => Ok(PBR_SCENE_REFRACTION_TRANSFORM_BYTES),
        _ => Err(Error::InvalidRange),
    }
}

/// Both dynamic bindings use individually aligned physical starts.
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
    let mut tints = Vec::new();
    tints
        .try_reserve_exact(inputs.len())
        .map_err(|_| Error::OutOfMemory)?;
    let mut total: u64 = 0;
    for input in inputs {
        let transform_end = total
            .checked_add(transform_bytes(input)?)
            .ok_or(Error::InvalidRange)?;
        let tint_start = align(transform_end)?;
        let tint_end = tint_start.checked_add(16).ok_or(Error::InvalidRange)?;
        transforms.push(total..transform_end);
        tints.push(tint_start..tint_end);
        total = align(tint_end)?;
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
            .chain(input.tint.iter())
            .any(|value| !value.is_finite())
    }) {
        return Err(Error::InvalidRange);
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(usize::try_from(total).map_err(|_| Error::InvalidRange)?)
        .map_err(|_| Error::OutOfMemory)?;
    bytes.resize(usize::try_from(total).map_err(|_| Error::InvalidRange)?, 0);
    for ((input, transform), tint) in inputs.iter().zip(&transforms).zip(&tints) {
        let start = transform.start;
        let tint = tint.start;
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
        for (component, value) in input.tint.iter().enumerate() {
            let offset = tint as usize + component * 4;
            bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
    }
    Ok(ForwardUniformData {
        bytes,
        transforms,
        tints,
    })
}

#[cfg(test)]
#[path = "../tests/frame_data/forward.rs"]
mod tests;

#[cfg(test)]
#[path = "../tests/frame_data/forward_handedness.rs"]
mod handedness_tests;
