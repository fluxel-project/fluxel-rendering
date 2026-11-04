//! Bounded, aligned uniform bytes for the Forward preparation boundary.
use fluxel_rhi::api::{Error, Result};
use std::ops::Range;

pub(crate) struct ForwardUniformInput {
    pub(crate) transform: [[f32; 4]; 4],
    pub(crate) tint: [f32; 4],
}

pub(crate) struct ForwardUniformData {
    pub(crate) bytes: Vec<u8>,
    pub(crate) transforms: Vec<Range<u64>>,
    pub(crate) tints: Vec<Range<u64>>,
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
    let tint_offset = align(64)?;
    let stride = align(tint_offset.checked_add(16).ok_or(Error::InvalidRange)?)?;
    let count = u64::try_from(inputs.len()).map_err(|_| Error::InvalidRange)?;
    let total = stride
        .checked_mul(count)
        .ok_or(Error::InvalidRange)?
        .max(16);
    if total > max_bytes || total > u64::from(u32::MAX) {
        return Err(Error::InvalidRange);
    }
    if inputs.iter().any(|input| {
        input
            .transform
            .iter()
            .flatten()
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
    let mut transforms = Vec::new();
    transforms
        .try_reserve_exact(inputs.len())
        .map_err(|_| Error::OutOfMemory)?;
    let mut tints = Vec::new();
    tints
        .try_reserve_exact(inputs.len())
        .map_err(|_| Error::OutOfMemory)?;
    for (index, input) in inputs.iter().enumerate() {
        let start = u64::try_from(index).map_err(|_| Error::InvalidRange)? * stride;
        let tint = start + tint_offset;
        for (component, value) in input.transform.iter().flatten().enumerate() {
            let offset = start as usize + component * 4;
            bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        for (component, value) in input.tint.iter().enumerate() {
            let offset = tint as usize + component * 4;
            bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        transforms.push(start..start + 64);
        tints.push(tint..tint + 16);
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
