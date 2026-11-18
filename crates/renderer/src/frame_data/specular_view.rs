//! Exact camera uniform bytes for Deferred lighting and specular environments.

use fluxel_rhi::api::{Error, Result};

pub(crate) fn pack_specular_view(
    view: &crate::RenderView,
    max_lod: f32,
    lod_scale: f32,
    lod_offset: f32,
) -> Result<[u8; 112]> {
    if view.extent.contains(&0)
        || !max_lod.is_finite()
        || max_lod < 0.0
        || !lod_scale.is_finite()
        || !lod_offset.is_finite()
    {
        return Err(Error::InvalidRange);
    }
    let camera = view.shading_camera.ok_or(Error::InvalidState)?.uniform();
    let inverse = inverse(view.transform.columns())?;
    let mut values = Vec::with_capacity(28);
    values.extend(inverse.into_iter().flatten());
    values.extend(camera);
    values.extend([view.extent[0] as f32, view.extent[1] as f32, max_lod, 0.0]);
    values.extend([lod_scale, lod_offset, 0.0, 0.0]);
    let mut bytes = [0; 112];
    for (index, value) in values.into_iter().enumerate() {
        bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    Ok(bytes)
}

pub(crate) const BASIC_DEFERRED_VIEW_BYTES: u64 = 96;

pub(crate) fn append_basic_deferred_view(
    bytes: &mut Vec<u8>,
    view: &crate::RenderView,
    max_bytes: u64,
) -> Result<std::ops::Range<u64>> {
    let camera = view
        .shading_camera
        .map_or([0.0, 0.0, 1.0, 0.0], crate::ShadingCamera::uniform);
    let inverse = if camera[3] != 0.0 {
        inverse(view.transform.columns())?
    } else {
        crate::RenderTransform::identity().columns()
    };
    let start = u64::try_from(bytes.len()).map_err(|_| Error::InvalidRange)?;
    let end = start
        .checked_add(BASIC_DEFERRED_VIEW_BYTES)
        .ok_or(Error::InvalidRange)?;
    if view.extent.contains(&0) || end > max_bytes || end > u64::from(u32::MAX) {
        return Err(Error::InvalidRange);
    }
    bytes
        .try_reserve_exact(BASIC_DEFERRED_VIEW_BYTES as usize)
        .map_err(|_| Error::OutOfMemory)?;
    for value in inverse.into_iter().flatten().chain(camera).chain([
        view.extent[0] as f32,
        view.extent[1] as f32,
        0.0,
        0.0,
    ]) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(start..end)
}

fn inverse(columns: [[f32; 4]; 4]) -> Result<[[f32; 4]; 4]> {
    let mut rows = [[0.0_f64; 8]; 4];
    for r in 0..4 {
        for c in 0..4 {
            rows[r][c] = f64::from(columns[c][r]);
        }
        rows[r][r + 4] = 1.0;
    }
    for c in 0..4 {
        let pivot = (c..4)
            .max_by(|&a, &b| rows[a][c].abs().total_cmp(&rows[b][c].abs()))
            .unwrap();
        if rows[pivot][c] == 0.0 {
            return Err(Error::InvalidRange);
        }
        rows.swap(c, pivot);
        let divisor = rows[c][c];
        for x in &mut rows[c] {
            *x /= divisor;
        }
        for r in 0..4 {
            if r == c {
                continue;
            }
            let factor = rows[r][c];
            for j in 0..8 {
                rows[r][j] -= factor * rows[c][j];
            }
        }
    }
    let columns = core::array::from_fn(|c| core::array::from_fn(|r| rows[r][c + 4] as f32));
    if columns.iter().flatten().any(|x| !x.is_finite()) {
        return Err(Error::InvalidRange);
    }
    Ok(columns)
}

#[cfg(test)]
#[path = "../tests/frame_data/specular_view.rs"]
mod tests;
