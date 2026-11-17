//! Exact per-view uniform bytes for the shared specular environment consumers.

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
