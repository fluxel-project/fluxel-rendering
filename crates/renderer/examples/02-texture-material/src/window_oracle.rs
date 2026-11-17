//! Dense RGB comparison between a captured opaque window and RHI readback.

use std::{error::Error, fs, path::Path};

pub const DWM_BOTTOM_CORNER: u32 = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DenseComparison {
    pub compared: usize,
    pub masked: usize,
}

pub fn write_contract(output: &Path, extent: [u32; 2]) -> Result<(), Box<dyn Error>> {
    let mut mask = Vec::with_capacity((extent[0] * extent[1]) as usize);
    for y in 0..extent[1] {
        for x in 0..extent[0] {
            mask.push(u8::from(masked([x, y], extent)));
        }
    }
    fs::write(output.join("visible-dense-mask.bin"), mask)?;
    fs::write(
        output.join("visible-dense-contract.json"),
        format!(
            "{{\"schemaVersion\":1,\"bmp\":\"Windows Graphics Capture BGRA8 top-down\",\"companion\":\"same-input RHI RGBA8 readback\",\"extent\":[{},{}],\"comparison\":\"exact RGB per unmasked pixel\",\"alpha\":\"ignored because Surface alpha is opaque\",\"mask\":\"bottom-left and bottom-right 16x16 DWM client corners only\"}}\n",
            extent[0], extent[1]
        ),
    )?;
    Ok(())
}

pub fn compare_bmp_rgb(
    bmp: &[u8],
    companion: &[u8],
    extent: [u32; 2],
) -> Result<DenseComparison, Box<dyn Error>> {
    let pixels = (extent[0] as usize)
        .checked_mul(extent[1] as usize)
        .ok_or("window extent overflows")?;
    if bmp.len() != 54 + pixels * 4 || &bmp[..2] != b"BM" || companion.len() != pixels * 4 {
        return Err("window dense comparison input shape is invalid".into());
    }
    let mut result = DenseComparison {
        compared: 0,
        masked: 0,
    };
    for y in 0..extent[1] {
        for x in 0..extent[0] {
            if masked([x, y], extent) {
                result.masked += 1;
                continue;
            }
            let at = ((y * extent[0] + x) * 4) as usize;
            let found = [bmp[54 + at + 2], bmp[54 + at + 1], bmp[54 + at]];
            let expected = [companion[at], companion[at + 1], companion[at + 2]];
            if found != expected {
                return Err(format!("window RGB differs from same-input companion at ({x},{y}): found {found:?}, expected {expected:?}").into());
            }
            result.compared += 1;
        }
    }
    Ok(result)
}

pub(super) fn masked(pixel: [u32; 2], extent: [u32; 2]) -> bool {
    pixel[1] >= extent[1].saturating_sub(DWM_BOTTOM_CORNER)
        && (pixel[0] < DWM_BOTTOM_CORNER || pixel[0] >= extent[0].saturating_sub(DWM_BOTTOM_CORNER))
}

#[cfg(test)]
#[path = "tests/window_oracle.rs"]
mod tests;
