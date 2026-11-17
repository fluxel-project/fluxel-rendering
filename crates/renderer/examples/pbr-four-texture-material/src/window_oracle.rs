//! Dense RGB comparison for opaque Windows Graphics Capture evidence.

use std::{error::Error, fs, path::Path};

pub const DWM_BOTTOM_CORNER: u32 = 16;

pub fn write_contract(output: &Path, extent: [u32; 2]) -> Result<(), Box<dyn Error>> {
    let mask: Vec<u8> = (0..extent[1])
        .flat_map(|y| (0..extent[0]).map(move |x| u8::from(masked([x, y], extent))))
        .collect();
    let masked_pixels = mask.iter().filter(|&&value| value != 0).count();
    if masked_pixels != 512 {
        return Err(format!("DWM mask must contain 512 pixels, found {masked_pixels}").into());
    }
    fs::write(output.join("visible-dense-mask.bin"), mask)?;
    fs::write(
        output.join("visible-dense-contract.json"),
        format!(
            "{{\"schemaVersion\":1,\"extent\":[{},{}],\"comparison\":\"exact RGB per unmasked pixel\",\"mask\":\"bottom-left and bottom-right 16x16 DWM client corners only\",\"maskedPixels\":512,\"alpha\":\"ignored for opaque Surface\"}}\n",
            extent[0], extent[1]
        ),
    )?;
    Ok(())
}

pub fn compare_bmp_rgb(
    bmp: &[u8],
    companion: &[u8],
    extent: [u32; 2],
) -> Result<(), Box<dyn Error>> {
    let pixels = extent[0] as usize * extent[1] as usize;
    if bmp.len() != 54 + pixels * 4 || companion.len() != pixels * 4 || &bmp[..2] != b"BM" {
        return Err("window capture and companion shapes differ".into());
    }
    for y in 0..extent[1] {
        for x in 0..extent[0] {
            if masked([x, y], extent) {
                continue;
            }
            let at = ((y * extent[0] + x) * 4) as usize;
            if [bmp[54 + at + 2], bmp[54 + at + 1], bmp[54 + at]]
                != [companion[at], companion[at + 1], companion[at + 2]]
            {
                return Err(
                    format!("window RGB differs from same-input companion at ({x},{y})").into(),
                );
            }
        }
    }
    Ok(())
}

pub(crate) fn masked(pixel: [u32; 2], extent: [u32; 2]) -> bool {
    pixel[1] >= extent[1].saturating_sub(DWM_BOTTOM_CORNER)
        && (pixel[0] < DWM_BOTTOM_CORNER || pixel[0] >= extent[0].saturating_sub(DWM_BOTTOM_CORNER))
}

#[cfg(test)]
#[path = "tests/window_oracle.rs"]
mod tests;
