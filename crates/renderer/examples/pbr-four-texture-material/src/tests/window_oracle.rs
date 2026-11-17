use super::{DWM_BOTTOM_CORNER, compare_bmp_rgb, masked};

#[test]
fn fixed_corners_mask_exactly_512_pixels() {
    let extent = [512, 512];
    let count = (0..extent[1])
        .flat_map(|y| (0..extent[0]).map(move |x| masked([x, y], extent)))
        .filter(|masked| *masked)
        .count();
    assert_eq!(count, 2 * (DWM_BOTTOM_CORNER * DWM_BOTTOM_CORNER) as usize);
}

#[test]
fn dense_comparison_ignores_only_declared_corners() {
    let extent = [32, 32];
    let companion = vec![17u8; (extent[0] * extent[1] * 4) as usize];
    let mut bmp = vec![0u8; 54 + companion.len()];
    bmp[..2].copy_from_slice(b"BM");
    for pixel in 0..(extent[0] * extent[1]) as usize {
        let at = pixel * 4;
        bmp[54 + at] = companion[at + 2];
        bmp[54 + at + 1] = companion[at + 1];
        bmp[54 + at + 2] = companion[at];
    }
    let corner = ((31 * extent[0]) * 4) as usize;
    bmp[54 + corner] ^= 1;
    assert!(compare_bmp_rgb(&bmp, &companion, extent).is_ok());
    let at = ((2 * extent[0] + 16) * 4) as usize;
    bmp[54 + at] ^= 1;
    assert!(compare_bmp_rgb(&bmp, &companion, extent).is_err());
}
