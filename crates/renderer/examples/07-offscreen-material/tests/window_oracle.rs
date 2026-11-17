use renderer_07_offscreen_material::window_oracle;

const EXTENT: [u32; 2] = [32, 32];

#[test]
fn dense_rgb_accepts_exact_pixels_and_only_fixed_dwm_corners() {
    let raw = vec![0_u8; (EXTENT[0] * EXTENT[1] * 4) as usize];
    let bmp = bmp_from_rgba(&raw);
    let result = window_oracle::compare_bmp_rgb(&bmp, &raw, EXTENT).expect("exact RGB");
    assert_eq!(result.masked, 512);
    assert_eq!(result.compared, 512);
}

#[test]
fn dense_rgb_rejects_half_texel_mixed_window_pixel() {
    let raw = vec![0_u8; (EXTENT[0] * EXTENT[1] * 4) as usize];
    let mut bmp = bmp_from_rgba(&raw);
    let at = 54 + ((8 * EXTENT[0] + 9) * 4) as usize;
    bmp[at..at + 4].copy_from_slice(&[9, 23, 56, 255]);
    let error =
        window_oracle::compare_bmp_rgb(&bmp, &raw, EXTENT).expect_err("mixed RGB must fail");
    assert!(error.to_string().contains("(9,8)"));
}

fn bmp_from_rgba(raw: &[u8]) -> Vec<u8> {
    let mut bmp = vec![0; 54 + raw.len()];
    bmp[..2].copy_from_slice(b"BM");
    for (source, target) in raw.chunks_exact(4).zip(bmp[54..].chunks_exact_mut(4)) {
        target.copy_from_slice(&[source[2], source[1], source[0], 255]);
    }
    bmp
}
