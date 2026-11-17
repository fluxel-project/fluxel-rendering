use crate::{verify_probes, window_oracle::compare_bmp_rgb, Probe, EXTENT};

#[test]
fn fixed_mask_only_excludes_two_bottom_corners() {
    let extent = [64, 64];
    let mut bmp = vec![0; 54 + 64 * 64 * 4];
    bmp[..2].copy_from_slice(b"BM");
    let raw = vec![0; 64 * 64 * 4];
    assert_eq!(compare_bmp_rgb(&bmp, &raw, extent).unwrap().masked, 512);
    // x=32 is between the two fixed 16-pixel corners; y=48 is bottom-row territory.
    bmp[54 + (48 * 64 + 32) * 4 + 2] = 1;
    assert!(compare_bmp_rgb(&bmp, &raw, extent).is_err());
}

#[test]
fn dense_comparison_rejects_half_texel_mixed_rgb() {
    let extent = [64, 64];
    let mut bmp = vec![0; 54 + 64 * 64 * 4];
    bmp[..2].copy_from_slice(b"BM");
    let raw = vec![0; 64 * 64 * 4];
    // [56, 23, 9] is the retained half-texel failure shape, not a valid exact match.
    let at = 54 + (20 * 64 + 20) * 4;
    bmp[at..at + 3].copy_from_slice(&[9, 23, 56]);
    assert!(compare_bmp_rgb(&bmp, &raw, extent).is_err());
}

#[test]
fn numerical_probe_allows_one_quantized_code_but_not_two() {
    let probe = Probe {
        x: 17,
        y: 19,
        expected: [100, 120, 140, 255],
    };
    let mut raw = vec![0; (EXTENT[0] * EXTENT[1] * 4) as usize];
    let at = ((probe.y * EXTENT[0] + probe.x) * 4) as usize;
    raw[at..at + 4].copy_from_slice(&[101, 119, 141, 255]);
    assert!(verify_probes(&raw, std::slice::from_ref(&probe), 1).is_ok());
    raw[at] = 102;
    assert!(verify_probes(&raw, &[probe], 1).is_err());
}
