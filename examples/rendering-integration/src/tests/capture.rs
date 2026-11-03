use super::validate_capture_pixels;

#[test]
fn rejects_desktop_pixels_that_are_not_the_fixture_triangle() {
    let mut pixels = vec![0_u8; 4 * 32 * 18];
    // A wallpaper-colored stable interior must fail even when every corner is black.
    let offset = ((18 * 57 / 100) * 32 + (32 * 8 / 20)) * 4;
    pixels[offset..offset + 4].copy_from_slice(&[71, 92, 113, 255]);
    assert!(validate_capture_pixels(&pixels, [32, 18], [255, 0, 0, 255]).is_err());
}

#[test]
fn accepts_a_fixture_colored_stable_interior_and_rejects_zero_extent() {
    let mut pixels = vec![0_u8; 4 * 32 * 18];
    let offset = ((18 * 57 / 100) * 32 + (32 * 8 / 20)) * 4;
    pixels[offset..offset + 4].copy_from_slice(&[0, 0, 255, 255]);
    assert!(validate_capture_pixels(&pixels, [32, 18], [255, 0, 0, 255]).is_ok());
    assert!(validate_capture_pixels(&[], [0, 0], [255, 0, 0, 255]).is_err());
}
