use super::validate_capture_pixels;

fn interior_offset(width: usize, height: usize) -> usize {
    ((height * 57 / 100) * width + (width * 8 / 20)) * 4
}

#[test]
fn rejects_desktop_pixels_that_are_not_the_fixture_triangle() {
    let mut pixels = vec![0_u8; 4 * 32 * 18];
    // A wallpaper-colored stable interior must fail even when every corner is black.
    let offset = interior_offset(32, 18);
    pixels[offset..offset + 4].copy_from_slice(&[71, 92, 113, 255]);
    assert!(validate_capture_pixels(&pixels, [32, 18], [255, 0, 0, 255], 1).is_err());
}

#[test]
fn accepts_a_fixture_colored_stable_interior_and_rejects_zero_extent() {
    let mut pixels = vec![0_u8; 4 * 32 * 18];
    let offset = interior_offset(32, 18);
    pixels[offset..offset + 4].copy_from_slice(&[0, 0, 255, 255]);
    assert!(validate_capture_pixels(&pixels, [32, 18], [255, 0, 0, 255], 0).is_ok());
    assert!(validate_capture_pixels(&[], [0, 0], [255, 0, 0, 255], 0).is_err());
}

#[test]
fn triangle_tolerance_is_opt_in_and_bounded() {
    let mut pixels = vec![0_u8; 4 * 32 * 18];
    let offset = interior_offset(32, 18);
    // BGRA for RGB [254, 0, 0].
    pixels[offset..offset + 4].copy_from_slice(&[0, 0, 254, 255]);
    assert!(validate_capture_pixels(&pixels, [32, 18], [255, 0, 0, 255], 0).is_err());
    assert!(validate_capture_pixels(&pixels, [32, 18], [255, 0, 0, 255], 1).is_ok());
    pixels[offset..offset + 4].copy_from_slice(&[0, 0, 253, 255]);
    assert!(validate_capture_pixels(&pixels, [32, 18], [255, 0, 0, 255], 1).is_err());
}

#[test]
fn tolerance_never_relaxes_background_corners() {
    let mut pixels = vec![0_u8; 4 * 32 * 18];
    let offset = interior_offset(32, 18);
    pixels[offset..offset + 4].copy_from_slice(&[0, 0, 254, 255]);
    // BGRA [1, 0, 0] in a background corner must remain an exact failure.
    pixels[..4].copy_from_slice(&[0, 0, 1, 255]);
    assert!(validate_capture_pixels(&pixels, [32, 18], [255, 0, 0, 255], 1).is_err());
}
