use super::{sample, sample_at_pixel, srgb_to_linear, texture};

#[test]
fn black_white_half_texel_is_linear_half_not_srgb_half() {
    let texture = texture();
    let sample = sample(&texture, 0.5, 0.0);
    assert_eq!(sample[0], 128, "decoded black/white midpoint is linear 0.5");
    assert_eq!(
        sample_at_pixel(&texture, 319, 0)[0],
        128,
        "the actual shifted-UV raster probe is the midpoint"
    );
    assert_eq!(
        sample_at_pixel(&texture, 319, 0)[3],
        128,
        "alpha filters as linear bytes"
    );
    assert_eq!(srgb_to_linear(0.5).mul_add(255.0, 0.0).round() as u8, 55);
}

#[test]
fn nonbinary_texel_rejects_ignored_or_filter_before_decode_paths() {
    let texture = texture();
    let expected = sample(&texture, 0.0, 0.0);
    assert_eq!(
        expected[2], 55,
        "sRGB byte 128 decodes exactly by the piecewise curve"
    );
    assert!(
        expected[2].abs_diff(128) > 1,
        "ignoring sRGB decode exceeds the fixed tolerance"
    );
    let midpoint = sample(&texture, 0.5, 0.0);
    assert!(
        midpoint[0].abs_diff(55) > 1,
        "filter-before-decode exceeds the fixed tolerance"
    );
}

#[test]
fn actual_runtime_probes_distinguish_each_incorrect_transfer_path() {
    let input = texture();
    let probes = super::probes(&input);
    let mut encoded = texture();
    encoded.encoding = fluxel_renderer::TextureEncoding::Linear;
    let mut rejected = [false; 3];
    for probe in probes {
        let filtered_encoded = sample_at_pixel(&encoded, probe.x, probe.y);
        for channel in 0..3 {
            let decode = |byte: u8| (srgb_to_linear(f64::from(byte) / 255.0) * 255.0).round() as u8;
            let wrong = [
                filtered_encoded[channel],
                decode(filtered_encoded[channel]),
                decode(probe.expected[channel]),
            ];
            for (rejected, wrong) in rejected.iter_mut().zip(wrong) {
                *rejected |= probe.expected[channel].abs_diff(wrong) > 1;
            }
        }
    }
    assert!(rejected[0], "runtime probes reject missing RGB decode");
    assert!(rejected[1], "runtime probes reject filtering before decode");
    assert!(rejected[2], "runtime probes reject decoding RGB twice");
}
