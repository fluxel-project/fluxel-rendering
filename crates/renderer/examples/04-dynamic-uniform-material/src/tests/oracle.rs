use crate::oracle::{Sample, less_equal, overlap_expected};

#[test]
fn nearer_colour_survives_both_source_orders() {
    let far = Sample {
        depth: 0.75,
        rgb: [241, 92, 32],
    };
    let near = Sample {
        depth: 0.25,
        rgb: [20, 176, 213],
    };
    assert_eq!(overlap_expected([far, near]), near.rgb);
    assert_eq!(overlap_expected([near, far]), near.rgb);
}

#[test]
fn equal_depth_replaces_under_less_equal() {
    assert_eq!(
        less_equal(
            Some(Sample {
                depth: 0.5,
                rgb: [1, 2, 3]
            }),
            Sample {
                depth: 0.5,
                rgb: [4, 5, 6]
            }
        )
        .rgb,
        [4, 5, 6]
    );
}

#[test]
fn depth_is_linear_in_screen_space_while_colour_is_perspective_correct() {
    let mut image = crate::oracle::RasterImage::clear([4, 4]);
    // At pixel (0,2), barycentrics are (1/2,1/8,3/8):
    // depth = .5*.2 + .125*.8 + .375*.6 = .425.
    image.triangle(
        [[-1., -1., 0.2, 1.], [2., -2., 1.6, 2.], [-4., 4., 2.4, 4.]],
        [[1., 0., 0.]; 3],
    );
    image.triangle(
        [
            [-1., -1., 0.375, 1.],
            [1., -1., 0.375, 1.],
            [-1., 1., 0.375, 1.],
        ],
        [[0., 1., 0.]; 3],
    );
    assert_eq!(&image.rgba[32..36], &[0, 255, 0, 255]);
}
