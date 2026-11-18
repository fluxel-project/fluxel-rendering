use super::*;
use crate::{RenderTransform, RenderView, ShadingCamera};

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect()
}

#[test]
fn view_inverse_and_camera_bytes_follow_the_actual_uniform_layout() {
    let view = RenderView::new(
        RenderTransform::from_columns([
            [2., 0., 0., 0.],
            [0., 4., 0., 0.],
            [0., 0., 0.5, 0.],
            [6., 8., 1., 1.],
        ])
        .unwrap(),
        [512, 256],
    )
    .unwrap()
    .with_shading_camera(ShadingCamera::perspective([3., 4., 5.]).unwrap());
    let data = floats(&pack_specular_view(&view, 3., 3., -0.25).unwrap());
    assert_eq!(
        &data[..16],
        &[
            0.5, 0., 0., 0., 0., 0.25, 0., 0., 0., 0., 2., 0., -3., -2., -2., 1.
        ]
    );
    assert_eq!(
        &data[16..28],
        &[3., 4., 5., 1., 512., 256., 3., 0., 3., -0.25, 0., 0.]
    );
}

#[test]
fn missing_camera_or_singular_projection_refuses_uniform_preparation() {
    let view = RenderView::new(RenderTransform::identity(), [4, 4]).unwrap();
    assert!(matches!(
        pack_specular_view(&view, 2., 2., 0.),
        Err(Error::InvalidState)
    ));
    let view = RenderView::new(RenderTransform::from_columns([[0.; 4]; 4]).unwrap(), [4, 4])
        .unwrap()
        .with_shading_camera(ShadingCamera::orthographic([0., 0., 1.]).unwrap());
    assert!(matches!(
        pack_specular_view(&view, 2., 2., 0.),
        Err(Error::InvalidRange)
    ));
}

#[test]
fn camera_constructor_refuses_invalid_inputs_and_normalizes_large_directions() {
    assert!(ShadingCamera::perspective([f32::NAN, 0., 0.]).is_err());
    assert!(ShadingCamera::orthographic([0.; 3]).is_err());
    let camera = ShadingCamera::orthographic([f32::MAX, 0., 0.]).unwrap();
    assert_eq!(camera.direction_to_eye(), Some([1., 0., 0.]));
    assert_eq!(camera.position(), None);
}

#[test]
fn invalid_maximum_or_nonfinite_lod_metadata_is_refused() {
    let view = RenderView::new(RenderTransform::identity(), [4, 4])
        .unwrap()
        .with_shading_camera(ShadingCamera::orthographic([0., 0., 1.]).unwrap());
    for (max, scale, offset) in [(-1., 1., 0.), (2., f32::NAN, 0.), (2., 1., f32::NAN)] {
        assert!(pack_specular_view(&view, max, scale, offset).is_err());
    }
}

#[test]
fn finite_signed_lod_conversion_survives_the_uniform_boundary() {
    let view = RenderView::new(RenderTransform::identity(), [4, 4])
        .unwrap()
        .with_shading_camera(ShadingCamera::orthographic([0., 0., 1.]).unwrap());
    let values = floats(&pack_specular_view(&view, 4., -2., 3.).unwrap());
    assert_eq!(&values[24..28], &[-2., 3., 0., 0.]);
    let lod = crate::LinearSpecularLod {
        scale: -2.,
        offset: 3.,
    };
    assert_eq!(lod.map(0., 5), Some(3.));
    assert_eq!(lod.map(0.25, 5), Some(1.));
    assert_eq!(lod.map(0.5, 5), Some(0.));
}
