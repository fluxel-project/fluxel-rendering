use super::*;

#[test]
fn near_plane_clipping_interpolates_position_and_uv() {
    let polygon = clip([
        Vertex {
            clip: [-0.5, -0.5, -0.5, 1.0],
            uv: [0.0, 0.0],
            local: [-0.5, -0.5, -0.5],
        },
        Vertex {
            clip: [0.5, -0.5, 0.5, 1.0],
            uv: [1.0, 0.0],
            local: [0.5, -0.5, 0.5],
        },
        Vertex {
            clip: [-0.5, 0.5, 0.5, 1.0],
            uv: [0.0, 1.0],
            local: [-0.5, 0.5, 0.5],
        },
    ]);
    assert_eq!(polygon.len(), 4);
    assert!(
        polygon
            .iter()
            .any(|v| v.clip == [0.0, -0.5, 0.0, 1.0] && v.uv == [0.5, 0.0])
    );
    assert!(
        polygon
            .iter()
            .any(|v| v.clip == [-0.5, 0.0, 0.0, 1.0] && v.uv == [0.0, 0.5])
    );
}

#[test]
fn clipped_foreground_faces_still_occlude_candidate_probe_triangles() {
    let mut pixels = vec![None; (scene::EXTENT[0] * scene::EXTENT[1]) as usize];
    let vertex = |x, y, z| Vertex {
        clip: [x, y, z, 1.0],
        uv: [0.0, 0.0],
        local: [x, y, z],
    };
    triangle(
        &mut pixels,
        1,
        [
            vertex(-0.8, -0.8, 0.8),
            vertex(0.8, -0.8, 0.8),
            vertex(0.0, 0.8, 0.8),
        ],
    )
    .unwrap();
    assert_eq!(pixels[256 * 512 + 256].unwrap().triangle, 1);
    triangle(
        &mut pixels,
        2,
        [
            vertex(-0.5, -0.5, 0.2),
            vertex(2.0, -0.5, 0.2),
            vertex(-0.5, 2.0, 0.2),
        ],
    )
    .unwrap();
    let winner = pixels[256 * 512 + 256].unwrap();
    assert_eq!(winner.triangle, 2);
    assert!((winner.depth - 0.2).abs() < 1e-6);
}
