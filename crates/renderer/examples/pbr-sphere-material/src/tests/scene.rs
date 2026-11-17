use crate::scene::{input, mesh, multi_input, sphere_centres, ORM};
use fluxel_material::MaterialCompiler;
#[test]
fn combined_mesh_has_six_orm_centres_and_required_attributes() {
    let x = input(false, [0., 0.35, 7.5]).unwrap();
    assert_eq!(x.assets.geometry[0].value().vertices.end, 1440);
    assert_eq!(x.orm_bytes.len(), 24);
    assert_eq!(&x.orm_bytes[..4], &ORM[0]);
    assert_eq!(
        x.assets.geometry[0].value().layout.streams[0]
            .attributes
            .len(),
        4
    );
}

#[test]
fn triangles_are_nondegenerate_tbn_is_orthogonal_and_material_compiles() {
    let geometry = mesh();
    for triangle in geometry.streams[0].chunks_exact(48 * 3) {
        let point = |vertex: usize| -> [f32; 3] {
            core::array::from_fn(|i| {
                f32::from_le_bytes(
                    triangle[vertex * 48 + i * 4..vertex * 48 + i * 4 + 4]
                        .try_into()
                        .unwrap(),
                )
            })
        };
        let (a, b, c) = (point(0), point(1), point(2));
        let cross = [
            (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]),
            (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]),
            (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]),
        ];
        assert!(cross.into_iter().map(|x| x * x).sum::<f32>() > 1e-8);
    }
    for vertex in geometry.streams[0].chunks_exact(48) {
        let f = |offset| f32::from_le_bytes(vertex[offset..offset + 4].try_into().unwrap());
        let normal = [f(12), f(16), f(20)];
        let tangent = [f(32), f(36), f(40)];
        assert!(
            normal
                .into_iter()
                .zip(tangent)
                .map(|(a, b)| a * b)
                .sum::<f32>()
                .abs()
                < 1e-5
        );
    }
    let input = input(false, [0., 0.35, 7.5]).unwrap();
    let variant = MaterialCompiler::compile(&input.material).unwrap();
    assert!(
        variant
            .contribution()
            .source_units()
            .iter()
            .any(|unit| unit.source.contains("return vec3<f32>(0.0,0.0,1.0);")),
        "the material contract must provide the unencoded neutral tangent-space normal"
    );
    assert_eq!(input.eye, [0., 0.35, 7.5]);
    assert_eq!(input.diffuse_bytes.len(), 16);
    assert_eq!(input.radiance_mip_bytes.len(), 3);
    assert_eq!(input.brdf_lut_bytes.len(), 32);
}

#[test]
fn right_handed_perspective_projects_all_six_world_centres_inside_distinct_cells() {
    let input = input(false, [0., 0.35, 7.5]).unwrap();
    let screen: Vec<_> = sphere_centres()
        .into_iter()
        .map(|point| {
            let source = [point[0], point[1], point[2], 1.];
            let clip: [f32; 4] = core::array::from_fn(|row| {
                (0..4)
                    .map(|column| input.vp[column][row] * source[column])
                    .sum()
            });
            assert!(clip[3] > 0., "RH forward sphere must have positive clip w");
            let ndc = [clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3]];
            assert!(ndc[0].abs() < 0.8 && ndc[1].abs() < 0.8 && (0.0..=1.0).contains(&ndc[2]));
            (
                (ndc[0] * 256. + 256.).round() as i32,
                (256. - ndc[1] * 256.).round() as i32,
            )
        })
        .collect();
    screen
        .windows(2)
        .for_each(|pair| assert_ne!(pair[0], pair[1]));
}

#[test]
fn multi_scene_has_six_independent_asset_owners_and_canonical_active_orm() {
    let input = multi_input(true, [0., 0.35, 7.5]).unwrap();
    let baseline = multi_input(false, [0., 0.35, 7.5]).unwrap();
    assert_eq!(input.assets.geometry.len(), 6);
    assert_eq!(input.assets.textures.len(), 6);
    assert_eq!(input.assets.materials.len(), 2);
    assert_eq!(input.object_sources.len(), 6);
    for (index, source) in input.object_sources.iter().enumerate() {
        assert_eq!(source.geometry_index, index);
        assert_eq!(source.texture_index, index);
        assert_eq!(source.material_index, usize::from(index == 3));
        assert_eq!(input.assets.geometry[index].value().vertices.end, 240);
        assert_eq!(&source.world_from_local[3][..3], &sphere_centres()[index]);
        assert_eq!(
            input
                .snapshot
                .record(input.snapshot.record_ids().nth(index).unwrap())
                .unwrap()
                .transform
                .columns(),
            source.world_from_local,
            "record must consume the published source matrix"
        );
        let texture = &input.assets.textures[index].value().rgba8;
        assert_eq!(
            &input.orm_bytes[index * 4..index * 4 + 4],
            &texture[index * 4..index * 4 + 4]
        );
        for other in 0..6 {
            if other != index {
                assert_eq!(&texture[other * 4..other * 4 + 4], &[255, 255, 0, 255]);
            }
        }
    }
    assert_eq!(&input.orm_bytes[..4], &[24, 38, 0, 255]);
    assert_eq!(&input.orm_bytes[12..16], &[255, 204, 0, 255]);
    for index in 1..6 {
        assert_eq!(
            &input.orm_bytes[index * 4..index * 4 + 4],
            &baseline.orm_bytes[index * 4..index * 4 + 4],
            "AO-only may change only sphere zero"
        );
    }
}

#[test]
fn multi_local_geometry_and_distinct_transforms_reconstruct_canonical_world_mesh() {
    let input = multi_input(false, [0., 0.35, 7.5]).unwrap();
    let canonical = mesh().streams[0].clone();
    for (index, source) in input.object_sources.iter().enumerate() {
        let local = &input.assets.geometry[index].value().streams[0];
        let centre = sphere_centres()[index];
        assert!(local
            .chunks_exact(48)
            .all(|vertex| { (0..3).all(|axis| decode_f32(vertex, axis * 4).abs() <= 0.72001) }));
        for (vertex, expected) in local
            .chunks_exact(48)
            .zip(canonical[index * 240 * 48..(index + 1) * 240 * 48].chunks_exact(48))
        {
            for axis in 0..3 {
                let reconstructed = decode_f32(vertex, axis * 4) + source.world_from_local[3][axis];
                assert!((reconstructed - decode_f32(expected, axis * 4)).abs() < 1e-6);
            }
        }
        assert_eq!(&source.world_from_local[3][..3], &centre);
    }
    assert_ne!(
        input.object_sources[0].world_from_local, input.object_sources[1].world_from_local,
        "each draw needs a distinct dynamic model offset"
    );
}

fn decode_f32(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

#[test]
fn multi_scene_sphere_three_uses_material_without_surface_effects() {
    let input = multi_input(false, [0., 0.35, 7.5]).unwrap();
    let effects = input.assets.materials[0].value().surface_extensions();
    let no_effects = input.assets.materials[1].value().surface_extensions();
    assert_eq!(effects.occlusion.as_deref(), Some("ao"));
    assert!(no_effects.occlusion.is_none() && no_effects.emissive.is_none());
    assert_eq!(input.object_sources[3].material_index, 1);
    assert_eq!(&input.orm_bytes[12..16], &[255, 204, 0, 255]);
}
