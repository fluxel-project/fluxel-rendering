use base64::{Engine as _, engine::general_purpose::STANDARD};
use renderer_07_offscreen_material::gltf;
use serde_json::json;

fn fixture(indices: &[u8]) -> Vec<u8> {
    let mut data = Vec::new();
    for value in [0.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
        data.extend_from_slice(&value.to_le_bytes());
    }
    for _ in 0..3 {
        for value in [0.0_f32, 0.0, 1.0] {
            data.extend_from_slice(&value.to_le_bytes());
        }
    }
    data.extend_from_slice(indices);
    serde_json::to_vec(&json!({
        "asset":{"version":"2.0"},
        "buffers":[{"byteLength":data.len(),"uri":format!("data:application/octet-stream;base64,{}",STANDARD.encode(&data))}],
        "bufferViews":[
          {"buffer":0,"byteOffset":0,"byteLength":36},
          {"buffer":0,"byteOffset":36,"byteLength":36},
          {"buffer":0,"byteOffset":72,"byteLength":indices.len()}
        ],
        "accessors":[
          {"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"},
          {"bufferView":1,"componentType":5126,"count":3,"type":"VEC3"},
          {"bufferView":2,"componentType":5121,"count":3,"type":"SCALAR"}
        ],
        "meshes":[{"primitives":[{"attributes":{"POSITION":0,"NORMAL":1},"indices":2}]}],
        "nodes":[{"mesh":0}],
        "scenes":[{"nodes":[0]}],
        "scene":0
    })).expect("fixture JSON")
}

#[test]
fn expands_indices_and_explicitly_defaults_only_missing_uvs() {
    let geometry = gltf::load(&fixture(&[2, 1, 0])).expect("valid indexed triangle");
    assert_eq!(geometry.vertices, 0..3);
    assert_eq!(geometry.streams[0].len(), 96);
    assert_eq!(position(&geometry.streams[0], 0), [0.0, 1.0, 0.0]);
    assert_eq!(position(&geometry.streams[0], 1), [1.0, 0.0, 0.0]);
    assert_eq!(position(&geometry.streams[0], 2), [0.0, 0.0, 0.0]);
    assert_eq!(&geometry.streams[0][24..32], &[0; 8]);
}

#[test]
fn active_scene_node_uniform_trs_changes_expanded_positions() {
    let mut document: serde_json::Value =
        serde_json::from_slice(&fixture(&[2, 1, 0])).expect("JSON");
    document["nodes"][0]["translation"] = json!([3.0, 4.0, 5.0]);
    document["nodes"][0]["scale"] = json!([2.0, 2.0, 2.0]);
    let geometry =
        gltf::load(&serde_json::to_vec(&document).expect("JSON")).expect("uniform TRS must decode");
    assert_eq!(position(&geometry.streams[0], 0), [3.0, 6.0, 5.0]);
    assert_eq!(position(&geometry.streams[0], 1), [5.0, 4.0, 5.0]);
    assert_eq!(position(&geometry.streams[0], 2), [3.0, 4.0, 5.0]);
}

#[test]
fn rejects_out_of_range_index_before_geometry_is_emitted() {
    let error = gltf::load(&fixture(&[0, 1, 3])).expect_err("bad index must reject");
    assert!(error.to_string().contains("index exceeds POSITION"));
}

#[test]
fn rejects_truncated_accessor_range() {
    let mut document: serde_json::Value =
        serde_json::from_slice(&fixture(&[0, 1, 2])).expect("JSON");
    document["accessors"][0]["count"] = json!(4);
    let error =
        gltf::load(&serde_json::to_vec(&document).expect("JSON")).expect_err("range must reject");
    assert!(error.to_string().contains("accessor exceeds bufferView"));
}

#[test]
fn rejects_stride_smaller_than_the_float_attribute() {
    let mut document: serde_json::Value =
        serde_json::from_slice(&fixture(&[0, 1, 2])).expect("JSON");
    document["bufferViews"][0]["byteStride"] = json!(8);
    let error =
        gltf::load(&serde_json::to_vec(&document).expect("JSON")).expect_err("stride must reject");
    assert!(error.to_string().contains("byteStride"));
}

#[test]
fn rejects_non_data_uri_and_non_triangle_mode() {
    let mut document: serde_json::Value =
        serde_json::from_slice(&fixture(&[0, 1, 2])).expect("JSON");
    document["buffers"][0]["uri"] = json!("file.bin");
    assert!(gltf::load(&serde_json::to_vec(&document).expect("JSON")).is_err());
    let mut document: serde_json::Value =
        serde_json::from_slice(&fixture(&[0, 1, 2])).expect("JSON");
    document["meshes"][0]["primitives"][0]["mode"] = json!(5);
    assert!(gltf::load(&serde_json::to_vec(&document).expect("JSON")).is_err());
}

#[test]
#[ignore = "requires unredistributed local RHI reference assets"]
fn selected_rhi_models_decode_to_nonempty_renderer_geometry() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let assets = root.join("../../../../../fluxel-rhi/crates/rhi/examples/assets/models");
    for (name, expected_vertices) in [("chinesedragon.gltf", 302_880), ("plane.gltf", 6)] {
        let bytes = std::fs::read(assets.join(name)).expect("local RHI reference asset");
        let geometry = gltf::load(&bytes).expect("selected RHI model must decode");
        assert!(!geometry.streams[0].is_empty(), "{name}");
        assert_eq!(geometry.streams[0].len() % 32, 0, "{name}");
        assert_eq!(geometry.vertices.end, expected_vertices, "{name}");
    }
}

fn position(bytes: &[u8], vertex: usize) -> [f32; 3] {
    let start = vertex * 32;
    [
        f32::from_le_bytes(bytes[start..start + 4].try_into().expect("x")),
        f32::from_le_bytes(bytes[start + 4..start + 8].try_into().expect("y")),
        f32::from_le_bytes(bytes[start + 8..start + 12].try_into().expect("z")),
    ]
}

#[test]
fn nonuniform_signed_trs_uses_inverse_transpose_normals() {
    let mut document: serde_json::Value = serde_json::from_slice(&fixture(&[0, 1, 2])).unwrap();
    let uri = document["buffers"][0]["uri"].as_str().unwrap();
    let mut bytes = STANDARD.decode(uri.split_once(',').unwrap().1).unwrap();
    for vertex in 0..3 {
        for (axis, value) in [1.0_f32, 1.0, 0.0].into_iter().enumerate() {
            let offset = 36 + vertex * 12 + axis * 4;
            bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
    }
    document["buffers"][0]["uri"] = json!(format!(
        "data:application/octet-stream;base64,{}",
        STANDARD.encode(bytes)
    ));
    document["nodes"][0]["scale"] = json!([-2.0, 3.0, 4.0]);
    let half = std::f32::consts::FRAC_1_SQRT_2;
    document["nodes"][0]["rotation"] = json!([0.0, 0.0, half, half]);
    document["nodes"][0]["translation"] = json!([3.0, 4.0, 5.0]);
    let geometry = gltf::load(&serde_json::to_vec(&document).unwrap()).unwrap();
    for (a, e) in position(&geometry.streams[0], 1)
        .into_iter()
        .zip([3.0, 2.0, 5.0])
    {
        assert!((a - e).abs() < 1e-5);
    }
    let normal: Vec<_> = geometry.streams[0][12..24]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    for (a, e) in normal
        .into_iter()
        .zip([-2.0 / 13.0_f32.sqrt(), -3.0 / 13.0_f32.sqrt(), 0.0])
    {
        assert!((a - e).abs() < 1e-5, "{a} != {e}");
    }
}

#[test]
fn singular_node_scales_refuse_geometry() {
    for scale in [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]] {
        let mut document: serde_json::Value = serde_json::from_slice(&fixture(&[0, 1, 2])).unwrap();
        document["nodes"][0]["scale"] = json!(scale);
        assert!(
            gltf::load(&serde_json::to_vec(&document).unwrap())
                .unwrap_err()
                .to_string()
                .contains("scale must be nonzero")
        );
    }
}
