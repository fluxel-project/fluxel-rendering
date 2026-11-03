use fluxel_shader::{
    BindingDeclaration, BindingKind, BindingShape, EntryPoint, ScalarType, ShaderComposer,
    ShaderCompositionRequest, ShaderStage, SourceUnit, VertexInputDeclaration, VertexType,
};

const COMPUTE_REFLECTION: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<storage, read> query_only: array<u32>;
@group(0) @binding(2) var<storage, read_write> atomics: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> unused: array<u32>;

@compute @workgroup_size(1)
fn compute_main() {
    let loaded = data[0];
    data[1] = loaded;
    _ = arrayLength(&query_only);
    atomicAdd(&atomics[0], 1u);
}
"#;

const VERTEX_REFLECTION: &str = r#"
struct VertexInput {
    @location(2) position: vec2<f32>,
    @builtin(vertex_index) vertex_index: u32,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

@vertex
fn vertex_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.position = vec4<f32>(input.position, 0.0, 1.0);
    return output;
}
"#;

fn request(source: &str) -> ShaderCompositionRequest {
    ShaderCompositionRequest::new(vec![SourceUnit::new("reflection.wgsl", source)])
}

fn compute_request() -> ShaderCompositionRequest {
    request(COMPUTE_REFLECTION)
        .with_entry_point(EntryPoint::new("compute_main", ShaderStage::Compute))
        .with_binding(BindingDeclaration::new(0, 0, BindingKind::StorageReadWrite))
        .with_binding(BindingDeclaration::new(0, 1, BindingKind::StorageRead))
        .with_binding(BindingDeclaration::new(0, 2, BindingKind::StorageReadWrite))
}

#[test]
fn reflection_is_entry_reachable_and_preserves_global_access() {
    let artifact = ShaderComposer::new().compose(&compute_request()).unwrap();
    let bindings = artifact.interface().bindings();

    assert_eq!(bindings.len(), 3, "unused bound globals are not reflected");

    let data = bindings
        .iter()
        .find(|binding| (binding.group, binding.binding) == (0, 0))
        .unwrap();
    assert_eq!(data.kind, BindingKind::StorageReadWrite);
    assert!(data.visibility.contains(&ShaderStage::Compute));
    assert!(data.read);
    assert!(data.write);

    let query_only = bindings
        .iter()
        .find(|binding| (binding.group, binding.binding) == (0, 1))
        .unwrap();
    assert!(
        query_only.query,
        "arrayLength use must remain visible to reflection"
    );

    let atomics = bindings
        .iter()
        .find(|binding| (binding.group, binding.binding) == (0, 2))
        .unwrap();
    // Naga 27 reports buffer atomics as READ | WRITE; its ATOMIC flag denotes
    // image atomics, so the portable buffer contract must retain both accesses.
    assert!(atomics.read && atomics.write);
    assert!(
        bindings
            .iter()
            .all(|binding| (binding.group, binding.binding) != (0, 3))
    );
}

#[test]
fn vertex_reflection_keeps_typed_locations_and_excludes_builtins() {
    let mut composer = ShaderComposer::new();
    let base = composer
        .compose(
            &request(VERTEX_REFLECTION)
                .with_entry_point(EntryPoint::new("vertex_main", ShaderStage::Vertex)),
        )
        .unwrap();
    let inputs = base.interface().vertex_inputs();

    assert_eq!(inputs.len(), 1);
    assert_eq!(inputs[0].location, 2);
    assert_eq!(inputs[0].ty, VertexType::Vec2(ScalarType::F32));

    let declared = VertexInputDeclaration::new("position", 2, inputs[0].ty);
    composer
        .compose(
            &request(VERTEX_REFLECTION)
                .with_entry_point(EntryPoint::new("vertex_main", ShaderStage::Vertex))
                .with_vertex_input(declared),
        )
        .unwrap();
}

#[test]
fn binding_metadata_must_match_exactly_once() {
    let missing = request(COMPUTE_REFLECTION)
        .with_entry_point(EntryPoint::new("compute_main", ShaderStage::Compute));
    assert!(ShaderComposer::new().compose(&missing).is_err());

    let extra = request(COMPUTE_REFLECTION)
        .with_entry_point(EntryPoint::new("compute_main", ShaderStage::Compute))
        .with_binding(BindingDeclaration::new(0, 0, BindingKind::StorageReadWrite))
        .with_binding(BindingDeclaration::new(0, 1, BindingKind::StorageRead))
        .with_binding(BindingDeclaration::new(0, 2, BindingKind::StorageReadWrite))
        .with_binding(BindingDeclaration::new(0, 9, BindingKind::StorageRead));
    assert!(ShaderComposer::new().compose(&extra).is_err());

    let duplicate = compute_request().with_binding(BindingDeclaration::new(
        0,
        0,
        BindingKind::StorageReadWrite,
    ));
    assert!(ShaderComposer::new().compose(&duplicate).is_err());
}

#[test]
fn missing_requested_entry_and_empty_entry_metadata_are_rejected() {
    let missing = request(COMPUTE_REFLECTION)
        .with_entry_point(EntryPoint::new("not_present", ShaderStage::Compute));
    assert!(ShaderComposer::new().compose(&missing).is_err());

    let no_entry = request("const value: u32 = 1u;");
    assert!(ShaderComposer::new().compose(&no_entry).is_err());
}

#[test]
fn multiple_vertex_entries_are_rejected_before_shared_input_matching() {
    let source = r#"
        @vertex fn first(@location(0) value: vec2<f32>) -> @builtin(position) vec4<f32> {
            return vec4<f32>(value, 0.0, 1.0);
        }
        @vertex fn second(@location(0) value: vec4<f32>) -> @builtin(position) vec4<f32> {
            return value;
        }
    "#;
    let request = request(source)
        .with_entry_point(EntryPoint::new("first", ShaderStage::Vertex))
        .with_entry_point(EntryPoint::new("second", ShaderStage::Vertex));
    assert!(matches!(
        ShaderComposer::new().compose(&request),
        Err(fluxel_shader::ShaderError::MultipleVertexEntries)
    ));
}

#[test]
fn specified_binding_shape_and_visibility_must_match_actual_program() {
    let source = "@group(0) @binding(0) var<uniform> value: vec2<f32>; @compute @workgroup_size(1) fn main() { _ = value; }";
    let base = request(source).with_entry_point(EntryPoint::new("main", ShaderStage::Compute));
    let wrong_shape = base.clone().with_binding(
        BindingDeclaration::new(0, 0, BindingKind::Uniform)
            .with_shape(BindingShape::Uniform(VertexType::Vec4(ScalarType::F32))),
    );
    assert!(ShaderComposer::new().compose(&wrong_shape).is_err());
    let wrong_stage = base.with_binding(
        BindingDeclaration::new(0, 0, BindingKind::Uniform)
            .with_shape(BindingShape::Uniform(VertexType::Vec2(ScalarType::F32)))
            .with_visibility([ShaderStage::Fragment]),
    );
    assert!(ShaderComposer::new().compose(&wrong_stage).is_err());
}

#[test]
fn access_is_attributed_to_each_selected_entry() {
    let source = "@group(0) @binding(0) var<storage, read_write> data: array<u32>;\n@compute @workgroup_size(1) fn read_main() { let value = data[0]; _ = value; }\n@compute @workgroup_size(1) fn write_main() { data[0] = 7u; }";
    let read = EntryPoint::new("read_main", ShaderStage::Compute);
    let write = EntryPoint::new("write_main", ShaderStage::Compute);
    let request = request(source)
        .with_entry_point(read.clone())
        .with_entry_point(write.clone())
        .with_binding(BindingDeclaration::new(0, 0, BindingKind::StorageReadWrite));
    let artifact = ShaderComposer::new().compose(&request).unwrap();
    let binding = &artifact.interface().bindings()[0];
    assert!(binding.entry_access[&read].read);
    assert!(!binding.entry_access[&read].write);
    assert!(binding.entry_access[&write].write);
    assert!(!binding.entry_access[&write].read);
}
