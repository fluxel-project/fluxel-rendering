//! Tests for typed material-node authoring and its generated WGSL wrappers.

use std::collections::BTreeMap;

use crate::{
    FragmentInput, MaterialCodeNode, MaterialCompiler, MaterialDefinition, MaterialError,
    MaterialFragment, MaterialNode, MaterialNodeGraph, MaterialNodeId, MaterialNodeInput,
    MaterialNodeInputSource, MaterialType, SurfaceOutput,
};
use fluxel_shader::{
    EntryPoint, ShaderComposer, ShaderCompositionRequest, ShaderStage, SourceUnit,
};

fn fragment(
    name: &str,
    inputs: Vec<FragmentInput>,
    output: MaterialType,
    wgsl: &str,
) -> MaterialFragment {
    MaterialFragment {
        name: name.into(),
        inputs,
        output,
        calls: Vec::new(),
        wgsl: wgsl.into(),
    }
}

fn definition(graph: MaterialNodeGraph) -> MaterialDefinition {
    MaterialDefinition::new(
        vec![
            fragment(
                "base",
                Vec::new(),
                MaterialType::Vec4,
                "fn base() -> vec4<f32> { return vec4<f32>(0.25, 0.5, 0.75, 1.0); }",
            ),
            fragment(
                "tint",
                vec![FragmentInput {
                    name: "color".into(),
                    ty: MaterialType::Vec4,
                }],
                MaterialType::Vec4,
                "fn tint(color: vec4<f32>) -> vec4<f32> { return color * vec4<f32>(0.5); }",
            ),
            fragment(
                "scalar",
                Vec::new(),
                MaterialType::F32,
                "fn scalar() -> f32 { return 1.0; }",
            ),
        ],
        "legacy_unused_root",
        Vec::new(),
        Vec::new(),
        BTreeMap::new(),
        SurfaceOutput::Color,
    )
    .with_node_graph(graph)
}

fn graph(nodes: Vec<MaterialNode>, output: u32) -> MaterialNodeGraph {
    MaterialNodeGraph {
        nodes,
        output: MaterialNodeId(output),
    }
}

#[test]
fn typed_connections_lower_to_deterministic_wgsl_and_compose() {
    let definition = definition(graph(
        vec![
            MaterialNode {
                id: MaterialNodeId(20),
                callable: "tint".into(),
                inputs: vec![MaterialNodeInput {
                    input: "color".into(),
                    source: MaterialNodeInputSource::Node(MaterialNodeId(10)),
                }],
            },
            MaterialNode {
                id: MaterialNodeId(10),
                callable: "base".into(),
                inputs: Vec::new(),
            },
        ],
        20,
    ));
    let variant = MaterialCompiler::compile(&definition).unwrap();
    let source = variant.contribution().source();
    assert!(source.contains("let fluxel_material_node_10_value: vec4<f32> = base();"));
    assert!(source.contains(
        "let fluxel_material_node_20_value: vec4<f32> = tint(fluxel_material_node_10_value);"
    ));
    assert!(source.contains("return fluxel_material_node_20();"));

    let contribution = variant.contribution();
    let mut request = ShaderCompositionRequest::new({
        let mut units = contribution.source_units();
        units.push(SourceUnit::new(
            "node-graph-test-template",
            "@fragment fn main() -> @location(0) vec4<f32> { return evaluate_material(); }",
        ));
        units
    })
    .with_entry_point(EntryPoint::new("main", ShaderStage::Fragment));
    request.callables = contribution.callable_declarations();
    ShaderComposer::new().compose(&request).unwrap();
}

#[test]
fn node_graph_rejects_invalid_ports_types_and_all_cycles_before_shader_work() {
    let missing = definition(graph(
        vec![MaterialNode {
            id: MaterialNodeId(1),
            callable: "tint".into(),
            inputs: Vec::new(),
        }],
        1,
    ));
    assert!(matches!(
        MaterialCompiler::compile(&missing),
        Err(MaterialError::MissingNodeInput { node: MaterialNodeId(1), input }) if input == "color"
    ));

    let wrong_type = definition(graph(
        vec![
            MaterialNode {
                id: MaterialNodeId(1),
                callable: "tint".into(),
                inputs: vec![MaterialNodeInput {
                    input: "color".into(),
                    source: MaterialNodeInputSource::Node(MaterialNodeId(2)),
                }],
            },
            MaterialNode {
                id: MaterialNodeId(2),
                callable: "scalar".into(),
                inputs: Vec::new(),
            },
        ],
        1,
    ));
    assert!(matches!(
        MaterialCompiler::compile(&wrong_type),
        Err(MaterialError::NodeInputTypeMismatch { node: MaterialNodeId(1), input, expected: MaterialType::Vec4, actual: MaterialType::F32 }) if input == "color"
    ));

    let unreachable_cycle = definition(graph(
        vec![
            MaterialNode {
                id: MaterialNodeId(1),
                callable: "base".into(),
                inputs: Vec::new(),
            },
            MaterialNode {
                id: MaterialNodeId(2),
                callable: "tint".into(),
                inputs: vec![MaterialNodeInput {
                    input: "color".into(),
                    source: MaterialNodeInputSource::Node(MaterialNodeId(3)),
                }],
            },
            MaterialNode {
                id: MaterialNodeId(3),
                callable: "tint".into(),
                inputs: vec![MaterialNodeInput {
                    input: "color".into(),
                    source: MaterialNodeInputSource::Node(MaterialNodeId(2)),
                }],
            },
        ],
        1,
    ));
    assert!(matches!(
        MaterialCompiler::compile(&unreachable_cycle),
        Err(MaterialError::NodeCycle { .. })
    ));
}

#[test]
fn node_graph_reserves_generated_wrappers_and_callable_input_names() {
    let reentrant = MaterialDefinition::new(
        vec![fragment(
            "reentrant",
            Vec::new(),
            MaterialType::Vec4,
            "fn reentrant() -> vec4<f32> { return fluxel_material_node_1(); }",
        )],
        "reentrant",
        Vec::new(),
        Vec::new(),
        BTreeMap::new(),
        SurfaceOutput::Color,
    )
    .with_node_graph(graph(
        vec![MaterialNode {
            id: MaterialNodeId(1),
            callable: "reentrant".into(),
            inputs: Vec::new(),
        }],
        1,
    ));
    assert!(matches!(
        MaterialCompiler::compile(&reentrant),
        Err(MaterialError::GeneratedNodeReference { source, generated })
            if source == "reentrant" && generated == "fluxel_material_node_1"
    ));

    let duplicate_inputs = MaterialDefinition::new(
        vec![fragment(
            "root",
            vec![
                FragmentInput {
                    name: "value".into(),
                    ty: MaterialType::Vec4,
                },
                FragmentInput {
                    name: "value".into(),
                    ty: MaterialType::Vec4,
                },
            ],
            MaterialType::Vec4,
            "fn root(value: vec4<f32>, other: vec4<f32>) -> vec4<f32> { return value + other; }",
        )],
        "root",
        Vec::new(),
        Vec::new(),
        BTreeMap::new(),
        SurfaceOutput::Color,
    );
    assert!(matches!(
        MaterialCompiler::compile(&duplicate_inputs),
        Err(MaterialError::DuplicateCallableInput { callable, input })
            if callable == "root" && input == "value"
    ));
}

#[test]
fn node_graph_rejects_code_node_wrapper_references_but_ignores_comments() {
    let code_reference = definition(graph(
        vec![MaterialNode {
            id: MaterialNodeId(1),
            callable: "base".into(),
            inputs: Vec::new(),
        }],
        1,
    ))
    .with_code_node(MaterialCodeNode {
        name: "author_code".into(),
        inputs: Vec::new(),
        output: MaterialType::Vec4,
        calls: Vec::new(),
        source: SourceUnit::new(
            "node-graph-code-reference",
            "fn author_code() -> vec4<f32> { return fluxel_material_node_1(); }",
        ),
        resources: Vec::new(),
    });
    assert!(matches!(
        MaterialCompiler::compile(&code_reference),
        Err(MaterialError::GeneratedNodeReference { source, generated })
            if source == "author_code" && generated == "fluxel_material_node_1"
    ));

    let comment_only = MaterialDefinition::new(
        vec![fragment(
            "base",
            Vec::new(),
            MaterialType::Vec4,
            "// fluxel_material_node_1 is intentionally only a comment\nfn base() -> vec4<f32> { return vec4<f32>(1.0); }",
        )],
        "ignored",
        Vec::new(),
        Vec::new(),
        BTreeMap::new(),
        SurfaceOutput::Color,
    )
    .with_node_graph(graph(
        vec![MaterialNode {
            id: MaterialNodeId(1),
            callable: "base".into(),
            inputs: Vec::new(),
        }],
        1,
    ));
    MaterialCompiler::compile(&comment_only).expect("comments do not create generated-node edges");
}

#[test]
fn uv0_source_requires_a_uv0_surface_evaluator() {
    let definition = definition(graph(
        vec![MaterialNode {
            id: MaterialNodeId(1),
            callable: "tint".into(),
            inputs: vec![MaterialNodeInput {
                input: "color".into(),
                source: MaterialNodeInputSource::Uv0,
            }],
        }],
        1,
    ));
    assert!(matches!(
        MaterialCompiler::compile(&definition),
        Err(MaterialError::NodeContextUnavailable { node: MaterialNodeId(1), input }) if input == "color"
    ));
}

#[test]
fn color_uv0_node_graph_generates_a_typed_unlit_root() {
    let definition = MaterialDefinition::new(
        Vec::new(),
        "unused",
        Vec::new(),
        Vec::new(),
        BTreeMap::new(),
        SurfaceOutput::ColorUv0,
    )
    .with_code_node(MaterialCodeNode {
        name: "sample".into(),
        inputs: vec![FragmentInput {
            name: "uv".into(),
            ty: MaterialType::Vec2,
        }],
        output: MaterialType::Vec4,
        calls: Vec::new(),
        source: fluxel_shader::SourceUnit::new(
            "color-uv0-test",
            "fn sample(uv: vec2<f32>) -> vec4<f32> { return vec4<f32>(uv, 0.0, 1.0); }",
        ),
        resources: Vec::new(),
    })
    .with_node_graph(graph(
        vec![MaterialNode {
            id: MaterialNodeId(9),
            callable: "sample".into(),
            inputs: vec![MaterialNodeInput {
                input: "uv".into(),
                source: MaterialNodeInputSource::Uv0,
            }],
        }],
        9,
    ));
    let source = MaterialCompiler::compile(&definition)
        .expect("ColorUv0 accepts a typed UV0 node root")
        .contribution()
        .source()
        .to_owned();
    assert!(source.contains("fn fluxel_material_node_9(uv: vec2<f32>)"));
    assert!(source.contains("fn evaluate_material_uv0(uv: vec2<f32>)"));
}
