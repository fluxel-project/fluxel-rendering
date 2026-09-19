//! Contract tests for program creation, reflection injection and the
//! shader dialects a program descriptor may carry.

use super::*;

fn compute_program(dialect: GlShaderDialect) -> GlProgramDescriptor {
    GlProgramDescriptor {
        kind: GlProgramKind::Compute {
            shader: GlShaderSource {
                stage: GlShaderStage::Compute,
                dialect,
                entry_point: "main".into(),
                source_hash: ShaderSourceHash([9; 32]),
                text: "void main() {}".into(),
                debug_name: None,
            },
        },
        layout: GlPipelineLayout { bindings: vec![] },
        debug_name: None,
    }
}

#[test]
fn mock_compute_program_requires_proved_capability_and_reflects_empty() {
    let mut web = MockGlFamilyApi::from_discovery(snapshot(GlFamilyProfile::WebGl2));
    let trace_len = web.calls().len();
    assert!(matches!(
        web.create_program(&compute_program(GlShaderDialect::Embedded { version: 300 })),
        Err(GlError::Unsupported { .. })
    ));
    assert!(
        !web.calls()[trace_len..]
            .iter()
            .any(|call| matches!(call, MockCall::CreateProgram(_)))
    );

    let mut desktop = MockGlFamilyApi::from_discovery(compute_storage_snapshot(false));
    let (program, reflection) = desktop
        .create_program(&compute_program(GlShaderDialect::Desktop { version: 430 }))
        .expect("proved compute capability accepts the compute kind");
    assert_eq!(
        reflection,
        GlProgramReflection {
            vertex_inputs: vec![],
            fragment_outputs: vec![],
            assignments: vec![],
        },
        "compute reflection stays empty until the reflection wave"
    );
    assert_eq!(
        desktop.calls().last(),
        Some(&MockCall::CreateProgram(program))
    );
}

#[test]
fn mock_program_reflection_injection_validates_against_layout() {
    let mut api = MockGlFamilyApi::from_discovery(snapshot(GlFamilyProfile::WebGl2));
    let location = GlBindingLocation {
        group: 0,
        binding: 0,
    };
    let descriptor = |name: &'static str| GlProgramDescriptor {
        kind: GlProgramKind::Raster {
            vertex: essl_vertex(),
            fragment: essl_fragment(),
        },
        layout: GlPipelineLayout {
            bindings: vec![GlLogicalBinding {
                name: name.into(),
                location,
                kind: GlShaderResourceKind::UniformBuffer,
                array_count: 1,
            }],
        },
        debug_name: None,
    };
    let matching = GlProgramReflection {
        vertex_inputs: vec![],
        fragment_outputs: vec![],
        assignments: vec![GlExecutableBindingAssignment {
            logical: location,
            executable: GlExecutableBindingLocation::UniformBlock(3),
        }],
    };
    api.set_next_program_reflection(matching);
    let (_, reflection) = api.create_program(&descriptor("Block")).expect("program");
    assert_eq!(
        reflection.assignments,
        vec![GlExecutableBindingAssignment {
            logical: location,
            executable: GlExecutableBindingLocation::UniformBlock(3),
        }]
    );

    let conflicting = GlProgramReflection {
        vertex_inputs: vec![],
        fragment_outputs: vec![],
        assignments: vec![GlExecutableBindingAssignment {
            logical: GlBindingLocation {
                group: 1,
                binding: 0,
            },
            executable: GlExecutableBindingLocation::UniformBlock(0),
        }],
    };
    api.set_next_program_reflection(conflicting);
    assert!(matches!(
        api.create_program(&descriptor("Block")),
        Err(GlError::Validation { .. })
    ));
}

fn essl_vertex() -> GlShaderSource {
    GlShaderSource {
        stage: GlShaderStage::Vertex,
        dialect: crate::shader_contract::GlslDialect::Embedded { version: 300 },
        entry_point: "main".into(),
        source_hash: ShaderSourceHash([1; 32]),
        text: "void main() {}".into(),
        debug_name: None,
    }
}

fn essl_fragment() -> GlShaderSource {
    GlShaderSource {
        stage: GlShaderStage::Fragment,
        dialect: crate::shader_contract::GlslDialect::Embedded { version: 300 },
        entry_point: "main".into(),
        source_hash: ShaderSourceHash([2; 32]),
        text: "void main() {}".into(),
        debug_name: None,
    }
}
