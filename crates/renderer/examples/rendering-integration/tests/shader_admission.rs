#[path = "../src/shader_admission.rs"]
mod shader_admission;

use fluxel_rhi::api::{DeviceCapabilities, ShaderEntryPoint, ShaderStage as RhiShaderStage};
use fluxel_shader::NagaCapability;
use fluxel_shader::{
    EntryPoint, ShaderComposer, ShaderCompositionRequest, ShaderStage, SourceUnit,
};
use shader_admission::{AdmissionError, NonuniformFacts, PipelineCapabilityFacts, admit, demand};

fn demands(names: &[&str]) -> Vec<String> {
    names.iter().map(|name| (*name).into()).collect()
}

fn artifact(source: &str, stage: ShaderStage) -> std::sync::Arc<fluxel_shader::ShaderArtifact> {
    let request = ShaderCompositionRequest::new(vec![SourceUnit::new("admission", source)])
        .with_entry_point(EntryPoint::new("main", stage));
    ShaderComposer::new().compose(&request).unwrap()
}

#[test]
fn reflected_multiview_is_admitted_only_by_actual_capability_and_view_limit() {
    let artifact = artifact(
        "@fragment fn main(@builtin(view_index) view: i32) -> @location(0) vec4<f32> { return vec4<f32>(f32(view)); }",
        ShaderStage::Fragment,
    );
    assert!(
        artifact
            .requirements()
            .naga_capabilities
            .iter()
            .any(|capability| capability.name() == demand::MULTIVIEW)
    );
    let demands = artifact
        .requirements()
        .naga_capabilities
        .iter()
        .map(|capability| capability.name().to_owned())
        .collect::<Vec<_>>();
    let facts = PipelineCapabilityFacts {
        multiview_views: Some(2),
        multisample_count: 1,
        ..Default::default()
    };
    let mut unsupported = DeviceCapabilities::default();
    unsupported.limits.max_multiview_views = 2;
    assert!(
        matches!(admit(&demands, &unsupported, facts), Err(AdmissionError::Unsupported { demand }) if demand == shader_admission::demand::MULTIVIEW)
    );
    let mut supported = unsupported;
    supported.multiview = true;
    assert!(admit(&demands, &supported, facts).is_ok());
    let too_many = PipelineCapabilityFacts {
        multiview_views: Some(3),
        ..facts
    };
    assert!(
        matches!(admit(&demands, &supported, too_many), Err(AdmissionError::Unsupported { demand }) if demand == shader_admission::demand::MULTIVIEW)
    );
}

#[test]
fn reflected_f16_is_unrepresentable_to_this_rhi_admission_layer() {
    let artifact = artifact(
        "enable f16; @compute @workgroup_size(1) fn main() { let value: f16 = 1.0h; _ = value; }",
        ShaderStage::Compute,
    );
    let demands = artifact
        .requirements()
        .naga_capabilities
        .iter()
        .map(|capability| capability.name().to_owned())
        .collect::<Vec<_>>();
    let f16 = demands
        .iter()
        .find(|demand| demand.starts_with("shader-float16"))
        .unwrap();
    assert!(
        matches!(admit(&demands, &DeviceCapabilities::default(), PipelineCapabilityFacts { multisample_count: 1, ..Default::default() }), Err(AdmissionError::Unrepresentable { demand }) if demand == *f16)
    );
}

#[test]
fn admits_only_when_public_capabilities_and_actual_pipeline_facts_agree() {
    let mut capabilities = DeviceCapabilities::default();
    capabilities.multiview = true;
    capabilities.limits.max_multiview_views = 4;
    capabilities.push_constants = true;
    capabilities.limits.max_push_constant_bytes = 32;
    capabilities.binding_arrays = true;
    capabilities.storage_buffers = true;
    capabilities.nonuniform_sampled_texture_indexing = true;
    capabilities.nonuniform_storage_buffer_indexing = true;
    let facts = PipelineCapabilityFacts {
        multiview_views: Some(2),
        push_constant_bytes: 16,
        multisample_count: 1,
        nonuniform: NonuniformFacts {
            binding_arrays: true,
            sampled_textures: true,
            storage_buffers: true,
            ..Default::default()
        },
    };

    assert_eq!(
        admit(
            &demands(&[
                demand::MULTIVIEW,
                demand::PUSH_CONSTANT,
                demand::NONUNIFORM_SAMPLED_TEXTURE_AND_STORAGE_BUFFER,
            ]),
            &capabilities,
            facts,
        ),
        Ok(())
    );
}

#[test]
fn multiview_and_push_constants_check_the_real_requested_limits() {
    let mut capabilities = DeviceCapabilities::default();
    capabilities.multiview = true;
    capabilities.limits.max_multiview_views = 2;
    capabilities.push_constants = true;
    capabilities.limits.max_push_constant_bytes = 16;

    assert_eq!(
        admit(
            &demands(&[demand::MULTIVIEW]),
            &capabilities,
            PipelineCapabilityFacts {
                multiview_views: Some(3),
                multisample_count: 1,
                ..Default::default()
            },
        ),
        Err(AdmissionError::Unsupported {
            demand: demand::MULTIVIEW.into()
        })
    );
    assert_eq!(
        admit(
            &demands(&[demand::PUSH_CONSTANT]),
            &capabilities,
            PipelineCapabilityFacts {
                push_constant_bytes: 20,
                multisample_count: 1,
                ..Default::default()
            },
        ),
        Err(AdmissionError::Unsupported {
            demand: demand::PUSH_CONSTANT.into()
        })
    );
}

#[test]
fn nonuniform_checks_are_descriptor_kind_specific() {
    let mut capabilities = DeviceCapabilities::default();
    capabilities.binding_arrays = true;
    capabilities.nonuniform_sampled_texture_indexing = true;

    assert_eq!(
        admit(
            &demands(&[demand::NONUNIFORM_STORAGE_TEXTURE]),
            &capabilities,
            PipelineCapabilityFacts {
                multisample_count: 1,
                nonuniform: NonuniformFacts {
                    binding_arrays: true,
                    storage_textures: true,
                    ..Default::default()
                },
                ..Default::default()
            },
        ),
        Err(AdmissionError::Unsupported {
            demand: demand::NONUNIFORM_STORAGE_TEXTURE.into()
        })
    );
}

#[test]
fn nonuniform_requires_the_device_base_array_and_resource_capabilities() {
    let facts = PipelineCapabilityFacts {
        multisample_count: 1,
        nonuniform: NonuniformFacts {
            binding_arrays: true,
            storage_buffers: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut capabilities = DeviceCapabilities::default();
    capabilities.storage_buffers = true;
    capabilities.nonuniform_storage_buffer_indexing = true;
    assert_eq!(
        admit(
            &demands(&[demand::NONUNIFORM_SAMPLED_TEXTURE_AND_STORAGE_BUFFER]),
            &capabilities,
            facts,
        ),
        Err(AdmissionError::Unsupported {
            demand: demand::NONUNIFORM_SAMPLED_TEXTURE_AND_STORAGE_BUFFER.into()
        })
    );

    capabilities.binding_arrays = true;
    capabilities.storage_buffers = false;
    assert_eq!(
        admit(
            &demands(&[demand::NONUNIFORM_SAMPLED_TEXTURE_AND_STORAGE_BUFFER]),
            &capabilities,
            facts,
        ),
        Err(AdmissionError::Unsupported {
            demand: demand::NONUNIFORM_SAMPLED_TEXTURE_AND_STORAGE_BUFFER.into()
        })
    );
}

#[test]
fn compute_without_multisample_demand_does_not_require_a_sample_count() {
    assert_eq!(
        admit(
            &demands(&[demand::PUSH_CONSTANT]),
            &DeviceCapabilities {
                push_constants: true,
                limits: fluxel_rhi::api::DeviceLimits {
                    max_push_constant_bytes: 4,
                    ..Default::default()
                },
                ..Default::default()
            },
            PipelineCapabilityFacts {
                push_constant_bytes: 4,
                multisample_count: 0,
                ..Default::default()
            },
        ),
        Ok(())
    );
}

#[test]
fn multisample_and_unknown_demands_are_unrepresentable_not_backend_guesses() {
    assert_eq!(
        admit(
            &demands(&[demand::MULTISAMPLED_SHADING]),
            &DeviceCapabilities::default(),
            PipelineCapabilityFacts {
                multisample_count: 4,
                ..Default::default()
            },
        ),
        Err(AdmissionError::Unrepresentable {
            demand: demand::MULTISAMPLED_SHADING.into()
        })
    );
    assert_eq!(
        admit(
            &demands(&["float64"]),
            &DeviceCapabilities::default(),
            PipelineCapabilityFacts {
                multisample_count: 1,
                ..Default::default()
            },
        ),
        Err(AdmissionError::Unrepresentable {
            demand: "float64".into()
        })
    );
}

#[test]
fn admission_errors_are_propagatable_std_errors() {
    let error = AdmissionError::Unsupported {
        demand: demand::MULTIVIEW.into(),
    };
    let propagated: &dyn std::error::Error = &error;

    assert_eq!(
        propagated.to_string(),
        "device does not support shader demand `multiview`"
    );
}

#[test]
fn collector_route_capability_names_are_not_rejected_by_consumer_admission() {
    let collector_name = NagaCapability::EarlyDepthTest.name();
    assert_eq!(collector_name, demand::RHI_ROUTE_EARLY_DEPTH_TEST);

    assert_eq!(
        admit(
            &demands(&[collector_name]),
            &DeviceCapabilities::default(),
            PipelineCapabilityFacts::default(),
        ),
        Ok(())
    );
}

#[test]
fn write_evidence_persists_escaped_success_and_refusal_before_route_work() {
    let path = std::env::temp_dir().join(format!("fluxel-admission-{}.json", std::process::id()));
    let mut capabilities = DeviceCapabilities::default();
    capabilities.multiview = true;
    capabilities.limits.max_multiview_views = 2;
    let facts = PipelineCapabilityFacts {
        multiview_views: Some(2),
        multisample_count: 1,
        ..Default::default()
    };
    let entries = [
        ShaderEntryPoint {
            name: "quoted\"vertex".into(),
            stage: RhiShaderStage::Vertex,
            workgroup_size: None,
        },
        ShaderEntryPoint {
            name: "compute".into(),
            stage: RhiShaderStage::Compute,
            workgroup_size: Some([8, 4, 1]),
        },
    ];
    let accepted = admit(&demands(&[demand::MULTIVIEW]), &capabilities, facts);
    shader_admission::write_evidence(
        &path,
        &demands(&[demand::MULTIVIEW]),
        &["quoted\"feature".into()],
        &entries,
        &capabilities,
        facts,
        &accepted,
    )
    .unwrap();
    let evidence = std::fs::read_to_string(&path).unwrap();
    assert!(evidence.contains("\"admission\": {\"status\":\"accepted\"}"));
    assert!(evidence.contains("quoted\\\"feature"));
    assert!(evidence.contains("\"targetEntries\": [{\"name\":\"quoted\\\"vertex\",\"stage\":\"vertex\",\"workgroupSize\":null},{\"name\":\"compute\",\"stage\":\"compute\",\"workgroupSize\":[8,4,1]}]"));

    let refused = admit(&demands(&["shader-float16"]), &capabilities, facts);
    assert!(matches!(
        refused,
        Err(AdmissionError::Unrepresentable { .. })
    ));
    shader_admission::write_evidence(
        &path,
        &demands(&["shader-float16"]),
        &[],
        &entries,
        &capabilities,
        facts,
        &refused,
    )
    .unwrap();
    let evidence = std::fs::read_to_string(&path).unwrap();
    assert!(evidence.contains("\"kind\":\"unrepresentable\""));
    assert!(evidence.contains("shader-float16"));

    let unsupported_caps = DeviceCapabilities::default();
    let unsupported = admit(&demands(&[demand::MULTIVIEW]), &unsupported_caps, facts);
    assert!(matches!(
        unsupported,
        Err(AdmissionError::Unsupported { .. })
    ));
    shader_admission::write_evidence(
        &path,
        &demands(&[demand::MULTIVIEW]),
        &[],
        &entries,
        &unsupported_caps,
        facts,
        &unsupported,
    )
    .unwrap();
    assert!(
        std::fs::read_to_string(&path)
            .unwrap()
            .contains("\"kind\":\"unsupported\"")
    );
    std::fs::remove_file(path).unwrap();
}
