//! Consumer-side admission for logical shader capability demands.
//!
//! This module only consumes public RHI capability and limit facts. It does
//! not infer support from a backend identity, and it refuses to claim support
//! for a demand that the selected RHI revision cannot represent.

use std::{fmt, fs, io, path::Path};

use fluxel_rhi::api::{DeviceCapabilities, ShaderEntryPoint, ShaderStage};

pub(crate) fn json_string(value: &str) -> String {
    let mut escaped = String::from("\"");
    for character in value.chars() {
        match character {
            '\"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            value if value.is_control() => escaped.push_str(&format!("\\u{:04x}", value as u32)),
            value => escaped.push(value),
        }
    }
    escaped.push('\"');
    escaped
}

/// Naga capability demand names understood by this consumer admission layer.
pub(crate) mod demand {
    pub(crate) const MULTIVIEW: &str = "multiview";
    pub(crate) const PUSH_CONSTANT: &str = "push-constant";
    pub(crate) const MULTISAMPLED_SHADING: &str = "multisampled-shading";
    pub(crate) const NONUNIFORM_UNIFORM_BUFFER: &str = "uniform-buffer-array-non-uniform-indexing";
    pub(crate) const NONUNIFORM_SAMPLED_TEXTURE_AND_STORAGE_BUFFER: &str =
        "sampled-texture-and-storage-buffer-array-non-uniform-indexing";
    pub(crate) const NONUNIFORM_STORAGE_TEXTURE: &str =
        "storage-texture-array-non-uniform-indexing";
    pub(crate) const NONUNIFORM_SAMPLER: &str = "sampler-non-uniform-indexing";

    /// These requirements are represented by `ShaderDesc.required_features` and
    /// must be admitted by the selected RHI route, not by generic capabilities.
    pub(crate) const RHI_ROUTE_PRIMITIVE_INDEX: &str = "primitive-index";
    pub(crate) const RHI_ROUTE_EARLY_DEPTH_TEST: &str = "early-depth-test";
    pub(crate) const RHI_ROUTE_RAY_QUERY: &str = "ray-query";
}

/// Concrete non-uniform resource uses in the physical pipeline being admitted.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct NonuniformFacts {
    pub(crate) uniform_buffers: bool,
    pub(crate) storage_buffers: bool,
    pub(crate) sampled_textures: bool,
    pub(crate) storage_textures: bool,
    pub(crate) samplers: bool,
    pub(crate) binding_arrays: bool,
    pub(crate) runtime_sized_binding_arrays: bool,
}

/// Actual pipeline properties that determine whether a demand is meaningful.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct PipelineCapabilityFacts {
    /// `None` means the concrete pipeline does not request multiview.
    pub(crate) multiview_views: Option<u32>,
    /// Total push-constant byte span present in the concrete pipeline layout.
    pub(crate) push_constant_bytes: u32,
    /// Actual raster sample count. Zero is never a valid physical pipeline fact.
    pub(crate) multisample_count: u32,
    /// Concrete descriptor kinds indexed non-uniformly by the pipeline.
    pub(crate) nonuniform: NonuniformFacts,
}

/// A device admission refusal with no backend-name-specific interpretation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum AdmissionError {
    /// The concrete pipeline facts contradict the declared demand.
    InvalidPipelineFacts { demand: String },
    /// RHI publishes a negative capability or insufficient public limit.
    Unsupported { demand: String },
    /// This RHI revision exposes no complete public fact for this demand.
    Unrepresentable { demand: String },
}

impl fmt::Display for AdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPipelineFacts { demand } => {
                write!(
                    formatter,
                    "invalid concrete pipeline facts for shader demand `{demand}`"
                )
            }
            Self::Unsupported { demand } => {
                write!(
                    formatter,
                    "device does not support shader demand `{demand}`"
                )
            }
            Self::Unrepresentable { demand } => write!(
                formatter,
                "selected RHI revision cannot represent shader demand `{demand}`"
            ),
        }
    }
}

impl std::error::Error for AdmissionError {}

/// Checks each logical demand against public device facts and actual pipeline facts.
pub(crate) fn admit(
    demands: &[String],
    capabilities: &DeviceCapabilities,
    facts: PipelineCapabilityFacts,
) -> Result<(), AdmissionError> {
    for demand in demands {
        match demand.as_str() {
            demand::MULTIVIEW => admit_multiview(capabilities, facts)?,
            demand::PUSH_CONSTANT => admit_push_constants(capabilities, facts)?,
            demand::MULTISAMPLED_SHADING => {
                if facts.multisample_count < 2 {
                    return Err(AdmissionError::InvalidPipelineFacts {
                        demand: demand.clone(),
                    });
                }
                // The selected RHI revision has no public multisampled-shading
                // capability or sample-count limit. Do not treat a route or
                // backend identity as evidence of support.
                return Err(AdmissionError::Unrepresentable {
                    demand: demand.clone(),
                });
            }
            demand::NONUNIFORM_UNIFORM_BUFFER => admit_nonuniform(
                demand,
                capabilities.nonuniform_uniform_buffer_indexing,
                capabilities.runtime_sized_binding_arrays,
                true,
                capabilities.binding_arrays,
                facts.nonuniform.uniform_buffers,
                facts.nonuniform,
            )?,
            demand::NONUNIFORM_SAMPLED_TEXTURE_AND_STORAGE_BUFFER => {
                admit_sampled_texture_and_storage_buffer_nonuniform(
                    demand,
                    capabilities,
                    facts.nonuniform,
                )?
            }
            demand::NONUNIFORM_STORAGE_TEXTURE => admit_nonuniform(
                demand,
                capabilities.nonuniform_storage_texture_indexing,
                capabilities.runtime_sized_binding_arrays,
                capabilities.storage_textures,
                capabilities.binding_arrays,
                facts.nonuniform.storage_textures,
                facts.nonuniform,
            )?,
            demand::NONUNIFORM_SAMPLER => admit_nonuniform(
                demand,
                capabilities.nonuniform_sampler_indexing,
                capabilities.runtime_sized_binding_arrays,
                true,
                capabilities.binding_arrays,
                facts.nonuniform.samplers,
                facts.nonuniform,
            )?,
            demand::RHI_ROUTE_PRIMITIVE_INDEX
            | demand::RHI_ROUTE_EARLY_DEPTH_TEST
            | demand::RHI_ROUTE_RAY_QUERY => {
                // These are passed through to `ShaderDesc.required_features`.
                // Device::shader_route owns route-specific feature admission.
            }
            _ => {
                return Err(AdmissionError::Unrepresentable {
                    demand: demand.clone(),
                });
            }
        }
    }
    Ok(())
}

/// Writes an independently reviewable admission request and result before any route query.
pub(crate) fn write_evidence(
    output: &Path,
    demands: &[String],
    rhi_feature_names: &[String],
    target_entries: &[ShaderEntryPoint],
    capabilities: &DeviceCapabilities,
    facts: PipelineCapabilityFacts,
    result: &Result<(), AdmissionError>,
) -> io::Result<()> {
    let per_demand = demands
        .iter()
        .map(|demand| {
            format!(
                "{{\"demand\":{},\"outcome\":{}}}",
                json_string(demand),
                admission_json(&admit(std::slice::from_ref(demand), capabilities, facts)),
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let demands = demands
        .iter()
        .map(|demand| json_string(demand))
        .collect::<Vec<_>>()
        .join(",");
    let rhi_feature_names = rhi_feature_names
        .iter()
        .map(|name| json_string(name))
        .collect::<Vec<_>>()
        .join(",");
    let target_entries = target_entries
        .iter()
        .map(|entry| {
            let stage = match entry.stage {
                ShaderStage::Vertex => "vertex",
                ShaderStage::Fragment => "fragment",
                ShaderStage::Compute => "compute",
                ShaderStage::Task => "task",
                ShaderStage::Mesh => "mesh",
                ShaderStage::RayGeneration => "ray-generation",
                ShaderStage::AnyHit => "any-hit",
                ShaderStage::ClosestHit => "closest-hit",
                ShaderStage::Miss => "miss",
                ShaderStage::Intersection => "intersection",
                ShaderStage::Callable => "callable",
            };
            let workgroup_size = entry
                .workgroup_size
                .map_or_else(|| "null".to_owned(), |[x, y, z]| format!("[{x},{y},{z}]"));
            format!(
                "{{\"name\":{},\"stage\":{},\"workgroupSize\":{workgroup_size}}}",
                json_string(&entry.name),
                json_string(stage),
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let multiview_views = facts
        .multiview_views
        .map_or_else(|| "null".into(), |views| views.to_string());
    let limits = &capabilities.limits;
    let outcome = admission_json(result);
    fs::write(
        output,
        format!(
            concat!(
                "{{\n",
                "  \"schemaVersion\": 1,\n",
                "  \"nagaCapabilityDemands\": [{}],\n",
                "  \"rhiFeatureNames\": [{}],\n",
                "  \"targetEntries\": [{}],\n",
                "  \"deviceCapabilities\": {{\"multiview\":{},\"pushConstants\":{},\"bindingArrays\":{},\"runtimeSizedBindingArrays\":{},\"storageBuffers\":{},\"storageTextures\":{},\"nonuniformUniformBufferIndexing\":{},\"nonuniformStorageBufferIndexing\":{},\"nonuniformSampledTextureIndexing\":{},\"nonuniformStorageTextureIndexing\":{},\"nonuniformSamplerIndexing\":{},\"limits\":{{\"maxMultiviewViews\":{},\"maxPushConstantBytes\":{}}}}},\n",
                "  \"pipelineFacts\": {{\"multiviewViews\":{},\"pushConstantBytes\":{},\"multisampleCount\":{},\"nonuniform\":{{\"uniformBuffers\":{},\"storageBuffers\":{},\"sampledTextures\":{},\"storageTextures\":{},\"samplers\":{},\"bindingArrays\":{},\"runtimeSizedBindingArrays\":{}}}}},\n",
                "  \"perDemand\": [{}],\n",
                "  \"admission\": {}\n",
                "}}\n"
            ),
            demands,
            rhi_feature_names,
            target_entries,
            capabilities.multiview,
            capabilities.push_constants,
            capabilities.binding_arrays,
            capabilities.runtime_sized_binding_arrays,
            capabilities.storage_buffers,
            capabilities.storage_textures,
            capabilities.nonuniform_uniform_buffer_indexing,
            capabilities.nonuniform_storage_buffer_indexing,
            capabilities.nonuniform_sampled_texture_indexing,
            capabilities.nonuniform_storage_texture_indexing,
            capabilities.nonuniform_sampler_indexing,
            limits.max_multiview_views,
            limits.max_push_constant_bytes,
            multiview_views,
            facts.push_constant_bytes,
            facts.multisample_count,
            facts.nonuniform.uniform_buffers,
            facts.nonuniform.storage_buffers,
            facts.nonuniform.sampled_textures,
            facts.nonuniform.storage_textures,
            facts.nonuniform.samplers,
            facts.nonuniform.binding_arrays,
            facts.nonuniform.runtime_sized_binding_arrays,
            per_demand,
            outcome,
        ),
    )
}

fn admission_json(result: &Result<(), AdmissionError>) -> String {
    match result {
        Ok(()) => "{\"status\":\"accepted\"}".to_owned(),
        Err(AdmissionError::InvalidPipelineFacts { demand }) => format!(
            "{{\"status\":\"refused\",\"error\":{{\"kind\":\"invalid-pipeline-facts\",\"demand\":{}}}}}",
            json_string(demand),
        ),
        Err(AdmissionError::Unsupported { demand }) => format!(
            "{{\"status\":\"refused\",\"error\":{{\"kind\":\"unsupported\",\"demand\":{}}}}}",
            json_string(demand),
        ),
        Err(AdmissionError::Unrepresentable { demand }) => format!(
            "{{\"status\":\"refused\",\"error\":{{\"kind\":\"unrepresentable\",\"demand\":{}}}}}",
            json_string(demand),
        ),
    }
}

fn admit_sampled_texture_and_storage_buffer_nonuniform(
    demand: &str,
    capabilities: &DeviceCapabilities,
    facts: NonuniformFacts,
) -> Result<(), AdmissionError> {
    if !facts.sampled_textures && !facts.storage_buffers {
        return Err(AdmissionError::InvalidPipelineFacts {
            demand: demand.into(),
        });
    }

    if facts.sampled_textures {
        admit_nonuniform(
            demand,
            capabilities.nonuniform_sampled_texture_indexing,
            capabilities.runtime_sized_binding_arrays,
            true,
            capabilities.binding_arrays,
            true,
            facts,
        )?;
    }
    if facts.storage_buffers {
        admit_nonuniform(
            demand,
            capabilities.nonuniform_storage_buffer_indexing,
            capabilities.runtime_sized_binding_arrays,
            capabilities.storage_buffers,
            capabilities.binding_arrays,
            true,
            facts,
        )?;
    }
    Ok(())
}

fn admit_multiview(
    capabilities: &DeviceCapabilities,
    facts: PipelineCapabilityFacts,
) -> Result<(), AdmissionError> {
    let Some(views) = facts.multiview_views else {
        return Err(AdmissionError::InvalidPipelineFacts {
            demand: demand::MULTIVIEW.into(),
        });
    };
    if !capabilities.multiview || views == 0 || views > capabilities.limits.max_multiview_views {
        return Err(AdmissionError::Unsupported {
            demand: demand::MULTIVIEW.into(),
        });
    }
    Ok(())
}

fn admit_push_constants(
    capabilities: &DeviceCapabilities,
    facts: PipelineCapabilityFacts,
) -> Result<(), AdmissionError> {
    if facts.push_constant_bytes == 0 {
        return Err(AdmissionError::InvalidPipelineFacts {
            demand: demand::PUSH_CONSTANT.into(),
        });
    }
    if !capabilities.push_constants
        || facts.push_constant_bytes > capabilities.limits.max_push_constant_bytes
    {
        return Err(AdmissionError::Unsupported {
            demand: demand::PUSH_CONSTANT.into(),
        });
    }
    Ok(())
}

fn admit_nonuniform(
    demand: &str,
    supported: bool,
    runtime_sized_supported: bool,
    resource_supported: bool,
    binding_arrays_supported: bool,
    concrete_kind_present: bool,
    facts: NonuniformFacts,
) -> Result<(), AdmissionError> {
    if !concrete_kind_present || !facts.binding_arrays {
        return Err(AdmissionError::InvalidPipelineFacts {
            demand: demand.into(),
        });
    }
    if !binding_arrays_supported
        || !resource_supported
        || !supported
        || (facts.runtime_sized_binding_arrays && !runtime_sized_supported)
    {
        return Err(AdmissionError::Unsupported {
            demand: demand.into(),
        });
    }
    Ok(())
}
