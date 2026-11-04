//! Device-fact admission for logical Shader capability demands.
//!
//! This private module consumes only the selected RHI revision's public
//! capability and limit facts. It never infers support from backend identity.

use std::fmt;

use fluxel_rhi::api::DeviceCapabilities;

/// Naga capability demand names understood by Renderer admission.
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

    /// Requirements delegated to `ShaderDesc.required_features` and its
    /// device route query.
    pub(crate) const RHI_ROUTE_PRIMITIVE_INDEX: &str = "primitive-index";
    /// Requirements delegated to `ShaderDesc.required_features` and its
    /// device route query.
    pub(crate) const RHI_ROUTE_EARLY_DEPTH_TEST: &str = "early-depth-test";
    /// Requirements delegated to `ShaderDesc.required_features` and its
    /// device route query.
    pub(crate) const RHI_ROUTE_RAY_QUERY: &str = "ray-query";
}

/// Concrete non-uniform resource uses in a physical pipeline.
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
pub enum AdmissionError {
    /// Concrete pipeline facts contradict the declared demand.
    InvalidPipelineFacts {
        /// The logical capability demand whose physical facts are invalid.
        demand: String,
    },
    /// RHI publishes a negative capability or insufficient public limit.
    Unsupported {
        /// The logical capability demand rejected by device facts.
        demand: String,
    },
    /// This RHI revision exposes no complete public fact for this demand.
    Unrepresentable {
        /// The logical capability demand that lacks a complete public fact.
        demand: String,
    },
}

impl fmt::Display for AdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPipelineFacts { demand } => write!(
                formatter,
                "invalid concrete pipeline facts for shader demand `{demand}`"
            ),
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

/// Checks each logical demand against public device facts and physical pipeline facts.
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
                // capability or sample-count limit. A backend or route name is
                // not evidence of support.
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
                // Device::shader_route owns route-specific admission after the
                // complete ShaderDesc is assembled.
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
