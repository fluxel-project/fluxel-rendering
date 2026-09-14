//! Platform-neutral shader, program, and pipeline-layout vocabulary.

use std::collections::BTreeSet;

pub(crate) use crate::shader_contract::ShaderSourceHash;
use crate::shader_contract::{GlslDialect, ShaderStage};

use super::{GlError, GlFamilyApi, GlFamilyProfile, ProgramId, ShaderId};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) enum GlShaderStage {
    Vertex,
    Fragment,
    Compute,
}

impl From<GlShaderStage> for ShaderStage {
    fn from(value: GlShaderStage) -> Self {
        match value {
            GlShaderStage::Vertex => Self::Vertex,
            GlShaderStage::Fragment => Self::Fragment,
            GlShaderStage::Compute => Self::Compute,
        }
    }
}

/// The exact GLSL family and version emitted by the GL lowering pass.
///
/// This is not an authoring-language choice. A `GlShaderSource` has already
/// been lowered from the private validated artifact and is ready for one GL
/// profile only.
pub(crate) type GlShaderDialect = GlslDialect;
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlShaderSource {
    pub stage: GlShaderStage,
    pub dialect: GlShaderDialect,
    pub entry_point: String,
    pub source_hash: ShaderSourceHash,
    pub text: String,
    pub debug_name: Option<String>,
}
/// Fluxel's logical binding key. It deliberately is not a GL binding point.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct GlBindingLocation {
    pub group: u32,
    pub binding: u32,
}
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) enum GlShaderResourceKind {
    UniformBuffer,
    Sampler,
    Texture,
    CombinedTextureSampler,
}
/// A logical RHI resource declaration, independent of program-link results.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlLogicalBinding {
    pub name: String,
    pub location: GlBindingLocation,
    pub kind: GlShaderResourceKind,
    pub array_count: u32,
}
/// An executable GL assignment selected after program link.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) enum GlExecutableBindingLocation {
    UniformBlock(u32),
    TextureUnit(u32),
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlExecutableBindingAssignment {
    pub logical: GlBindingLocation,
    pub executable: GlExecutableBindingLocation,
}
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlVertexInputReflection {
    pub name: String,
    pub location: u32,
    pub columns: u8,
}
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlFragmentOutputReflection {
    pub name: String,
    pub location: u32,
}
/// Immutable link evidence. Logical bindings and executable locations remain
/// distinct so a cache cannot mistake a GL assignment for a RHI contract.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlProgramReflection {
    pub vertex_inputs: Vec<GlVertexInputReflection>,
    pub fragment_outputs: Vec<GlFragmentOutputReflection>,
    pub assignments: Vec<GlExecutableBindingAssignment>,
}
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlPipelineLayout {
    pub bindings: Vec<GlLogicalBinding>,
}
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlProgramDescriptor {
    pub vertex: GlShaderSource,
    pub fragment: GlShaderSource,
    pub layout: GlPipelineLayout,
    pub debug_name: Option<String>,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GlShaderValidationError {
    WrongStage,
    EmptySource,
    EmptyEntryPoint,
    DialectProfileMismatch,
    ZeroArrayCount,
    DuplicateLogicalBinding,
    DuplicateLogicalAssignment,
    DuplicateExecutableAssignment,
    DuplicateVertexLocation,
    DuplicateFragmentOutputLocation,
    ZeroVertexColumns,
}

impl GlPipelineLayout {
    pub(crate) fn validate(&self) -> Result<(), GlShaderValidationError> {
        let mut locations = BTreeSet::new();
        for binding in &self.bindings {
            if binding.array_count == 0 {
                return Err(GlShaderValidationError::ZeroArrayCount);
            }
            if !locations.insert(binding.location) {
                return Err(GlShaderValidationError::DuplicateLogicalBinding);
            }
        }
        Ok(())
    }
}
impl GlProgramDescriptor {
    pub(crate) fn validate(&self) -> Result<(), GlShaderValidationError> {
        if self.vertex.stage != GlShaderStage::Vertex
            || self.fragment.stage != GlShaderStage::Fragment
        {
            return Err(GlShaderValidationError::WrongStage);
        }
        self.vertex.validate()?;
        self.fragment.validate()?;
        self.layout.validate()
    }

    /// Validates that both already-lowered sources target this exact context.
    pub(crate) fn validate_for(
        &self,
        profile: GlFamilyProfile,
    ) -> Result<(), GlShaderValidationError> {
        self.validate()?;
        self.vertex.validate_for(profile)?;
        self.fragment.validate_for(profile)
    }
}
impl GlShaderSource {
    pub(crate) fn validate(&self) -> Result<(), GlShaderValidationError> {
        if self.text.is_empty() {
            return Err(GlShaderValidationError::EmptySource);
        }
        if self.entry_point.is_empty() {
            return Err(GlShaderValidationError::EmptyEntryPoint);
        }
        Ok(())
    }

    /// Checks the profile-specific target selected by the lowering pass.
    pub(crate) fn validate_for(
        &self,
        profile: GlFamilyProfile,
    ) -> Result<(), GlShaderValidationError> {
        self.validate()?;
        let expected = match profile {
            GlFamilyProfile::WebGl2 => GlShaderDialect::Embedded { version: 300 },
            GlFamilyProfile::Embedded { major: 3, minor: 0 } => {
                GlShaderDialect::Embedded { version: 300 }
            }
            GlFamilyProfile::Embedded { major: 3, minor: 1 } => {
                GlShaderDialect::Embedded { version: 310 }
            }
            GlFamilyProfile::Embedded { major: 3, minor: 2 } => {
                GlShaderDialect::Embedded { version: 320 }
            }
            GlFamilyProfile::Desktop { major: 4, minor } => GlShaderDialect::Desktop {
                version: 400 + u16::from(minor) * 10,
            },
            _ => return Err(GlShaderValidationError::DialectProfileMismatch),
        };
        if self.dialect != expected {
            return Err(GlShaderValidationError::DialectProfileMismatch);
        }
        Ok(())
    }
}
impl GlProgramReflection {
    pub(crate) fn validate_against(
        &self,
        layout: &GlPipelineLayout,
    ) -> Result<(), GlShaderValidationError> {
        layout.validate()?;
        let logical: BTreeSet<_> = layout
            .bindings
            .iter()
            .map(|binding| binding.location)
            .collect();
        let mut assigned_logical = BTreeSet::new();
        let mut executable = BTreeSet::new();
        for assignment in &self.assignments {
            if !logical.contains(&assignment.logical)
                || !assigned_logical.insert(assignment.logical)
            {
                return Err(GlShaderValidationError::DuplicateLogicalAssignment);
            }
            if !executable.insert(assignment.executable) {
                return Err(GlShaderValidationError::DuplicateExecutableAssignment);
            }
        }
        let mut inputs = BTreeSet::new();
        for input in &self.vertex_inputs {
            if input.columns == 0 {
                return Err(GlShaderValidationError::ZeroVertexColumns);
            }
            if !inputs.insert(input.location) {
                return Err(GlShaderValidationError::DuplicateVertexLocation);
            }
        }
        let mut outputs = BTreeSet::new();
        if self
            .fragment_outputs
            .iter()
            .any(|output| !outputs.insert(output.location))
        {
            return Err(GlShaderValidationError::DuplicateFragmentOutputLocation);
        }
        Ok(())
    }
}
pub(crate) trait GlShaderApi: GlFamilyApi {
    fn create_shader(&mut self, source: &GlShaderSource) -> Result<ShaderId, GlError>;
    fn destroy_shader(&mut self, shader: ShaderId) -> Result<(), GlError>;
    fn create_program(
        &mut self,
        descriptor: &GlProgramDescriptor,
    ) -> Result<(ProgramId, GlProgramReflection), GlError>;
    fn destroy_program(&mut self, program: ProgramId) -> Result<(), GlError>;
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_duplicate_logical_bindings() {
        let location = GlBindingLocation {
            group: 0,
            binding: 0,
        };
        let binding = GlLogicalBinding {
            name: "a".into(),
            location,
            kind: GlShaderResourceKind::UniformBuffer,
            array_count: 1,
        };
        let layout = GlPipelineLayout {
            bindings: vec![binding.clone(), binding],
        };
        assert_eq!(
            layout.validate(),
            Err(GlShaderValidationError::DuplicateLogicalBinding)
        );
    }

    fn source(stage: GlShaderStage, dialect: GlShaderDialect) -> GlShaderSource {
        GlShaderSource {
            stage,
            dialect,
            entry_point: "main".into(),
            source_hash: ShaderSourceHash([7; 32]),
            text: "void main() {}".into(),
            debug_name: None,
        }
    }

    #[test]
    fn requires_exact_lowered_dialect_for_profile() {
        let web = source(
            GlShaderStage::Vertex,
            GlShaderDialect::Embedded { version: 300 },
        );
        assert!(web.validate_for(GlFamilyProfile::WebGl2).is_ok());
        assert_eq!(
            web.validate_for(GlFamilyProfile::Desktop { major: 4, minor: 6 }),
            Err(GlShaderValidationError::DialectProfileMismatch)
        );

        let es31 = source(
            GlShaderStage::Compute,
            GlShaderDialect::Embedded { version: 310 },
        );
        assert!(
            es31.validate_for(GlFamilyProfile::Embedded { major: 3, minor: 1 })
                .is_ok()
        );
        assert_eq!(
            es31.validate_for(GlFamilyProfile::Embedded { major: 3, minor: 0 }),
            Err(GlShaderValidationError::DialectProfileMismatch)
        );
    }

    #[test]
    fn rejects_empty_lowered_entry_point() {
        let mut shader = source(
            GlShaderStage::Fragment,
            GlShaderDialect::Desktop { version: 430 },
        );
        shader.entry_point.clear();
        assert_eq!(
            shader.validate_for(GlFamilyProfile::Desktop { major: 4, minor: 3 }),
            Err(GlShaderValidationError::EmptyEntryPoint)
        );
    }
    #[test]
    fn rejects_duplicate_executable_assignments() {
        let a = GlBindingLocation {
            group: 0,
            binding: 0,
        };
        let b = GlBindingLocation {
            group: 0,
            binding: 1,
        };
        let layout = GlPipelineLayout {
            bindings: vec![
                GlLogicalBinding {
                    name: "a".into(),
                    location: a,
                    kind: GlShaderResourceKind::UniformBuffer,
                    array_count: 1,
                },
                GlLogicalBinding {
                    name: "b".into(),
                    location: b,
                    kind: GlShaderResourceKind::UniformBuffer,
                    array_count: 1,
                },
            ],
        };
        let reflection = GlProgramReflection {
            vertex_inputs: vec![],
            fragment_outputs: vec![],
            assignments: vec![
                GlExecutableBindingAssignment {
                    logical: a,
                    executable: GlExecutableBindingLocation::UniformBlock(0),
                },
                GlExecutableBindingAssignment {
                    logical: b,
                    executable: GlExecutableBindingLocation::UniformBlock(0),
                },
            ],
        };
        assert_eq!(
            reflection.validate_against(&layout),
            Err(GlShaderValidationError::DuplicateExecutableAssignment)
        );
    }
}
