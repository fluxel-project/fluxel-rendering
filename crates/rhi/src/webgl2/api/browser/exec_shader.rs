//! Browser shader, program, reflection, and binding execution.
//!
//! ESSL 300 sources arrive already lowered and profile-validated; this module
//! compiles them, keeps every driver log inside a structured error, reflects
//! the actually linked resources, and rejects any layout/program mismatch
//! before a program object is published.

use web_sys::{WebGl2RenderingContext as Gl, WebGlProgram, WebGlShader};

use super::super::{
    BufferId, GlBindingApi, GlBindingLimits, GlBufferUsage, GlError, GlExecutableBindingAssignment,
    GlExecutableBindingLocation, GlFamilyApi as _, GlProgramDescriptor, GlProgramKind,
    GlProgramReflection, GlShaderApi, GlShaderResourceKind, GlShaderSource, GlShaderStage,
    GlTextureTarget, ProgramId, SamplerId, ShaderId, TextureId, validate_texture_unit,
    validate_uniform_buffer_binding, validate_uniform_range,
};
use super::discovery::WebGl2BrowserDiscovery;
use super::objects::BrowserProgram;

/// ESSL 3.0 sampler uniform types this backend can bind and reflect.
const SAMPLER_TYPES: &[u32] = &[
    0x8B5E, // SAMPLER_2D
    0x8B5F, // SAMPLER_3D
    0x8B60, // SAMPLER_CUBE
    0x8B62, // SAMPLER_2D_SHADOW
    0x8B63, // SAMPLER_2D_ARRAY
    0x8DC5, // SAMPLER_2D_ARRAY_SHADOW
];

/// Column count of an active vertex-input type, if it maps to the common
/// vertex-format vocabulary. Matrices expand to several locations and are
/// rejected instead of being silently collapsed.
fn vertex_input_columns(type_: u32) -> Option<u8> {
    Some(match type_ {
        0x1404..=0x1406 => 1,          // INT, UNSIGNED_INT, FLOAT
        0x8B50 | 0x8B56 | 0x8B57 => 2, // FLOAT_VEC2, INT_VEC2, UNSIGNED_INT_VEC2
        0x8B51 | 0x8B58 | 0x8B59 => 3, // FLOAT_VEC3, INT_VEC3, UNSIGNED_INT_VEC3
        0x8B52 | 0x8B5A | 0x8B5B => 4, // FLOAT_VEC4, INT_VEC4, UNSIGNED_INT_VEC4
        _ => return None,
    })
}

fn is_sampler_type(type_: u32) -> bool {
    SAMPLER_TYPES.contains(&type_)
}

/// Strips the `[0]` suffix WebGL reports for array uniforms.
fn base_uniform_name(name: &str) -> &str {
    name.strip_suffix("[0]").unwrap_or(name)
}

impl GlShaderApi for WebGl2BrowserDiscovery {
    fn create_shader(&mut self, source: &GlShaderSource) -> Result<ShaderId, GlError> {
        const OP: &str = "create-shader";
        self.assert_provider_ready(OP)?;
        source
            .validate_for(self.profile())
            .map_err(|_| Self::validation(OP, "shader source does not target this profile"))?;
        if source.stage == GlShaderStage::Compute {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "ESSL 300 has no compute stage on WebGL2",
            });
        }
        let raw = self.compile_shader_source(OP, source)?;
        let slot = Self::allocate_slot(&mut self.next_shader_slot, OP)?;
        let id = ShaderId::new(self.context_stamp(), slot, 0);
        self.shaders.insert(
            slot,
            super::objects::BrowserShader {
                generation: id.generation,
                raw,
            },
        );
        Ok(id)
    }

    fn destroy_shader(&mut self, shader: ShaderId) -> Result<(), GlError> {
        const OP: &str = "destroy-shader";
        self.shader(OP, shader)?;
        let entry = self
            .shaders
            .remove(&shader.slot)
            .ok_or_else(|| Self::validation(OP, "shader disappeared"))?;
        self.raw.delete_shader(Some(&entry.raw));
        self.driver_error(OP)
    }

    fn create_program(
        &mut self,
        descriptor: &GlProgramDescriptor,
    ) -> Result<(ProgramId, GlProgramReflection), GlError> {
        const OP: &str = "create-program";
        self.assert_provider_ready(OP)?;
        descriptor
            .validate_for(self.profile())
            .map_err(|_| Self::validation(OP, "invalid program descriptor"))?;
        // Plan WebGL2 rule: no compute program domain exists, so the compute
        // kind is rejected structurally before any browser side effect.
        if matches!(descriptor.kind, GlProgramKind::Compute { .. }) {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "WebGL2 has no compute program domain",
            });
        }
        let GlProgramKind::Raster { vertex, fragment } = &descriptor.kind else {
            return Err(Self::validation(OP, "program kind must be raster"));
        };
        let vertex_raw = self.compile_shader_source(OP, vertex)?;
        let fragment_raw = match self.compile_shader_source(OP, fragment) {
            Ok(raw) => raw,
            Err(error) => {
                self.raw.delete_shader(Some(&vertex_raw));
                return Err(error);
            }
        };
        let Some(program) = self.raw.create_program() else {
            self.raw.delete_shader(Some(&vertex_raw));
            self.raw.delete_shader(Some(&fragment_raw));
            return Err(GlError::OutOfMemory { operation: OP });
        };
        self.raw.attach_shader(&program, &vertex_raw);
        self.raw.attach_shader(&program, &fragment_raw);
        self.raw.link_program(&program);
        let linked = self
            .link_succeeded(OP, &program)
            .and_then(|()| self.driver_error(OP));
        // Shaders may be deleted once attached; the program owns its binaries.
        self.raw.delete_shader(Some(&vertex_raw));
        self.raw.delete_shader(Some(&fragment_raw));
        if let Err(error) = linked {
            self.raw.delete_program(Some(&program));
            return Err(error);
        }

        // Sampler-unit assignment needs the program current: `uniform1i`
        // targets the bound program, so reflection runs inside an explicit
        // bind scope and leaves no program selected afterwards (Layer 2 owns
        // the applied-program mirror).
        self.raw.use_program(Some(&program));
        let reflection = match self.reflect_and_assign(OP, &program, descriptor) {
            Ok(reflection) => reflection,
            Err(error) => {
                self.raw.use_program(None);
                self.raw.delete_program(Some(&program));
                return Err(error);
            }
        };
        self.raw.use_program(None);
        if let Err(error) = self.driver_error(OP) {
            self.raw.delete_program(Some(&program));
            return Err(error);
        }
        let slot = Self::allocate_slot(&mut self.next_program_slot, OP)?;
        let id = ProgramId::new(self.context_stamp(), slot, 0);
        self.programs.insert(
            slot,
            BrowserProgram {
                generation: id.generation,
                raw: program,
                descriptor: descriptor.clone(),
            },
        );
        Ok((id, reflection))
    }

    fn destroy_program(&mut self, program: ProgramId) -> Result<(), GlError> {
        const OP: &str = "destroy-program";
        self.program(OP, program)?;
        let entry = self
            .programs
            .remove(&program.slot)
            .ok_or_else(|| Self::validation(OP, "program disappeared"))?;
        self.raw.delete_program(Some(&entry.raw));
        self.driver_error(OP)
    }
}

impl WebGl2BrowserDiscovery {
    /// Compiles one already-lowered source, keeping the complete driver log.
    fn compile_shader_source(
        &self,
        op: &'static str,
        source: &GlShaderSource,
    ) -> Result<WebGlShader, GlError> {
        let kind = match source.stage {
            GlShaderStage::Vertex => Gl::VERTEX_SHADER,
            GlShaderStage::Fragment => Gl::FRAGMENT_SHADER,
            GlShaderStage::Compute => {
                return Err(GlError::Unsupported {
                    operation: op,
                    reason: "ESSL 300 has no compute stage on WebGL2",
                });
            }
        };
        let Some(raw) = self.raw.create_shader(kind) else {
            return Err(GlError::OutOfMemory { operation: op });
        };
        self.raw.shader_source(&raw, &source.text);
        self.raw.compile_shader(&raw);
        let compiled = self
            .shader_bool(op, &raw, Gl::COMPILE_STATUS)
            .and_then(|compiled| {
                if compiled {
                    Ok(())
                } else {
                    Err(GlError::Shader {
                        stage: shader_stage_name(source.stage),
                        log: self.raw.get_shader_info_log(&raw).unwrap_or_default(),
                    })
                }
            })
            .and_then(|()| self.driver_error(op));
        if let Err(error) = compiled {
            self.raw.delete_shader(Some(&raw));
            return Err(error);
        }
        Ok(raw)
    }

    fn shader_bool(
        &self,
        op: &'static str,
        shader: &WebGlShader,
        pname: u32,
    ) -> Result<bool, GlError> {
        self.raw
            .get_shader_parameter(shader, pname)
            .as_bool()
            .ok_or_else(|| GlError::Driver {
                operation: op,
                message: "shader parameter was not a boolean".into(),
            })
    }

    fn link_succeeded(&self, op: &'static str, program: &WebGlProgram) -> Result<(), GlError> {
        let linked = self
            .raw
            .get_program_parameter(program, Gl::LINK_STATUS)
            .as_bool()
            .ok_or_else(|| GlError::Driver {
                operation: op,
                message: "link status was not a boolean".into(),
            })?;
        if linked {
            Ok(())
        } else {
            Err(GlError::Program {
                operation: op,
                log: self.raw.get_program_info_log(program).unwrap_or_default(),
            })
        }
    }

    fn count_parameter(
        &self,
        op: &'static str,
        program: &WebGlProgram,
        pname: u32,
    ) -> Result<u32, GlError> {
        self.raw
            .get_program_parameter(program, pname)
            .as_f64()
            .filter(|value| value.is_finite() && *value >= 0.0 && value.fract() == 0.0)
            .map(|value| value as u32)
            .ok_or_else(|| GlError::Driver {
                operation: op,
                message: "resource count parameter was not a number".into(),
            })
    }

    /// Reflects the linked program against the frozen logical layout and
    /// applies the deterministic sampler-unit assignment.
    ///
    /// ESSL 3.00 fragment-output locations have no WebGL2 query entry point,
    /// so fragment-output facts stay empty rather than being invented; draw
    /// buffers select outputs at pass time.
    fn reflect_and_assign(
        &self,
        op: &'static str,
        program: &WebGlProgram,
        descriptor: &GlProgramDescriptor,
    ) -> Result<GlProgramReflection, GlError> {
        let attribute_count = self.count_parameter(op, program, Gl::ACTIVE_ATTRIBUTES)?;
        let mut vertex_inputs = Vec::new();
        for index in 0..attribute_count {
            let Some(info) = self.raw.get_active_attrib(program, index) else {
                return Err(GlError::Driver {
                    operation: op,
                    message: "active attribute was not reported".into(),
                });
            };
            let columns = vertex_input_columns(info.type_()).ok_or(GlError::Unsupported {
                operation: op,
                reason: "vertex input type is outside the common attribute vocabulary",
            })?;
            let location = self.raw.get_attrib_location(program, &info.name());
            let location = u32::try_from(location).map_err(|_| GlError::Driver {
                operation: op,
                message: "active attribute has no assigned location".into(),
            })?;
            vertex_inputs.push(super::super::GlVertexInputReflection {
                name: info.name(),
                location,
                columns,
            });
        }

        let block_count = self.count_parameter(op, program, Gl::ACTIVE_UNIFORM_BLOCKS)?;
        let mut blocks: Vec<String> = Vec::new();
        for index in 0..block_count {
            let Some(name) = self.raw.get_active_uniform_block_name(program, index) else {
                return Err(GlError::Driver {
                    operation: op,
                    message: "active uniform block was not reported".into(),
                });
            };
            blocks.push(name);
        }
        let uniform_count = self.count_parameter(op, program, Gl::ACTIVE_UNIFORMS)?;
        let mut samplers: Vec<(String, u32)> = Vec::new();
        for index in 0..uniform_count {
            let Some(info) = self.raw.get_active_uniform(program, index) else {
                return Err(GlError::Driver {
                    operation: op,
                    message: "active uniform was not reported".into(),
                });
            };
            if !is_sampler_type(info.type_()) {
                // Plain uniforms are outside the binding-layout vocabulary.
                continue;
            }
            let size = u32::try_from(info.size())
                .ok()
                .filter(|size| *size > 0)
                .unwrap_or(1);
            samplers.push((base_uniform_name(&info.name()).to_owned(), size));
        }

        // Deterministic assignment: uniform blocks take their GL block index;
        // sampler groups receive consecutive units in layout order.
        let mut assignments = Vec::new();
        let mut next_unit = 0u32;
        let max_units = self.discovery().limits().max_combined_texture_image_units;
        for binding in &descriptor.layout.bindings {
            match binding.kind {
                GlShaderResourceKind::UniformBuffer => {
                    let index = self.raw.get_uniform_block_index(program, &binding.name);
                    if index == Gl::INVALID_INDEX {
                        return Err(Self::validation(
                            op,
                            "declared uniform block is missing from the linked program",
                        ));
                    }
                    assignments.push(GlExecutableBindingAssignment {
                        logical: binding.location,
                        executable: GlExecutableBindingLocation::UniformBlock(index),
                    });
                    blocks.retain(|name| name != &binding.name);
                }
                GlShaderResourceKind::Sampler
                | GlShaderResourceKind::Texture
                | GlShaderResourceKind::CombinedTextureSampler => {
                    let declared_size = binding.array_count.max(1);
                    let found = samplers
                        .iter()
                        .find(|(name, _)| name == &binding.name)
                        .map(|&(_, size)| size);
                    let Some(active_size) = found else {
                        return Err(Self::validation(
                            op,
                            "declared sampler binding is missing from the linked program",
                        ));
                    };
                    if active_size != declared_size {
                        return Err(Self::validation(
                            op,
                            "sampler array size differs from the declared binding",
                        ));
                    }
                    let unit = next_unit;
                    next_unit = unit
                        .checked_add(active_size)
                        .ok_or(GlError::OutOfMemory { operation: op })?;
                    if next_unit > max_units {
                        return Err(Self::validation(
                            op,
                            "sampler assignments exceed the discovered unit count",
                        ));
                    }
                    for element in 0..active_size {
                        let element_name = if active_size == 1 {
                            binding.name.clone()
                        } else {
                            format!("{}[{element}]", binding.name)
                        };
                        let Some(location) = self.raw.get_uniform_location(program, &element_name)
                        else {
                            return Err(GlError::Driver {
                                operation: op,
                                message: "sampler uniform has no location".into(),
                            });
                        };
                        self.raw.uniform1i(Some(&location), (unit + element) as i32);
                    }
                    assignments.push(GlExecutableBindingAssignment {
                        logical: binding.location,
                        executable: GlExecutableBindingLocation::TextureUnit(unit),
                    });
                    samplers.retain(|(name, _)| name != &binding.name);
                }
                // The one refusal in this loop, and it is not a stub.  A WebGL2
                // program is never a compute program, so a layout naming a
                // storage binding describes a shader this profile's language
                // cannot express -- and one the linkage therefore cannot
                // produce, whatever the descriptor claims.  Refusing it here
                // rather than resolving it is what keeps the descriptor from
                // being the authority on what linked.
                GlShaderResourceKind::StorageBuffer | GlShaderResourceKind::StorageImage => {
                    return Err(GlError::Unsupported {
                        operation: op,
                        reason: "a WebGL2 program declares no storage binding, so a layout naming one cannot be reflected",
                    });
                }
            }
        }
        if !blocks.is_empty() {
            return Err(Self::validation(
                op,
                "linked program declares a uniform block absent from the layout",
            ));
        }
        if !samplers.is_empty() {
            return Err(Self::validation(
                op,
                "linked program declares a sampler absent from the layout",
            ));
        }
        let reflection = GlProgramReflection {
            vertex_inputs,
            fragment_outputs: Vec::new(),
            assignments,
        };
        reflection
            .validate_against(&descriptor.layout)
            .map_err(|_| {
                Self::validation(op, "linked reflection does not satisfy the frozen layout")
            })?;
        Ok(reflection)
    }
}

const fn shader_stage_name(stage: GlShaderStage) -> &'static str {
    match stage {
        GlShaderStage::Vertex => "vertex",
        GlShaderStage::Fragment => "fragment",
        GlShaderStage::Compute => "compute",
    }
}

impl GlBindingApi for WebGl2BrowserDiscovery {
    fn active_texture(&mut self, unit: u32) -> Result<(), GlError> {
        const OP: &str = "active-texture";
        self.assert_provider_ready(OP)?;
        validate_texture_unit(unit, self.binding_limits())
            .map_err(|error| Self::validation(OP, error.message()))?;
        self.raw.active_texture(Gl::TEXTURE0 + unit);
        self.driver_error(OP)
    }

    fn bind_texture(
        &mut self,
        unit: u32,
        target: GlTextureTarget,
        texture: Option<TextureId>,
    ) -> Result<(), GlError> {
        const OP: &str = "bind-texture";
        self.assert_provider_ready(OP)?;
        validate_texture_unit(unit, self.binding_limits())
            .map_err(|error| Self::validation(OP, error.message()))?;
        let raw = texture
            .map(|texture| {
                let entry = self.texture(OP, texture)?;
                Ok::<_, GlError>(entry.raw.clone())
            })
            .transpose()?;
        self.raw.active_texture(Gl::TEXTURE0 + unit);
        self.raw.bind_texture(texture_target(target), raw.as_ref());
        self.driver_error(OP)
    }

    fn bind_sampler(&mut self, unit: u32, sampler: Option<SamplerId>) -> Result<(), GlError> {
        const OP: &str = "bind-sampler";
        self.assert_provider_ready(OP)?;
        validate_texture_unit(unit, self.binding_limits())
            .map_err(|error| Self::validation(OP, error.message()))?;
        let raw = sampler
            .map(|sampler| {
                self.validate_object_context(OP, sampler.context)?;
                let entry = self
                    .samplers
                    .get(&sampler.slot)
                    .ok_or_else(|| Self::validation(OP, "sampler allocation is not live"))?;
                if entry.generation != sampler.generation {
                    return Err(Self::validation(
                        OP,
                        "sampler allocation generation is stale",
                    ));
                }
                Ok(entry.raw.clone())
            })
            .transpose()?;
        self.raw.bind_sampler(unit, raw.as_ref());
        self.driver_error(OP)
    }

    fn bind_uniform_buffer(
        &mut self,
        index: u32,
        buffer: Option<BufferId>,
        offset: u32,
        size: u32,
    ) -> Result<(), GlError> {
        const OP: &str = "bind-uniform-buffer";
        self.assert_provider_ready(OP)?;
        let limits = self.binding_limits();
        validate_uniform_buffer_binding(index, buffer, offset, size, limits)
            .map_err(|error| Self::validation(OP, error.message()))?;
        let Some(buffer) = buffer else {
            self.raw.bind_buffer_base(Gl::UNIFORM_BUFFER, index, None);
            return self.driver_error(OP);
        };
        let desc = {
            let entry = self.buffer(OP, buffer)?;
            if !entry.desc.usage.contains(GlBufferUsage::UNIFORM) {
                return Err(Self::validation(OP, "buffer lacks uniform usage"));
            }
            entry.desc
        };
        validate_uniform_range(offset, size, desc.size)
            .map_err(|error| Self::validation(OP, error.message()))?;
        let resolved = if size == 0 {
            desc.size - u64::from(offset)
        } else {
            u64::from(size)
        };
        let raw = self
            .buffers
            .get(&buffer.slot)
            .map(|entry| entry.raw.clone())
            .ok_or_else(|| Self::validation(OP, "buffer allocation disappeared"))?;
        // WebGL2 binds indexed uniform ranges through bindBufferBase/Range;
        // offsets and sizes travel as exact f64 integers.
        self.raw.bind_buffer_range_with_f64_and_f64(
            Gl::UNIFORM_BUFFER,
            index,
            Some(&raw),
            u64::from(offset) as f64,
            resolved as f64,
        );
        self.driver_error(OP)
    }
}

impl WebGl2BrowserDiscovery {
    /// The discovered limits one binding validation needs.
    fn binding_limits(&self) -> GlBindingLimits {
        let limits = self.discovery().limits();
        GlBindingLimits {
            max_texture_units: limits.max_combined_texture_image_units,
            max_uniform_buffer_bindings: limits.max_uniform_buffer_bindings,
            uniform_buffer_offset_alignment: limits.uniform_buffer_offset_alignment,
        }
    }
}

const fn texture_target(target: GlTextureTarget) -> u32 {
    match target {
        GlTextureTarget::D2 => Gl::TEXTURE_2D,
        GlTextureTarget::Cube => Gl::TEXTURE_CUBE_MAP,
        GlTextureTarget::D3 => Gl::TEXTURE_3D,
        GlTextureTarget::D2Array => Gl::TEXTURE_2D_ARRAY,
    }
}
