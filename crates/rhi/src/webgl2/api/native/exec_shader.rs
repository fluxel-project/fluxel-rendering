//! Native shader, program, reflection, and binding execution.
//!
//! Sources arrive already lowered and profile-validated: ESSL 300/310/320 for
//! the embedded family, the exact `4xx` desktop GLSL for the desktop family.
//! This module compiles them, keeps every driver log inside a structured
//! error, reflects the actually linked resources, and rejects any layout or
//! program mismatch before a program object is published.

use super::super::{
    BufferId, GlBindingApi, GlBindingLimits, GlBufferUsage, GlError, GlExecutableBindingAssignment,
    GlExecutableBindingLocation, GlFamilyApi as _, GlProgramDescriptor, GlProgramKind,
    GlProgramReflection, GlShaderApi, GlShaderResourceKind, GlShaderSource, GlShaderStage,
    GlTextureTarget, ProgramId, SamplerId, ShaderId, TextureId, validate_texture_unit,
    validate_uniform_buffer_binding, validate_uniform_range,
};
use super::provider::{NativeGlProvider, NativeProgram};

/// The sampler uniform types this backend can bind and reflect. The closed
/// list matches the shared contract; integer and cube-shadow sampler types
/// stay outside the common binding vocabulary and are skipped like plain
/// uniforms, exactly as the browser backend records them.
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

/// Strips the `[0]` suffix GL reports for array uniforms.
fn base_uniform_name(name: &str) -> &str {
    name.strip_suffix("[0]").unwrap_or(name)
}

impl GlShaderApi for NativeGlProvider<'_> {
    fn create_shader(&mut self, source: &GlShaderSource) -> Result<ShaderId, GlError> {
        const OP: &str = "create-shader";
        self.assert_ready(OP)?;
        source
            .validate_for(self.profile())
            .map_err(|_| Self::validation(OP, "shader source does not target this profile"))?;
        if source.stage == GlShaderStage::Compute
            && !self
                .discovery
                .capabilities()
                .supports(super::super::GlCapability::Compute)
        {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "this context did not prove the compute capability",
            });
        }
        let raw = self.compile_shader_source(OP, source)?;
        let slot = self.slot(OP)?;
        let id = ShaderId::new(self.context_stamp(), slot, 0);
        self.shaders.insert(id, raw);
        Ok(id)
    }

    fn destroy_shader(&mut self, shader: ShaderId) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "destroy-shader";
        self.assert_ready(OP)?;
        let raw = self.shader(OP, shader)?;
        // SAFETY: current-context contract; liveness was checked first.
        unsafe { self.gl.delete_shader(raw) };
        self.driver_error(OP)?;
        self.shaders.remove(&shader);
        Ok(())
    }

    fn create_program(
        &mut self,
        descriptor: &GlProgramDescriptor,
    ) -> Result<(ProgramId, GlProgramReflection), GlError> {
        const OP: &str = "create-program";
        self.assert_ready(OP)?;
        descriptor
            .validate_for(self.profile())
            .map_err(|_| Self::validation(OP, "invalid program descriptor"))?;
        // Compute programs route on the proved capability; without the probe
        // evidence the kind is rejected structurally before any driver call.
        if matches!(descriptor.kind, GlProgramKind::Compute { .. })
            && !self
                .discovery
                .capabilities()
                .supports(super::super::GlCapability::Compute)
        {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "this context did not prove the compute capability",
            });
        }
        let stages: Vec<&GlShaderSource> = match &descriptor.kind {
            GlProgramKind::Raster { vertex, fragment } => vec![vertex, fragment],
            GlProgramKind::Compute { shader } => vec![shader],
        };
        let mut compiled = Vec::new();
        for source in stages {
            match self.compile_shader_source(OP, source) {
                Ok(raw) => compiled.push(raw),
                Err(error) => {
                    self.delete_shaders(&compiled);
                    return Err(error);
                }
            }
        }
        let linked = self.link_and_reflect(OP, descriptor, &compiled);
        // Shaders may be deleted once attached; the program owns its binaries.
        self.delete_shaders(&compiled);
        let (raw, reflection) = linked?;
        let slot = self.slot(OP)?;
        let id = ProgramId::new(self.context_stamp(), slot, 0);
        self.programs.insert(
            id,
            NativeProgram {
                generation: id.generation,
                raw,
                descriptor: descriptor.clone(),
            },
        );
        Ok((id, reflection))
    }

    fn destroy_program(&mut self, program: ProgramId) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "destroy-program";
        self.assert_ready(OP)?;
        self.program(OP, program)?;
        let entry = self
            .programs
            .remove(&program)
            .ok_or_else(|| Self::validation(OP, "program disappeared"))?;
        // SAFETY: current-context contract; liveness was checked first.
        unsafe { self.gl.delete_program(entry.raw) };
        self.driver_error(OP)
    }
}

impl NativeGlProvider<'_> {
    fn delete_shaders(&self, shaders: &[glow::NativeShader]) {
        use glow::HasContext as _;
        for shader in shaders {
            // SAFETY: current-context contract; these handles were created on
            // it during this transaction.
            unsafe { self.gl.delete_shader(*shader) };
        }
    }

    /// Compiles one already-lowered source, keeping the complete driver log.
    fn compile_shader_source(
        &self,
        op: &'static str,
        source: &GlShaderSource,
    ) -> Result<glow::NativeShader, GlError> {
        use glow::HasContext as _;
        let kind = match source.stage {
            GlShaderStage::Vertex => glow::VERTEX_SHADER,
            GlShaderStage::Fragment => glow::FRAGMENT_SHADER,
            GlShaderStage::Compute => glow::COMPUTE_SHADER,
        };
        // SAFETY: current-context contract; a failed compile leaves no object.
        let raw = unsafe { self.gl.create_shader(kind) }.map_err(|message| GlError::Driver {
            operation: op,
            message,
        })?;
        // SAFETY: see above.
        let compiled = unsafe {
            self.gl.shader_source(raw, &source.text);
            self.gl.compile_shader(raw);
            let compiled = self.gl.get_shader_compile_status(raw);
            if compiled {
                Ok(())
            } else {
                Err(GlError::Shader {
                    stage: shader_stage_name(source.stage),
                    log: self.gl.get_shader_info_log(raw),
                })
            }
        }
        .and_then(|()| self.driver_error(op));
        if let Err(error) = compiled {
            // SAFETY: current-context contract.
            unsafe { self.gl.delete_shader(raw) };
            return Err(error);
        }
        Ok(raw)
    }

    /// Attaches, links, reflects, and validates one program inside an
    /// explicit `useProgram` scope that leaves no program selected.
    fn link_and_reflect(
        &mut self,
        op: &'static str,
        descriptor: &GlProgramDescriptor,
        shaders: &[glow::NativeShader],
    ) -> Result<(glow::NativeProgram, GlProgramReflection), GlError> {
        use glow::HasContext as _;
        // SAFETY: current-context contract; the program is either returned
        // into the caller's table or deleted on every error path below.
        unsafe {
            let program = self
                .gl
                .create_program()
                .map_err(|message| GlError::Driver {
                    operation: op,
                    message,
                })?;
            let linked = 'link: {
                for shader in shaders {
                    self.gl.attach_shader(program, *shader);
                }
                self.gl.link_program(program);
                if !self.gl.get_program_link_status(program) {
                    break 'link Err(GlError::Program {
                        operation: op,
                        log: self.gl.get_program_info_log(program),
                    });
                }
                break 'link self.driver_error(op);
            };
            if let Err(error) = linked {
                self.gl.delete_program(program);
                return Err(error);
            }
            // Sampler-unit assignment needs the program current: `uniform1i`
            // targets the bound program, so reflection runs inside an explicit
            // bind scope and leaves no program selected afterwards.  The
            // provider's record of the driver's current program has to follow it
            // out of the scope -- a pipeline installed before this link is no
            // longer the driver's current program, and a record that still named
            // it would skip the selection a draw needs.
            self.gl.use_program(Some(program));
            let reflection = self.reflect_and_assign(op, program, descriptor);
            self.gl.use_program(None);
            self.current_program = None;
            match reflection {
                Ok(reflection) => match self.driver_error(op) {
                    Ok(()) => Ok((program, reflection)),
                    Err(error) => {
                        self.gl.delete_program(program);
                        Err(error)
                    }
                },
                Err(error) => {
                    self.gl.delete_program(program);
                    Err(error)
                }
            }
        }
    }

    /// Reflects the linked program against the frozen logical layout and
    /// applies the deterministic sampler-unit assignment.
    ///
    /// Fragment-output facts stay empty across the shared contract; draw
    /// buffers select outputs at pass time.
    ///
    /// # Safety
    ///
    /// Current-context contract; `program` must be the current program.
    unsafe fn reflect_and_assign(
        &self,
        op: &'static str,
        program: glow::NativeProgram,
        descriptor: &GlProgramDescriptor,
    ) -> Result<GlProgramReflection, GlError> {
        use glow::HasContext as _;
        // SAFETY: current-context contract; `program` is the current program
        // and every statement below is a reflection query or sampler-unit
        // assignment against it.
        unsafe {
            let attribute_count = self.gl.get_active_attributes(program);
            let mut vertex_inputs = Vec::new();
            for index in 0..attribute_count {
                let Some(info) = self.gl.get_active_attribute(program, index) else {
                    return Err(GlError::Driver {
                        operation: op,
                        message: "active attribute was not reported".into(),
                    });
                };
                let columns = vertex_input_columns(info.atype).ok_or(GlError::Unsupported {
                    operation: op,
                    reason: "vertex input type is outside the common attribute vocabulary",
                })?;
                let location =
                    self.gl
                        .get_attrib_location(program, &info.name)
                        .ok_or(GlError::Driver {
                            operation: op,
                            message: "active attribute has no assigned location".into(),
                        })?;
                vertex_inputs.push(super::super::GlVertexInputReflection {
                    name: info.name,
                    location,
                    columns,
                });
            }

            let block_count = u32::try_from(
                self.gl
                    .get_program_parameter_i32(program, glow::ACTIVE_UNIFORM_BLOCKS),
            )
            .map_err(|_| GlError::Driver {
                operation: op,
                message: "uniform block count was negative".into(),
            })?;
            let mut blocks: Vec<(u32, String)> = Vec::new();
            for index in 0..block_count {
                blocks.push((index, self.gl.get_active_uniform_block_name(program, index)));
            }
            let uniform_count = self.gl.get_active_uniforms(program);
            let mut samplers: Vec<(String, u32)> = Vec::new();
            for index in 0..uniform_count {
                let Some(info) = self.gl.get_active_uniform(program, index) else {
                    return Err(GlError::Driver {
                        operation: op,
                        message: "active uniform was not reported".into(),
                    });
                };
                if !is_sampler_type(info.utype) {
                    // Plain uniforms are outside the binding-layout vocabulary.
                    continue;
                }
                let size = u32::try_from(info.size)
                    .ok()
                    .filter(|size| *size > 0)
                    .unwrap_or(1);
                samplers.push((base_uniform_name(&info.name).to_owned(), size));
            }

            // Deterministic assignment: uniform blocks take their GL block index;
            // sampler groups receive consecutive units in layout order.
            let mut assignments = Vec::new();
            let mut next_unit = 0u32;
            let max_units = self.discovery.limits().max_combined_texture_image_units;
            // Image units are their own namespace, so this counter starts at
            // zero independently of the sampler one above.
            let mut next_image_unit = 0u32;
            let max_image_units = self.discovery.limits().max_image_units;
            for binding in &descriptor.layout.bindings {
                match binding.kind {
                    GlShaderResourceKind::UniformBuffer => {
                        let Some(position) =
                            blocks.iter().position(|(_, name)| *name == binding.name)
                        else {
                            return Err(Self::validation(
                                op,
                                "declared uniform block is missing from the linked program",
                            ));
                        };
                        let (index, _) = blocks.remove(position);
                        assignments.push(GlExecutableBindingAssignment {
                            logical: binding.location,
                            executable: GlExecutableBindingLocation::UniformBlock(index),
                        });
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
                            let Some(location) =
                                self.gl.get_uniform_location(program, &element_name)
                            else {
                                return Err(GlError::Driver {
                                    operation: op,
                                    message: "sampler uniform has no location".into(),
                                });
                            };
                            self.gl
                                .uniform_1_i32(Some(&location), (unit + element) as i32);
                        }
                        assignments.push(GlExecutableBindingAssignment {
                            logical: binding.location,
                            executable: GlExecutableBindingLocation::TextureUnit(unit),
                        });
                        let Some(position) =
                            samplers.iter().position(|(name, _)| *name == binding.name)
                        else {
                            return Err(Self::validation(
                                op,
                                "sampler disappeared during reflection",
                            ));
                        };
                        samplers.remove(position);
                    }
                    // Both storage kinds refuse an array, and the reason is the
                    // same for both: no recipe lowers one, so the array path
                    // would be code that no context has ever run.  The uniform
                    // block and sampler arms above support arrays because a
                    // raster recipe samples one; this arm will when a compute
                    // recipe declares one, and not before.
                    GlShaderResourceKind::StorageBuffer => {
                        if binding.array_count != 1 {
                            return Err(Self::validation(
                                op,
                                "a storage block array is not lowered by any recipe",
                            ));
                        }
                        // Resolved by name rather than by walking the program,
                        // which is where this arm and the uniform block arm
                        // above part company.  The linked program's storage
                        // blocks cannot be enumerated through this GL
                        // abstraction -- there is no counterpart to the active
                        // uniform block count -- and none is needed: the name is
                        // the lowering's own, so the question is only whether
                        // the program has a block by that name.  The abstraction
                        // answers `None` for the driver's invalid index, so that
                        // check is the `let else` rather than a comparison.
                        //
                        // What that costs is stated rather than implied: the
                        // "linked program declares a block absent from the
                        // layout" check below has no storage counterpart, so a
                        // compute shader declaring a storage block the layout
                        // does not mention is not caught here.
                        let Some(index) = self
                            .gl
                            .get_shader_storage_block_index(program, &binding.name)
                        else {
                            return Err(Self::validation(
                                op,
                                "declared storage block is missing from the linked program",
                            ));
                        };
                        assignments.push(GlExecutableBindingAssignment {
                            logical: binding.location,
                            executable: GlExecutableBindingLocation::StorageBlock(index),
                        });
                    }
                    GlShaderResourceKind::StorageImage => {
                        if binding.array_count != 1 {
                            return Err(Self::validation(
                                op,
                                "a storage image array is not lowered by any recipe",
                            ));
                        }
                        // The image's unit is the value of its own uniform,
                        // exactly as a sampler's is, so it is assigned and
                        // written here rather than left to the binding path --
                        // and that is also what keeps the shader and the
                        // adapter agreeing on a unit no declaration fixes.
                        let unit = next_image_unit;
                        next_image_unit = unit
                            .checked_add(1)
                            .ok_or(GlError::OutOfMemory { operation: op })?;
                        if next_image_unit > max_image_units {
                            return Err(Self::validation(
                                op,
                                "storage image assignments exceed the discovered image unit count",
                            ));
                        }
                        let Some(location) = self.gl.get_uniform_location(program, &binding.name)
                        else {
                            return Err(GlError::Driver {
                                operation: op,
                                message: "storage image uniform has no location".into(),
                            });
                        };
                        self.gl.uniform_1_i32(Some(&location), unit as i32);
                        assignments.push(GlExecutableBindingAssignment {
                            logical: binding.location,
                            executable: GlExecutableBindingLocation::ImageUnit(unit),
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
}

const fn shader_stage_name(stage: GlShaderStage) -> &'static str {
    match stage {
        GlShaderStage::Vertex => "vertex",
        GlShaderStage::Fragment => "fragment",
        GlShaderStage::Compute => "compute",
    }
}

impl GlBindingApi for NativeGlProvider<'_> {
    fn active_texture(&mut self, unit: u32) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "active-texture";
        self.assert_ready(OP)?;
        validate_texture_unit(unit, self.binding_limits())
            .map_err(|error| Self::validation(OP, error.message()))?;
        // SAFETY: current-context contract; the unit was bounds-checked.
        unsafe { self.gl.active_texture(glow::TEXTURE0 + unit) };
        self.driver_error(OP)
    }

    fn bind_texture(
        &mut self,
        unit: u32,
        target: GlTextureTarget,
        texture: Option<TextureId>,
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "bind-texture";
        self.assert_ready(OP)?;
        validate_texture_unit(unit, self.binding_limits())
            .map_err(|error| Self::validation(OP, error.message()))?;
        let raw = texture
            .map(|texture| {
                let entry = self.texture(OP, texture)?;
                Ok::<_, GlError>(entry.0)
            })
            .transpose()?;
        // SAFETY: current-context contract; liveness and unit bounds checked.
        unsafe {
            self.gl.active_texture(glow::TEXTURE0 + unit);
            self.gl.bind_texture(texture_target(target), raw);
        }
        self.driver_error(OP)
    }

    fn bind_sampler(&mut self, unit: u32, sampler: Option<SamplerId>) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "bind-sampler";
        self.assert_ready(OP)?;
        validate_texture_unit(unit, self.binding_limits())
            .map_err(|error| Self::validation(OP, error.message()))?;
        let raw = sampler
            .map(|sampler| self.sampler(OP, sampler))
            .transpose()?;
        // SAFETY: current-context contract; liveness and unit bounds checked.
        unsafe { self.gl.bind_sampler(unit, raw) };
        self.driver_error(OP)
    }

    fn bind_uniform_buffer(
        &mut self,
        index: u32,
        buffer: Option<BufferId>,
        offset: u32,
        size: u32,
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "bind-uniform-buffer";
        self.assert_ready(OP)?;
        let limits = self.binding_limits();
        validate_uniform_buffer_binding(index, buffer, offset, size, limits)
            .map_err(|error| Self::validation(OP, error.message()))?;
        let Some(buffer) = buffer else {
            // SAFETY: current-context contract; the unbind shape was checked.
            unsafe {
                self.gl.bind_buffer_base(glow::UNIFORM_BUFFER, index, None);
            }
            return self.driver_error(OP);
        };
        let (name, desc) = {
            let entry = self.buffer(OP, buffer)?;
            if !entry.1.usage.contains(GlBufferUsage::UNIFORM) {
                return Err(Self::validation(OP, "buffer lacks uniform usage"));
            }
            entry
        };
        validate_uniform_range(offset, size, desc.size)
            .map_err(|error| Self::validation(OP, error.message()))?;
        let resolved = if size == 0 {
            desc.size - u64::from(offset)
        } else {
            u64::from(size)
        };
        let (offset, resolved) = {
            let offset = i32::try_from(offset)
                .map_err(|_| Self::validation(OP, "offset exceeds GLintptr"))?;
            let resolved = i32::try_from(resolved)
                .map_err(|_| Self::validation(OP, "range exceeds GLintptr"))?;
            (offset, resolved)
        };
        // SAFETY: current-context contract; usage, alignment, and range were
        // validated against the live allocation.
        unsafe {
            self.gl
                .bind_buffer_range(glow::UNIFORM_BUFFER, index, Some(name), offset, resolved);
        }
        self.driver_error(OP)
    }
}

impl NativeGlProvider<'_> {
    /// The discovered limits one binding validation needs.
    pub(super) fn binding_limits(&self) -> GlBindingLimits {
        let limits = self.discovery.limits();
        GlBindingLimits {
            max_texture_units: limits.max_combined_texture_image_units,
            max_uniform_buffer_bindings: limits.max_uniform_buffer_bindings,
            uniform_buffer_offset_alignment: limits.uniform_buffer_offset_alignment,
        }
    }
}

const fn texture_target(target: GlTextureTarget) -> u32 {
    match target {
        GlTextureTarget::D2 => glow::TEXTURE_2D,
        GlTextureTarget::Cube => glow::TEXTURE_CUBE_MAP,
        GlTextureTarget::D3 => glow::TEXTURE_3D,
        GlTextureTarget::D2Array => glow::TEXTURE_2D_ARRAY,
    }
}
