//! Native vertex-array execution.
//!
//! Like the browser backend, a VAO cannot pre-store a layout without buffers,
//! so `create_vertex_array` records the structural layout and
//! `bind_vertex_array` re-emits attribute pointers against the supplied
//! bindings; redundant-call elimination belongs to the Layer 2 cache, never
//! to this contract.

use super::provider::{NativeGlProvider, NativeVertexArray};
use crate::backend::gl::api::{
    GlError, GlFamilyApi as _, GlIndexBinding, GlIndexFormat, GlVertexApi, GlVertexBufferBinding,
    GlVertexBufferMetadata, GlVertexFormat, GlVertexLayout, GlVertexStepMode, VertexArrayId,
};

/// How one attribute reaches its shader input.
enum AttributePath {
    /// `vertexAttribPointer` with these `(size, type, normalized)` values.
    Float(i32, u32, bool),
    /// `vertexAttribIPointer` with these `(size, type)` values.
    Integer(i32, u32),
}

const fn attribute_path(format: GlVertexFormat) -> AttributePath {
    use GlVertexFormat as F;
    match format {
        F::Uint8x2 => AttributePath::Float(2, glow::UNSIGNED_BYTE, false),
        F::Uint8x4 => AttributePath::Float(4, glow::UNSIGNED_BYTE, false),
        F::Sint8x2 => AttributePath::Float(2, glow::BYTE, false),
        F::Sint8x4 => AttributePath::Float(4, glow::BYTE, false),
        F::Unorm8x2 => AttributePath::Float(2, glow::UNSIGNED_BYTE, true),
        F::Unorm8x4 => AttributePath::Float(4, glow::UNSIGNED_BYTE, true),
        F::Snorm8x2 => AttributePath::Float(2, glow::BYTE, true),
        F::Snorm8x4 => AttributePath::Float(4, glow::BYTE, true),
        F::Uint16x2 => AttributePath::Float(2, glow::UNSIGNED_SHORT, false),
        F::Uint16x4 => AttributePath::Float(4, glow::UNSIGNED_SHORT, false),
        F::Sint16x2 => AttributePath::Float(2, glow::SHORT, false),
        F::Sint16x4 => AttributePath::Float(4, glow::SHORT, false),
        F::Unorm16x2 => AttributePath::Float(2, glow::UNSIGNED_SHORT, true),
        F::Unorm16x4 => AttributePath::Float(4, glow::UNSIGNED_SHORT, true),
        F::Snorm16x2 => AttributePath::Float(2, glow::SHORT, true),
        F::Snorm16x4 => AttributePath::Float(4, glow::SHORT, true),
        F::Float16x2 => AttributePath::Float(2, glow::HALF_FLOAT, false),
        F::Float16x4 => AttributePath::Float(4, glow::HALF_FLOAT, false),
        F::Float32 => AttributePath::Float(1, glow::FLOAT, false),
        F::Float32x2 => AttributePath::Float(2, glow::FLOAT, false),
        F::Float32x3 => AttributePath::Float(3, glow::FLOAT, false),
        F::Float32x4 => AttributePath::Float(4, glow::FLOAT, false),
        F::Uint32 => AttributePath::Integer(1, glow::UNSIGNED_INT),
        F::Uint32x2 => AttributePath::Integer(2, glow::UNSIGNED_INT),
        F::Uint32x3 => AttributePath::Integer(3, glow::UNSIGNED_INT),
        F::Uint32x4 => AttributePath::Integer(4, glow::UNSIGNED_INT),
        F::Sint32 => AttributePath::Integer(1, glow::INT),
        F::Sint32x2 => AttributePath::Integer(2, glow::INT),
        F::Sint32x3 => AttributePath::Integer(3, glow::INT),
        F::Sint32x4 => AttributePath::Integer(4, glow::INT),
    }
}

const fn index_type(format: GlIndexFormat) -> u32 {
    match format {
        GlIndexFormat::Uint16 => glow::UNSIGNED_SHORT,
        GlIndexFormat::Uint32 => glow::UNSIGNED_INT,
    }
}

const fn index_size(format: GlIndexFormat) -> u64 {
    match format {
        GlIndexFormat::Uint16 => 2,
        GlIndexFormat::Uint32 => 4,
    }
}

impl GlVertexApi for NativeGlProvider {
    fn create_vertex_array(&mut self, layout: &GlVertexLayout) -> Result<VertexArrayId, GlError> {
        use glow::HasContext as _;
        const OP: &str = "create-vertex-array";
        self.assert_ready(OP)?;
        layout
            .validate()
            .map_err(|_| Self::validation(OP, "invalid vertex layout"))?;
        let max_attributes = self.discovery.limits().max_vertex_attributes;
        if layout
            .attributes
            .iter()
            .any(|attribute| attribute.location >= max_attributes)
        {
            return Err(Self::validation(
                OP,
                "attribute location exceeds the discovered limit",
            ));
        }
        // SAFETY: current-context contract.
        let raw = unsafe { self.gl.create_vertex_array() }.map_err(|message| GlError::Driver {
            operation: OP,
            message,
        })?;
        let slot = self.slot(OP)?;
        let id = VertexArrayId::new(self.context_stamp(), slot, 0);
        self.vertex_arrays.insert(
            id,
            NativeVertexArray {
                generation: id.generation,
                raw,
                layout: layout.clone(),
                index: None,
            },
        );
        Ok(id)
    }

    fn destroy_vertex_array(&mut self, vertex_array: VertexArrayId) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "destroy-vertex-array";
        self.assert_ready(OP)?;
        self.vertex_array(OP, vertex_array)?;
        let entry = self
            .vertex_arrays
            .remove(&vertex_array)
            .ok_or_else(|| Self::validation(OP, "vertex array disappeared"))?;
        // SAFETY: current-context contract; liveness was checked first.
        unsafe { self.gl.delete_vertex_array(entry.raw) };
        self.driver_error(OP)
    }

    fn bind_vertex_array(
        &mut self,
        vertex_array: VertexArrayId,
        buffers: &[GlVertexBufferBinding],
        index: Option<GlIndexBinding>,
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "bind-vertex-array";
        self.assert_ready(OP)?;
        self.vertex_array(OP, vertex_array)?;
        // Allocation facts come from this provider's own table, so callers
        // never forge byte lengths.
        let mut metadata: Vec<GlVertexBufferMetadata> = Vec::with_capacity(buffers.len() + 1);
        for binding in buffers {
            let entry = self.buffer(OP, binding.buffer)?;
            metadata.push(GlVertexBufferMetadata {
                buffer: binding.buffer,
                byte_length: entry.1.size,
                usage: entry.1.usage,
            });
        }
        if let Some(index) = index {
            let entry = self.buffer(OP, index.buffer)?;
            // The index buffer reaches validation through the same table as
            // the attribute buffers, so its size and roles are checked there
            // rather than re-derived here.
            metadata.push(GlVertexBufferMetadata {
                buffer: index.buffer,
                byte_length: entry.1.size,
                usage: entry.1.usage,
            });
        }
        let layout = {
            match self.vertex_arrays.get(&vertex_array) {
                Some(entry) if entry.generation == vertex_array.generation => entry.layout.clone(),
                _ => return Err(Self::validation(OP, "vertex array disappeared")),
            }
        };
        layout
            .validate_bindings(buffers, index, &metadata, self.context_stamp())
            .map_err(|_| Self::validation(OP, "invalid vertex bindings"))?;
        let raw = match self.vertex_arrays.get(&vertex_array) {
            Some(entry) => entry.raw,
            None => return Err(Self::validation(OP, "vertex array disappeared")),
        };
        // SAFETY: current-context contract; every binding was validated above
        // against the live allocation table before any GL state changed.
        unsafe {
            self.gl.bind_vertex_array(Some(raw));
            // Recorded the moment the driver takes the binding rather than after
            // the rest of this verb succeeds, because the record answers what the
            // driver holds and not what this call intended: an attribute emission
            // that fails below leaves the array bound all the same, and a draw
            // that trusted a stale record would re-bind an array the caller never
            // asked for.  This is one of the two writers of that field.
            self.bound_vertex_array = Some(vertex_array);
            self.emit_attributes(&layout, buffers)?;
            // ELEMENT_ARRAY_BUFFER state lives inside the VAO, so the index
            // binding is recorded for draw-time bounds checks and byte offsets.
            if let Some(index) = index {
                let raw = self
                    .buffers
                    .get(&index.buffer)
                    .map(|entry| entry.0)
                    .ok_or_else(|| Self::validation(OP, "index buffer disappeared"))?;
                self.gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(raw));
            } else {
                self.gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, None);
            }
        }
        let result = self.driver_error(OP);
        match result {
            Ok(()) => {
                if let Some(entry) = self.vertex_arrays.get_mut(&vertex_array) {
                    entry.index = index;
                }
                Ok(())
            }
            Err(error) => Err(error),
        }
    }
}

impl NativeGlProvider {
    /// Emits one `vertexAttribPointer` group per layout slot.
    ///
    /// # Safety
    ///
    /// Current-context contract; a VAO must be bound by the caller.
    unsafe fn emit_attributes(
        &self,
        layout: &GlVertexLayout,
        buffers: &[GlVertexBufferBinding],
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        unsafe {
            for slot_layout in &layout.buffers {
                let Some(binding) = buffers
                    .iter()
                    .find(|binding| binding.slot == slot_layout.slot)
                else {
                    return Err(Self::validation(
                        "bind-vertex-array",
                        "layout slot lost its binding",
                    ));
                };
                let raw = self
                    .buffers
                    .get(&binding.buffer)
                    .map(|entry| entry.0)
                    .ok_or_else(|| {
                        Self::validation("bind-vertex-array", "bound buffer disappeared")
                    })?;
                self.gl.bind_buffer(glow::ARRAY_BUFFER, Some(raw));
                for attribute in layout
                    .attributes
                    .iter()
                    .filter(|attribute| attribute.buffer_slot == slot_layout.slot)
                {
                    // Stride zero means tightly packed attribute values.
                    let stride = if slot_layout.stride == 0 {
                        attribute_byte_size(attribute.format)
                    } else {
                        slot_layout.stride
                    } as u64;
                    let offset = binding.offset.checked_add(u64::from(attribute.offset));
                    let offset = match offset.and_then(|value| i32::try_from(value).ok()) {
                        Some(offset) => offset,
                        None => {
                            return Err(Self::validation(
                                "bind-vertex-array",
                                "attribute offset exceeds the binding representation",
                            ));
                        }
                    };
                    match attribute_path(attribute.format) {
                        AttributePath::Float(size, type_, normalized) => {
                            self.gl.vertex_attrib_pointer_f32(
                                attribute.location,
                                size,
                                type_,
                                normalized,
                                stride as i32,
                                offset,
                            );
                        }
                        AttributePath::Integer(size, type_) => {
                            self.gl.vertex_attrib_pointer_i32(
                                attribute.location,
                                size,
                                type_,
                                stride as i32,
                                offset,
                            );
                        }
                    }
                    self.gl.enable_vertex_attrib_array(attribute.location);
                    self.gl.vertex_attrib_divisor(
                        attribute.location,
                        match slot_layout.step_mode {
                            GlVertexStepMode::Vertex => 0,
                            GlVertexStepMode::Instance => 1,
                        },
                    );
                }
            }
        }
        Ok(())
    }
}

const fn attribute_byte_size(format: GlVertexFormat) -> u32 {
    // The same closed table `GlVertexFormat::byte_size` uses, kept local so
    // this module does not depend on private trait items.
    match format {
        GlVertexFormat::Uint8x2
        | GlVertexFormat::Sint8x2
        | GlVertexFormat::Unorm8x2
        | GlVertexFormat::Snorm8x2 => 2,
        GlVertexFormat::Uint8x4
        | GlVertexFormat::Sint8x4
        | GlVertexFormat::Unorm8x4
        | GlVertexFormat::Snorm8x4 => 4,
        GlVertexFormat::Uint16x2
        | GlVertexFormat::Sint16x2
        | GlVertexFormat::Unorm16x2
        | GlVertexFormat::Snorm16x2
        | GlVertexFormat::Float16x2 => 4,
        GlVertexFormat::Uint16x4
        | GlVertexFormat::Sint16x4
        | GlVertexFormat::Unorm16x4
        | GlVertexFormat::Snorm16x4
        | GlVertexFormat::Float16x4 => 8,
        GlVertexFormat::Float32 | GlVertexFormat::Uint32 | GlVertexFormat::Sint32 => 4,
        GlVertexFormat::Float32x2 | GlVertexFormat::Uint32x2 | GlVertexFormat::Sint32x2 => 8,
        GlVertexFormat::Float32x3 | GlVertexFormat::Uint32x3 | GlVertexFormat::Sint32x3 => 12,
        GlVertexFormat::Float32x4 | GlVertexFormat::Uint32x4 | GlVertexFormat::Sint32x4 => 16,
    }
}

/// Total bytes one indexed draw consumes, from its base binding offset.
pub(super) fn indexed_draw_span(
    index: GlIndexBinding,
    first_index: u32,
    index_count: u32,
) -> Option<u64> {
    let size = index_size(index.format);
    let start = u64::from(first_index).checked_mul(size)?;
    let count = u64::from(index_count).checked_mul(size)?;
    index.offset.checked_add(start)?.checked_add(count)
}

/// The index type constant for an index binding.
pub(super) const fn indexed_draw_type(index: GlIndexBinding) -> u32 {
    index_type(index.format)
}

/// The byte offset one indexed draw starts at.
pub(super) fn indexed_draw_offset(index: GlIndexBinding, first_index: u32) -> Option<u64> {
    let size = index_size(index.format);
    index
        .offset
        .checked_add(u64::from(first_index).checked_mul(size)?)
}
