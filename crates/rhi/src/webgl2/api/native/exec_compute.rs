//! Native compute, storage, and indirect execution.
//!
//! Every entry point preflights the proved capability, the installed program,
//! and the exact range or binding limits before touching the driver. The
//! compute program is installed through `set_compute_program`; indirect draws
//! reuse the installed raster topology and its recorded index binding.

use super::super::{
    GlBufferRange, GlBufferUsage, GlCapability, GlComputeDispatchApi, GlComputeLimits,
    GlDispatchGroups, GlDispatchIndirectApi, GlDrawIndirectApi, GlError, GlFamilyApi as _,
    GlIndirectAbi, GlIndirectCommandRange, GlMemoryBarrier, GlProgramKind, GlStorageBufferApi,
    GlStorageBufferLimits, GlStorageBufferRange, GlStorageImageAccess, GlStorageImageApi,
    GlStorageImageBinding, GlStorageImageLimits, GlTextureUsage,
};
use super::provider::NativeGlProvider;

impl GlComputeDispatchApi for NativeGlProvider<'_> {
    /// Installs the compute program dispatch work executes.
    fn set_compute_program(&mut self, program: super::super::ProgramId) -> Result<(), GlError> {
        const OP: &str = "set-compute-program";
        self.assert_ready(OP)?;
        if !self
            .discovery
            .capabilities()
            .supports(GlCapability::Compute)
        {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "this context did not prove the compute capability",
            });
        }
        let record = self.program(OP, program)?;
        if !matches!(&record.descriptor.kind, GlProgramKind::Compute { .. }) {
            return Err(Self::validation(
                OP,
                "compute dispatch requires a compute program",
            ));
        }
        self.active_compute_program = Some(program);
        Ok(())
    }

    fn dispatch(&mut self, groups: GlDispatchGroups) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "dispatch";
        self.assert_ready(OP)?;
        if !self
            .discovery
            .capabilities()
            .supports(GlCapability::Compute)
        {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "this context did not prove the compute capability",
            });
        }
        let Some(program) = self.active_compute_program else {
            return Err(Self::validation(OP, "no compute program is installed"));
        };
        self.program(OP, program)?;
        let limits = self.discovery.limits();
        if groups
            .validate(GlComputeLimits {
                max_group_count: limits.max_compute_work_group_count,
                max_group_size: limits.max_compute_work_group_size,
                max_group_invocations: limits.max_compute_work_group_invocations,
            })
            .is_err()
        {
            return Err(Self::validation(
                OP,
                "workgroup count is zero or exceeds a discovered axis limit",
            ));
        }
        // SAFETY: current-context contract; capability, program, and group
        // limits were validated before the dispatch.
        unsafe {
            self.gl
                .dispatch_compute(groups.0[0], groups.0[1], groups.0[2]);
        }
        self.driver_error(OP)
    }

    fn memory_barrier(&mut self, barriers: GlMemoryBarrier) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "memory-barrier";
        self.assert_ready(OP)?;
        barriers.validate_nonempty()?;
        // SAFETY: current-context contract; the bit set is nonempty and every
        // translated bit is a valid barrier class on this profile.
        unsafe {
            self.gl.memory_barrier(native_barrier_bits(barriers.0));
        }
        self.driver_error(OP)
    }
}

impl GlStorageBufferApi for NativeGlProvider<'_> {
    fn bind_storage_buffer(
        &mut self,
        binding: u32,
        range: GlStorageBufferRange,
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "bind-storage-buffer";
        self.assert_ready(OP)?;
        if !self
            .discovery
            .capabilities()
            .supports(GlCapability::StorageBuffer)
        {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "this context did not prove the storage-buffer capability",
            });
        }
        let limits = self.discovery.limits();
        range.validate(
            binding,
            GlStorageBufferLimits {
                max_bindings: limits.max_storage_buffer_bindings,
                max_block_size: limits.max_storage_block_size,
                offset_alignment: limits.storage_buffer_offset_alignment,
            },
        )?;
        let (name, desc) = self.buffer(OP, range.buffer)?;
        if !desc.usage.contains(GlBufferUsage::STORAGE) {
            return Err(Self::validation(OP, "buffer lacks storage usage"));
        }
        GlBufferRange {
            buffer: range.buffer,
            offset: range.offset,
            size: range.size,
        }
        .validate_for(desc)
        .map_err(|_| Self::validation(OP, "storage buffer range is outside the allocation"))?;
        let (offset, size) = {
            let offset = i32::try_from(range.offset)
                .map_err(|_| Self::validation(OP, "offset exceeds GLintptr"))?;
            let size = i32::try_from(range.size)
                .map_err(|_| Self::validation(OP, "size exceeds GLintptr"))?;
            (offset, size)
        };
        // SAFETY: current-context contract; binding index, alignment, usage,
        // and range were validated against the live allocation.
        unsafe {
            self.gl.bind_buffer_range(
                glow::SHADER_STORAGE_BUFFER,
                binding,
                Some(name),
                offset,
                size,
            );
        }
        self.driver_error(OP)
    }
}

impl GlStorageImageApi for NativeGlProvider<'_> {
    fn bind_storage_image(
        &mut self,
        binding: u32,
        image: GlStorageImageBinding,
    ) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "bind-storage-image";
        self.assert_ready(OP)?;
        if !self
            .discovery
            .capabilities()
            .supports(GlCapability::StorageImage)
        {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "this context did not prove the storage-image capability",
            });
        }
        image.validate(
            binding,
            GlStorageImageLimits {
                max_image_units: self.discovery.limits().max_image_units,
            },
            self.discovery.formats(),
        )?;
        let (name, desc) = self.texture(OP, image.texture)?;
        if !desc.usage.contains(GlTextureUsage::STORAGE_BINDING) {
            return Err(Self::validation(OP, "texture lacks storage-binding usage"));
        }
        if desc.format != image.format || desc.sample_count != image.sample_count {
            return Err(Self::validation(
                OP,
                "image format or sample count does not match the allocation",
            ));
        }
        let mip = desc
            .mip_extent(image.level)
            .ok_or_else(|| Self::validation(OP, "storage image mip level is invalid"))?;
        let layered = matches!(
            desc.dimension,
            super::super::GlTextureDimension::D3
                | super::super::GlTextureDimension::D2Array
                | super::super::GlTextureDimension::Cube
        );
        let layer = match (image.layered, image.layer) {
            (true, None) if layered => -1,
            (false, Some(layer)) if !layered && (layer as u64) < mip.depth_or_layers as u64 => {
                layer as i32
            }
            _ => {
                return Err(Self::validation(
                    OP,
                    "image layer selection is invalid for the texture shape",
                ));
            }
        };
        let format = native_image_format(image.format).ok_or(GlError::Unsupported {
            operation: OP,
            reason: "format has no proved native image-unit mapping",
        })?;
        // SAFETY: current-context contract; capability, unit index, layer
        // selection, and the exact format access fact were all validated.
        unsafe {
            self.gl.bind_image_texture(
                binding,
                Some(name),
                image.level as i32,
                image.layered,
                layer,
                native_image_access(image.access),
                format,
            );
        }
        self.driver_error(OP)
    }
}

impl GlDrawIndirectApi for NativeGlProvider<'_> {
    fn draw_indirect(&mut self, command: GlIndirectCommandRange) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "draw-indirect";
        self.assert_ready(OP)?;
        if !self
            .discovery
            .capabilities()
            .supports(GlCapability::IndirectDraw)
        {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "this context did not prove the indirect-draw capability",
            });
        }
        command.validate(OP)?;
        let raster = self
            .raster
            .as_ref()
            .ok_or_else(|| Self::validation(OP, "no raster pipeline is installed"))?;
        let vertex_array = self.vertex_array(OP, raster.vertex_array)?;
        let (name, desc) = self.buffer(OP, command.range.buffer)?;
        if !desc.usage.contains(GlBufferUsage::INDIRECT) {
            return Err(Self::validation(OP, "command buffer lacks indirect usage"));
        }
        let offset = i32::try_from(command.command_offset)
            .map_err(|_| Self::validation(OP, "command offset exceeds i32"))?;
        let mode = super::exec_raster::topology_mode(raster.topology);
        // SAFETY: current-context contract; the ABI, range, usage, and active
        // pipeline state were validated before any binding changed.
        unsafe {
            self.gl.bind_buffer(glow::DRAW_INDIRECT_BUFFER, Some(name));
            match command.abi {
                GlIndirectAbi::NonIndexed => {
                    self.gl.draw_arrays_indirect_offset(mode, offset);
                }
                GlIndirectAbi::Indexed => {
                    let Some(index) = vertex_array.index else {
                        return Err(Self::validation(
                            OP,
                            "indexed indirect draw requires a bound index buffer",
                        ));
                    };
                    self.buffer(OP, index.buffer)?;
                    self.gl.draw_elements_indirect_offset(
                        mode,
                        super::exec_vertex::indexed_draw_type(index),
                        offset,
                    );
                }
            }
        }
        self.driver_error(OP)
    }
}

impl GlDispatchIndirectApi for NativeGlProvider<'_> {
    fn dispatch_indirect(&mut self, command: GlBufferRange, offset: u64) -> Result<(), GlError> {
        use glow::HasContext as _;
        const OP: &str = "dispatch-indirect";
        self.assert_ready(OP)?;
        if !self
            .discovery
            .capabilities()
            .supports(GlCapability::IndirectDispatch)
        {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "this context did not prove the indirect-dispatch capability",
            });
        }
        let Some(program) = self.active_compute_program else {
            return Err(Self::validation(OP, "no compute program is installed"));
        };
        self.program(OP, program)?;
        let (name, desc) = self.buffer(OP, command.buffer)?;
        if !desc.usage.contains(GlBufferUsage::INDIRECT) {
            return Err(Self::validation(OP, "command buffer lacks indirect usage"));
        }
        // One dispatch record is three u32 work-group counts.
        let record_bytes = 12u64;
        if offset % 4 != 0
            || offset
                .checked_add(record_bytes)
                .is_none_or(|end| end > command.size)
        {
            return Err(Self::validation(
                OP,
                "indirect dispatch offset leaves the buffer range",
            ));
        }
        let buffer_offset = command
            .offset
            .checked_add(offset)
            .ok_or_else(|| Self::validation(OP, "indirect dispatch offset overflows"))?;
        let offset =
            i32::try_from(buffer_offset).map_err(|_| Self::validation(OP, "offset exceeds i32"))?;
        // SAFETY: current-context contract; the range, usage, and program were
        // validated before the binding changed.
        unsafe {
            self.gl
                .bind_buffer(glow::DISPATCH_INDIRECT_BUFFER, Some(name));
            self.gl.dispatch_compute_indirect(offset);
        }
        self.driver_error(OP)
    }
}

/// Translates the contract's barrier classes into GL memory-barrier bits.
const fn native_barrier_bits(barriers: u32) -> u32 {
    let mut bits = 0;
    if barriers & GlMemoryBarrier::SHADER_STORAGE.0 != 0 {
        bits |= glow::SHADER_STORAGE_BARRIER_BIT;
    }
    if barriers & GlMemoryBarrier::SHADER_IMAGE_ACCESS.0 != 0 {
        bits |= glow::SHADER_IMAGE_ACCESS_BARRIER_BIT;
    }
    if barriers & GlMemoryBarrier::TEXTURE_FETCH.0 != 0 {
        bits |= glow::TEXTURE_FETCH_BARRIER_BIT;
    }
    if barriers & GlMemoryBarrier::VERTEX_ATTRIB_ARRAY.0 != 0 {
        bits |= glow::VERTEX_ATTRIB_ARRAY_BARRIER_BIT;
    }
    if barriers & GlMemoryBarrier::COMMAND.0 != 0 {
        bits |= glow::COMMAND_BARRIER_BIT;
    }
    bits
}

/// The image-unit format constant for one proved storage format.
const fn native_image_format(format: super::super::GlFormat) -> Option<u32> {
    match format {
        super::super::GlFormat::Rgba8Unorm => Some(glow::RGBA8),
        _ => None,
    }
}

const fn native_image_access(access: GlStorageImageAccess) -> u32 {
    match access {
        GlStorageImageAccess::ReadOnly => glow::READ_ONLY,
        GlStorageImageAccess::WriteOnly => glow::WRITE_ONLY,
        GlStorageImageAccess::ReadWrite => glow::READ_WRITE,
    }
}
