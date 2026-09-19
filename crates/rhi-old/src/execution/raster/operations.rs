//! Raster command validation for the fixed execution profile.
//!
//! These checks define the closed recipe ABI before forwarding a command to
//! the private native layer. In particular, recipe-specific base offsets and
//! index origins are intentionally not general drawing controls: accepting
//! them would make the registered resource recipe describe different data.

use super::*;
use crate::execution::helpers::require_device;

/// Closed mesh input recipes keep their binding base fixed at zero; R02
/// retains its established aligned-offset behavior.
pub(in crate::execution) fn raster_recipe_allows_vertex_offset(
    kernel: crate::RasterKernel,
    offset: u64,
) -> bool {
    !matches!(
        kernel,
        crate::RasterKernel::IndexedPositionFloat32x3
            | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterial
            | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTexture
            | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUv
            | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
            | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb
            | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialVertexColor
    ) || offset == 0
}

/// The same fixed-base rule applies to the recipe's Uint32 index data.
pub(in crate::execution) fn raster_recipe_allows_index_offset(
    kernel: crate::RasterKernel,
    offset: u64,
) -> bool {
    !matches!(
        kernel,
        crate::RasterKernel::IndexedPositionFloat32x3
            | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterial
            | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTexture
            | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUv
            | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
            | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb
            | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialVertexColor
    ) || offset == 0
}

/// Fixed mesh recipes draw from their first index; R02 alone may select an
/// aligned subrange.
pub(in crate::execution) fn raster_recipe_allows_first_index(
    kernel: Option<crate::RasterKernel>,
    first_index: u32,
) -> bool {
    !matches!(
        kernel,
        Some(
            crate::RasterKernel::IndexedPositionFloat32x3
                | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterial
                | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTexture
                | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUv
                | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
                | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb
                | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialVertexColor
        )
    ) || first_index == 0
}

impl RasterBackend {
    pub(in crate::execution) fn check_texture_pipeline(
        &self,
        pipeline: &RasterPipeline,
    ) -> Result<(), NativeExecutionError> {
        require_device(
            pipeline.device_identity(),
            self.device.identity(),
            NativeExecutionError::ForeignResource,
        )
    }

    pub(in crate::execution) fn set_vertex(
        &mut self,
        encoder: &mut CopyEncoder,
        slot: u32,
        buffer: &Buffer,
        offset: u64,
    ) -> Result<(), NativeExecutionError> {
        self.check_encoder(encoder)?;
        self.check_buffer(buffer)?;
        if encoder
            .active_raster
            .as_ref()
            .is_some_and(|pipeline| Self::is_vertex_color_kernel(pipeline.kernel()))
        {
            if slot > 1 {
                return Err(NativeExecutionError::RasterVertexSlotOutOfRange { slot });
            }
            if encoder.raster_vertex_color_binding_epoch != Some(encoder.raster_vertex_color_epoch)
            {
                return Err(NativeExecutionError::RasterEpochMismatch);
            }
            let bindings = encoder
                .bound_raster_vertex_color
                .as_ref()
                .ok_or(NativeExecutionError::RasterStateMismatch)?;
            if encoder.raster_vertex_color_slots[slot as usize].is_some() {
                return Err(NativeExecutionError::RasterVertexSlotAlreadyBound { slot });
            }
            let expected_identity = if slot == 0 {
                bindings.position_identity()
            } else {
                bindings.color_identity()
            };
            if buffer.identity() != expected_identity {
                return Err(NativeExecutionError::RasterVertexRoleMismatch { slot });
            }
            let required = u64::from(bindings.vertex_count()) * if slot == 0 { 12 } else { 4 };
            if offset != 0 || buffer.descriptor().buffer.size != required {
                return Err(NativeExecutionError::RasterVertexRangeMismatch { slot });
            }
            if !buffer.allowed_usage().contains(BufferUsageKind::Vertex) {
                return Err(NativeExecutionError::RasterVertexUsageMismatch { slot });
            }
            crate::imp::set_vertex_buffer(
                &mut encoder.native,
                buffer.native(),
                offset,
                required,
                encoder
                    .active_raster
                    .as_ref()
                    .expect("checked active vertex-color pipeline")
                    .kernel(),
                slot,
                buffer.identity(),
                None,
                None,
                Some(bindings.native()),
            )
            .map_err(NativeExecutionError::Recording)?;
            encoder.raster_vertex_color_slots[slot as usize] = Some((buffer.clone(), offset));
            encoder.leases.push(buffer.lease().into());
            return Ok(());
        }
        if encoder
            .active_raster
            .as_ref()
            .is_some_and(|pipeline| Self::is_uv_kernel(pipeline.kernel()))
        {
            if slot > 1 {
                return Err(NativeExecutionError::RasterVertexSlotOutOfRange { slot });
            }
            if encoder.raster_uv_binding_epoch != Some(encoder.raster_uv_epoch) {
                return Err(NativeExecutionError::RasterEpochMismatch);
            }
            let bindings = encoder
                .bound_raster_uv_texture
                .as_ref()
                .ok_or(NativeExecutionError::RasterStateMismatch)?;
            if encoder.raster_uv_vertex_slots[slot as usize].is_some() {
                return Err(NativeExecutionError::RasterVertexSlotAlreadyBound { slot });
            }
            let expected_identity = if slot == 0 {
                bindings.position_identity()
            } else {
                bindings.texture_coordinate_identity()
            };
            if buffer.identity() != expected_identity {
                return Err(NativeExecutionError::RasterVertexRoleMismatch { slot });
            }
            let stride = if slot == 0 { 12 } else { 8 };
            let required = u64::from(bindings.vertex_count()) * stride;
            if offset != 0 || buffer.descriptor().buffer.size != required {
                return Err(NativeExecutionError::RasterVertexRangeMismatch { slot });
            }
            if !buffer.allowed_usage().contains(BufferUsageKind::Vertex) {
                return Err(NativeExecutionError::RasterVertexUsageMismatch { slot });
            }
            crate::imp::set_vertex_buffer(
                &mut encoder.native,
                buffer.native(),
                offset,
                required,
                encoder
                    .active_raster
                    .as_ref()
                    .expect("checked active UV pipeline")
                    .kernel(),
                slot,
                buffer.identity(),
                Some(bindings.native()),
                None,
                None,
            )
            .map_err(NativeExecutionError::Recording)?;
            encoder.raster_uv_vertex_slots[slot as usize] = Some((buffer.clone(), offset));
            encoder.leases.push(buffer.lease().into());
            return Ok(());
        }
        if encoder
            .active_raster
            .as_ref()
            .is_some_and(|pipeline| Self::is_normal_kernel(pipeline.kernel()))
        {
            if slot > 1 {
                return Err(NativeExecutionError::RasterVertexSlotOutOfRange { slot });
            }
            if encoder.raster_normal_binding_epoch != Some(encoder.raster_normal_epoch) {
                return Err(NativeExecutionError::RasterEpochMismatch);
            }
            let bindings = encoder
                .bound_raster_normal
                .as_ref()
                .ok_or(NativeExecutionError::RasterStateMismatch)?;
            if encoder.raster_normal_vertex_slots[slot as usize].is_some() {
                return Err(NativeExecutionError::RasterVertexSlotAlreadyBound { slot });
            }
            let expected_identity = if slot == 0 {
                bindings.position_identity()
            } else {
                bindings.normal_identity()
            };
            if buffer.identity() != expected_identity {
                return Err(NativeExecutionError::RasterVertexRoleMismatch { slot });
            }
            let required = u64::from(bindings.vertex_count()) * 12;
            if offset != 0 || buffer.descriptor().buffer.size != required {
                return Err(NativeExecutionError::RasterVertexRangeMismatch { slot });
            }
            if !buffer.allowed_usage().contains(BufferUsageKind::Vertex) {
                return Err(NativeExecutionError::RasterVertexUsageMismatch { slot });
            }
            crate::imp::set_vertex_buffer(
                &mut encoder.native,
                buffer.native(),
                offset,
                required,
                encoder
                    .active_raster
                    .as_ref()
                    .expect("checked active normal pipeline")
                    .kernel(),
                slot,
                buffer.identity(),
                None,
                Some(bindings.native()),
                None,
            )
            .map_err(NativeExecutionError::Recording)?;
            encoder.raster_normal_vertex_slots[slot as usize] = Some((buffer.clone(), offset));
            encoder.leases.push(buffer.lease().into());
            return Ok(());
        }
        if slot != 0
            || encoder.raster_extent.is_none()
            || offset > buffer.descriptor().buffer.size
            || !buffer.allowed_usage().contains(BufferUsageKind::Vertex)
        {
            return Err(NativeExecutionError::RasterStateMismatch);
        }
        let pipeline = encoder
            .active_raster
            .as_ref()
            .ok_or(NativeExecutionError::RasterStateMismatch)?;
        if !matches!(
            pipeline.kernel(),
            crate::RasterKernel::IndexedPositionColor
                | crate::RasterKernel::IndexedPositionFloat32x3
                | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterial
                | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTexture
        ) || !raster_recipe_allows_vertex_offset(pipeline.kernel(), offset)
            || !offset.is_multiple_of(4)
            || (buffer.descriptor().buffer.size - offset)
                < u64::from(pipeline.kernel().vertex_stride())
        {
            return Err(NativeExecutionError::RasterStateMismatch);
        }
        crate::imp::set_vertex_buffer(
            &mut encoder.native,
            buffer.native(),
            offset,
            buffer.descriptor().buffer.size - offset,
            pipeline.kernel(),
            slot,
            buffer.identity(),
            None,
            None,
            None,
        )
        .map_err(NativeExecutionError::Recording)?;
        encoder.vertex_buffer = Some((buffer.clone(), offset));
        encoder.leases.push(buffer.lease().into());
        Ok(())
    }

    pub(in crate::execution) fn set_index(
        &mut self,
        encoder: &mut CopyEncoder,
        buffer: &Buffer,
        offset: u64,
        format: IndexFormat,
    ) -> Result<(), NativeExecutionError> {
        self.check_encoder(encoder)?;
        self.check_buffer(buffer)?;
        if encoder
            .active_raster
            .as_ref()
            .is_some_and(|pipeline| Self::is_vertex_color_kernel(pipeline.kernel()))
        {
            if encoder.raster_vertex_color_binding_epoch != Some(encoder.raster_vertex_color_epoch)
                || encoder.bound_raster_vertex_color.is_none()
            {
                return Err(NativeExecutionError::RasterEpochMismatch);
            }
            if format != IndexFormat::Uint32
                || offset != 0
                || !buffer.allowed_usage().contains(BufferUsageKind::Index)
            {
                return Err(NativeExecutionError::RasterStateMismatch);
            }
            crate::imp::set_index_buffer(
                &mut encoder.native,
                buffer.native(),
                offset,
                buffer.descriptor().buffer.size,
                format,
            )
            .map_err(NativeExecutionError::Recording)?;
            encoder.index_buffer = Some((buffer.clone(), offset, format));
            encoder.raster_vertex_color_index_ready = true;
            encoder.leases.push(buffer.lease().into());
            return Ok(());
        }
        if encoder
            .active_raster
            .as_ref()
            .is_some_and(|pipeline| Self::is_uv_kernel(pipeline.kernel()))
        {
            if encoder.raster_uv_binding_epoch != Some(encoder.raster_uv_epoch)
                || encoder.bound_raster_uv_texture.is_none()
            {
                return Err(NativeExecutionError::RasterEpochMismatch);
            }
            if format != IndexFormat::Uint32
                || offset != 0
                || !buffer.allowed_usage().contains(BufferUsageKind::Index)
            {
                return Err(NativeExecutionError::RasterStateMismatch);
            }
            crate::imp::set_index_buffer(
                &mut encoder.native,
                buffer.native(),
                offset,
                buffer.descriptor().buffer.size,
                format,
            )
            .map_err(NativeExecutionError::Recording)?;
            encoder.index_buffer = Some((buffer.clone(), offset, format));
            encoder.raster_uv_index_ready = true;
            encoder.leases.push(buffer.lease().into());
            return Ok(());
        }
        if encoder
            .active_raster
            .as_ref()
            .is_some_and(|pipeline| Self::is_normal_kernel(pipeline.kernel()))
        {
            if encoder.raster_normal_binding_epoch != Some(encoder.raster_normal_epoch)
                || encoder.bound_raster_normal.is_none()
            {
                return Err(NativeExecutionError::RasterEpochMismatch);
            }
            if format != IndexFormat::Uint32
                || offset != 0
                || !buffer.allowed_usage().contains(BufferUsageKind::Index)
            {
                return Err(NativeExecutionError::RasterStateMismatch);
            }
            crate::imp::set_index_buffer(
                &mut encoder.native,
                buffer.native(),
                offset,
                buffer.descriptor().buffer.size,
                format,
            )
            .map_err(NativeExecutionError::Recording)?;
            encoder.index_buffer = Some((buffer.clone(), offset, format));
            encoder.raster_normal_index_ready = true;
            encoder.leases.push(buffer.lease().into());
            return Ok(());
        }
        if encoder.raster_extent.is_none()
            || !offset.is_multiple_of(match format {
                IndexFormat::Uint16 => 2,
                IndexFormat::Uint32 => 4,
            })
            || offset >= buffer.descriptor().buffer.size
            || !buffer.allowed_usage().contains(BufferUsageKind::Index)
        {
            return Err(NativeExecutionError::RasterStateMismatch);
        }
        let expected_format = match encoder.active_raster.as_ref().map(RasterPipeline::kernel) {
            Some(crate::RasterKernel::IndexedPositionColor) => IndexFormat::Uint16,
            Some(crate::RasterKernel::IndexedPositionFloat32x3)
            | Some(crate::RasterKernel::IndexedPositionFloat32x3CameraMaterial)
            | Some(crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTexture) => {
                IndexFormat::Uint32
            }
            Some(crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialNormalLambert) => {
                IndexFormat::Uint32
            }
            _ => return Err(NativeExecutionError::RasterStateMismatch),
        };
        if format != expected_format
            || !raster_recipe_allows_index_offset(
                encoder
                    .active_raster
                    .as_ref()
                    .expect("active recipe")
                    .kernel(),
                offset,
            )
        {
            return Err(NativeExecutionError::RasterStateMismatch);
        }
        crate::imp::set_index_buffer(
            &mut encoder.native,
            buffer.native(),
            offset,
            buffer.descriptor().buffer.size - offset,
            format,
        )
        .map_err(NativeExecutionError::Recording)?;
        encoder.index_buffer = Some((buffer.clone(), offset, format));
        encoder.leases.push(buffer.lease().into());
        Ok(())
    }

    pub(in crate::execution) fn set_viewport_checked(
        &mut self,
        encoder: &mut CopyEncoder,
        viewport: Viewport,
    ) -> Result<(), NativeExecutionError> {
        self.check_encoder(encoder)?;
        let (width, height) = encoder
            .raster_extent
            .ok_or(NativeExecutionError::RasterStateMismatch)?;
        let inside = viewport.x >= 0.0
            && viewport.y >= 0.0
            && viewport.x + viewport.width <= width as f32
            && viewport.y + viewport.height <= height as f32;
        if !viewport.x.is_finite()
            || !viewport.y.is_finite()
            || !viewport.width.is_finite()
            || !viewport.height.is_finite()
            || !viewport.min_depth.is_finite()
            || !viewport.max_depth.is_finite()
            || viewport.width <= 0.0
            || viewport.height <= 0.0
            || !(0.0..=1.0).contains(&viewport.min_depth)
            || !(0.0..=1.0).contains(&viewport.max_depth)
            || viewport.min_depth > viewport.max_depth
            || !inside
        {
            return Err(NativeExecutionError::RasterStateMismatch);
        }
        crate::imp::set_viewport(
            &mut encoder.native,
            viewport.x,
            viewport.y,
            viewport.width,
            viewport.height,
            viewport.min_depth,
            viewport.max_depth,
        )
        .map_err(NativeExecutionError::Recording)
    }

    pub(in crate::execution) fn set_scissor_checked(
        &mut self,
        encoder: &mut CopyEncoder,
        scissor: ScissorRect,
    ) -> Result<(), NativeExecutionError> {
        self.check_encoder(encoder)?;
        let (width, height) = encoder
            .raster_extent
            .ok_or(NativeExecutionError::RasterStateMismatch)?;
        let Some(right) = scissor.x.checked_add(scissor.width) else {
            return Err(NativeExecutionError::RasterStateMismatch);
        };
        let Some(bottom) = scissor.y.checked_add(scissor.height) else {
            return Err(NativeExecutionError::RasterStateMismatch);
        };
        if scissor.width == 0 || scissor.height == 0 || right > width || bottom > height {
            return Err(NativeExecutionError::RasterStateMismatch);
        }
        crate::imp::set_scissor(
            &mut encoder.native,
            scissor.x,
            scissor.y,
            scissor.width,
            scissor.height,
        )
        .map_err(NativeExecutionError::Recording)
    }

    pub(in crate::execution) fn draw_checked(
        &mut self,
        encoder: &mut CopyEncoder,
        vertices: Range<u32>,
        instances: Range<u32>,
    ) -> Result<(), NativeExecutionError> {
        self.check_encoder(encoder)?;
        if vertices != (0..3)
            || instances != (0..1)
            || encoder.raster_extent.is_none()
            || encoder.active_raster.as_ref().map(RasterPipeline::kernel)
                != Some(crate::RasterKernel::Triangle)
        {
            return Err(NativeExecutionError::RasterStateMismatch);
        }
        crate::imp::draw(
            &mut encoder.native,
            vertices.start,
            vertices.end - vertices.start,
            instances.start,
            instances.end - instances.start,
        )
        .map_err(NativeExecutionError::Recording)
    }

    pub(in crate::execution) fn draw_indexed_checked(
        &mut self,
        encoder: &mut CopyEncoder,
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    ) -> Result<(), NativeExecutionError> {
        self.check_encoder(encoder)?;
        let uv_kernel = encoder
            .active_raster
            .as_ref()
            .is_some_and(|pipeline| Self::is_uv_kernel(pipeline.kernel()));
        let normal_kernel = encoder
            .active_raster
            .as_ref()
            .is_some_and(|pipeline| Self::is_normal_kernel(pipeline.kernel()));
        let vertex_color_kernel = encoder
            .active_raster
            .as_ref()
            .is_some_and(|pipeline| Self::is_vertex_color_kernel(pipeline.kernel()));
        if indices.start >= indices.end
            || !raster_recipe_allows_first_index(
                encoder.active_raster.as_ref().map(RasterPipeline::kernel),
                indices.start,
            )
            || base_vertex != 0
            || instances != (0..1)
            || encoder.raster_extent.is_none()
            || (!uv_kernel
                && !normal_kernel
                && !vertex_color_kernel
                && encoder.vertex_buffer.is_none())
            || encoder.index_buffer.is_none()
        {
            return Err(NativeExecutionError::RasterStateMismatch);
        }
        let (buffer, offset, format) = encoder
            .index_buffer
            .as_ref()
            .expect("checked index binding");
        let kernel = encoder.active_raster.as_ref().map(RasterPipeline::kernel);
        let expected_format = match kernel {
            Some(crate::RasterKernel::IndexedPositionColor) => IndexFormat::Uint16,
            Some(crate::RasterKernel::IndexedPositionFloat32x3)
            | Some(crate::RasterKernel::IndexedPositionFloat32x3CameraMaterial)
            | Some(crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTexture)
            | Some(crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUv)
            | Some(
                crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp,
            )
            | Some(
                crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb,
            ) => IndexFormat::Uint32,
            Some(crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialNormalLambert) => {
                IndexFormat::Uint32
            }
            Some(crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialVertexColor) => {
                IndexFormat::Uint32
            }
            _ => return Err(NativeExecutionError::RasterStateMismatch),
        };
        let index_size = match expected_format {
            IndexFormat::Uint16 => 2,
            IndexFormat::Uint32 => 4,
        };
        let byte_end = u64::from(indices.end)
            .checked_mul(index_size)
            .and_then(|end| offset.checked_add(end))
            .ok_or(NativeExecutionError::RasterStateMismatch)?;
        if *format != expected_format || byte_end > buffer.descriptor().buffer.size {
            return Err(NativeExecutionError::RasterStateMismatch);
        }
        if kernel == Some(crate::RasterKernel::IndexedPositionFloat32x3CameraMaterial)
            && encoder.bound_raster_uniform.is_none()
        {
            return Err(NativeExecutionError::RasterStateMismatch);
        }
        if kernel == Some(crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTexture)
            && encoder.bound_raster_texture.is_none()
        {
            return Err(NativeExecutionError::RasterStateMismatch);
        }
        if kernel.is_some_and(Self::is_uv_kernel) {
            if encoder.raster_uv_binding_epoch != Some(encoder.raster_uv_epoch) {
                return Err(NativeExecutionError::RasterEpochMismatch);
            }
            if encoder.raster_uv_vertex_slots[0].is_none() {
                return Err(NativeExecutionError::RasterVertexSlotMissing { slot: 0 });
            }
            if encoder.raster_uv_vertex_slots[1].is_none() {
                return Err(NativeExecutionError::RasterVertexSlotMissing { slot: 1 });
            }
            if !encoder.raster_uv_index_ready {
                return Err(NativeExecutionError::RasterStateMismatch);
            }
        }
        if kernel.is_some_and(Self::is_normal_kernel) {
            if encoder.raster_normal_binding_epoch != Some(encoder.raster_normal_epoch) {
                return Err(NativeExecutionError::RasterEpochMismatch);
            }
            if encoder.raster_normal_vertex_slots[0].is_none() {
                return Err(NativeExecutionError::RasterVertexSlotMissing { slot: 0 });
            }
            if encoder.raster_normal_vertex_slots[1].is_none() {
                return Err(NativeExecutionError::RasterVertexSlotMissing { slot: 1 });
            }
            if !encoder.raster_normal_index_ready {
                return Err(NativeExecutionError::RasterStateMismatch);
            }
        }
        if vertex_color_kernel {
            if encoder.raster_vertex_color_binding_epoch != Some(encoder.raster_vertex_color_epoch)
            {
                return Err(NativeExecutionError::RasterEpochMismatch);
            }
            if encoder.raster_vertex_color_slots[0].is_none() {
                return Err(NativeExecutionError::RasterVertexSlotMissing { slot: 0 });
            }
            if encoder.raster_vertex_color_slots[1].is_none() {
                return Err(NativeExecutionError::RasterVertexSlotMissing { slot: 1 });
            }
            if !encoder.raster_vertex_color_index_ready {
                return Err(NativeExecutionError::RasterStateMismatch);
            }
        }
        crate::imp::draw_indexed(
            &mut encoder.native,
            indices.start,
            indices.end - indices.start,
            base_vertex,
            instances.start,
            instances.end - instances.start,
        )
        .map_err(NativeExecutionError::Recording)
    }

    pub(in crate::execution) fn dispatch_checked(
        &mut self,
        encoder: &mut CopyEncoder,
        groups: [u32; 3],
    ) -> Result<(), NativeExecutionError> {
        self.check_encoder(encoder)?;
        if !encoder.compute_open {
            return Err(NativeExecutionError::ComputeBindingMismatch);
        }
        if !valid_compute_dispatch(
            groups,
            self.capabilities
                .limits
                .max_compute_workgroups_per_dimension,
        ) {
            return Err(NativeExecutionError::InvalidDispatch);
        }
        if encoder.active_compute.is_none() || encoder.bound_compute_pipeline.is_none() {
            return Err(NativeExecutionError::ComputeBindingMismatch);
        }
        crate::imp::dispatch(&mut encoder.native, groups).map_err(NativeExecutionError::Recording)
    }
}
