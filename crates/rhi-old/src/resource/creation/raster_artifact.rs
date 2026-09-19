//! Fixed raster artifact construction.
use super::super::bindings::{
    RasterNormalBindingsShared, RasterPipelineShared, RasterTextureBindingsShared,
    RasterUniformBindingsShared, RasterUvLinearClampTextureBindingsShared,
    RasterUvTextureBindingsShared, RasterVertexColorBindingsShared,
};
use super::super::validation::*;
use super::super::*;
use std::sync::Arc;
impl Device {
    /// Creates one fixed-artifact raster pipeline for this device.
    pub fn create_raster_pipeline(
        &self,
        kernel: RasterKernel,
    ) -> Result<RasterPipeline, RasterCreateError> {
        if kernel == RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
            && !self.capabilities().rgba8_unorm_filterable
        {
            return Err(RasterCreateError::TextureFormatNotFilterable {
                format: TextureFormat::Rgba8Unorm,
            });
        }
        if kernel == RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb
            && !self.capabilities().rgba8_unorm_srgb_filterable
        {
            return Err(RasterCreateError::TextureFormatNotFilterable {
                format: TextureFormat::Rgba8UnormSrgb,
            });
        }
        #[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
        let native = crate::imp::create_raster_pipeline(
            &self.inner,
            kernel.wgsl_source(),
            kernel.vertex_entry_point(),
            kernel.fragment_entry_point(),
            kernel,
        )
        .map_err(crate::resource::raster::map_raster_pipeline_create_error)?;
        #[cfg(not(all(windows, any(feature = "dx12", feature = "vulkan"))))]
        let native = crate::imp::create_raster_pipeline(
            &self.inner,
            kernel.wgsl_source(),
            kernel.vertex_entry_point(),
            kernel.fragment_entry_point(),
            kernel,
        )
        .map_err(RasterCreateError::NativeObjectCreation)?;
        Ok(RasterPipeline(Arc::new(RasterPipelineShared {
            _native: native,
            kernel,
            device: self.identity,
        })))
    }

    /// Creates the only group-0/binding-0 uniform binding accepted by the
    /// camera/material raster artifact.
    pub fn create_raster_uniform_bindings(
        &self,
        pipeline: &RasterPipeline,
        buffer: &Buffer,
    ) -> Result<RasterUniformBindings, RasterCreateError> {
        validate_raster_uniform_contract(
            pipeline.kernel(),
            buffer.descriptor().buffer.size,
            buffer.allowed_usage(),
        )?;
        if pipeline.device_identity() != self.identity || buffer.device_identity() != self.identity
        {
            return Err(RasterCreateError::ForeignDevice);
        }
        let native = crate::imp::create_raster_uniform_bindings(
            &self.inner,
            pipeline.native(),
            buffer.native(),
        )
        .map_err(RasterCreateError::NativeFailure)?;
        Ok(RasterUniformBindings(Arc::new(
            RasterUniformBindingsShared {
                _native: native,
                pipeline: pipeline.clone(),
                _buffer: buffer.lease(),
                device: self.identity,
            },
        )))
    }

    /// Creates the closed normal-Lambert binding. It fixes both vertex stream
    /// roles and exact whole-stream ranges before the native recording edge.
    #[allow(
        clippy::too_many_arguments,
        reason = "the closed ABI independently receives uniform, position, normal, and count"
    )]
    pub fn create_raster_normal_bindings(
        &self,
        pipeline: &RasterPipeline,
        uniform: &Buffer,
        positions: &Buffer,
        normals: &Buffer,
        vertex_count: u32,
    ) -> Result<RasterNormalBindings, RasterCreateError> {
        if pipeline.kernel() != RasterKernel::IndexedPositionFloat32x3CameraMaterialNormalLambert {
            return Err(RasterCreateError::BindingRecipeMismatch);
        }
        if pipeline.device_identity() != self.identity
            || uniform.device_identity() != self.identity
            || positions.device_identity() != self.identity
            || normals.device_identity() != self.identity
        {
            return Err(RasterCreateError::ForeignDevice);
        }
        validate_raster_uniform_contract(
            pipeline.kernel(),
            uniform.descriptor().buffer.size,
            uniform.allowed_usage(),
        )?;
        let stream_size = u64::from(vertex_count)
            .checked_mul(12)
            .ok_or(RasterCreateError::VertexStreamCountMismatch)?;
        if vertex_count == 0 || positions.descriptor().buffer.size != stream_size {
            return Err(RasterCreateError::InvalidPositionStreamRange);
        }
        if normals.descriptor().buffer.size != stream_size {
            return Err(RasterCreateError::InvalidNormalStreamRange);
        }
        if !positions.allowed_usage().contains(BufferUsageKind::Vertex)
            || !normals.allowed_usage().contains(BufferUsageKind::Vertex)
        {
            return Err(RasterCreateError::VertexUsageRequired);
        }
        let native = crate::imp::create_raster_normal_bindings(
            &self.inner,
            pipeline.native(),
            uniform.native(),
            positions.identity(),
            stream_size,
            normals.identity(),
            stream_size,
        )
        .map_err(RasterCreateError::NativeFailure)?;
        Ok(RasterNormalBindings(Arc::new(RasterNormalBindingsShared {
            _native: native,
            pipeline: pipeline.clone(),
            _uniform: uniform.lease(),
            _positions: positions.lease(),
            _normals: normals.lease(),
            position_identity: positions.identity(),
            normal_identity: normals.identity(),
            vertex_count,
            device: self.identity,
        })))
    }

    /// Creates the closed camera/material position-and-RGBA8 vertex-color binding.
    #[allow(
        clippy::too_many_arguments,
        reason = "the fixed ABI has independent uniform and stream roles"
    )]
    pub fn create_raster_vertex_color_bindings(
        &self,
        pipeline: &RasterPipeline,
        uniform: &Buffer,
        positions: &Buffer,
        colors: &Buffer,
        vertex_count: u32,
    ) -> Result<RasterVertexColorBindings, RasterCreateError> {
        if pipeline.kernel() != RasterKernel::IndexedPositionFloat32x3CameraMaterialVertexColor {
            return Err(RasterCreateError::BindingRecipeMismatch);
        }
        if pipeline.device_identity() != self.identity
            || uniform.device_identity() != self.identity
            || positions.device_identity() != self.identity
            || colors.device_identity() != self.identity
        {
            return Err(RasterCreateError::ForeignDevice);
        }
        validate_raster_uniform_contract(
            pipeline.kernel(),
            uniform.descriptor().buffer.size,
            uniform.allowed_usage(),
        )?;
        let position_size = u64::from(vertex_count)
            .checked_mul(12)
            .ok_or(RasterCreateError::VertexStreamCountMismatch)?;
        let color_size = u64::from(vertex_count)
            .checked_mul(4)
            .ok_or(RasterCreateError::VertexStreamCountMismatch)?;
        if vertex_count == 0 || positions.descriptor().buffer.size != position_size {
            return Err(RasterCreateError::InvalidPositionStreamRange);
        }
        if colors.descriptor().buffer.size != color_size {
            return Err(RasterCreateError::InvalidColorStreamRange);
        }
        if !positions.allowed_usage().contains(BufferUsageKind::Vertex)
            || !colors.allowed_usage().contains(BufferUsageKind::Vertex)
        {
            return Err(RasterCreateError::VertexUsageRequired);
        }
        let native = crate::imp::create_raster_vertex_color_bindings(
            &self.inner,
            pipeline.native(),
            uniform.native(),
            positions.identity(),
            position_size,
            colors.identity(),
            color_size,
        )
        .map_err(RasterCreateError::NativeFailure)?;
        Ok(RasterVertexColorBindings(Arc::new(
            RasterVertexColorBindingsShared {
                _native: native,
                pipeline: pipeline.clone(),
                _uniform: uniform.lease(),
                _positions: positions.lease(),
                _colors: colors.lease(),
                position_identity: positions.identity(),
                color_identity: colors.identity(),
                vertex_count,
                device: self.identity,
            },
        )))
    }

    /// Creates the only uniform-plus-whole-texture binding accepted by the textured raster artifact.
    pub fn create_raster_texture_bindings(
        &self,
        pipeline: &RasterPipeline,
        uniform: &Buffer,
        texture: &Texture,
    ) -> Result<RasterTextureBindings, RasterCreateError> {
        if pipeline.kernel() != RasterKernel::IndexedPositionFloat32x3CameraMaterialTexture {
            return Err(RasterCreateError::BindingRecipeMismatch);
        }
        if pipeline.device_identity() != self.identity
            || uniform.device_identity() != self.identity
            || texture.device_identity() != self.identity
        {
            return Err(RasterCreateError::ForeignDevice);
        }
        validate_raster_uniform_contract(
            RasterKernel::IndexedPositionFloat32x3CameraMaterial,
            uniform.descriptor().buffer.size,
            uniform.allowed_usage(),
        )?;
        validate_texture_pack_texture_desc(texture.descriptor().texture, texture.allowed_usage())
            .map_err(|_| RasterCreateError::BindingRecipeMismatch)?;
        let native = crate::imp::create_raster_texture_bindings(
            &self.inner,
            pipeline.native(),
            uniform.native(),
            texture.native(),
        )
        .map_err(RasterCreateError::NativeFailure)?;
        Ok(RasterTextureBindings(Arc::new(
            RasterTextureBindingsShared {
                _native: native,
                pipeline: pipeline.clone(),
                _uniform: uniform.lease(),
                _texture: texture.lease(),
                device: self.identity,
            },
        )))
    }

    /// Creates the closed explicit-UV textured raster binding. Both vertex
    /// streams are whole, tightly packed, same-device vertex buffers and are
    /// retained with their physical identities for later role validation.
    pub fn create_raster_uv_texture_bindings(
        &self,
        pipeline: &RasterPipeline,
        uniform: &Buffer,
        texture: &Texture,
        positions: &Buffer,
        texture_coordinates: &Buffer,
        vertex_count: u32,
    ) -> Result<RasterUvTextureBindings, RasterCreateError> {
        if pipeline.kernel() != RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUv {
            return Err(RasterCreateError::BindingRecipeMismatch);
        }
        if pipeline.device_identity() != self.identity
            || uniform.device_identity() != self.identity
            || texture.device_identity() != self.identity
            || positions.device_identity() != self.identity
            || texture_coordinates.device_identity() != self.identity
        {
            return Err(RasterCreateError::ForeignDevice);
        }
        validate_raster_uniform_contract(
            RasterKernel::IndexedPositionFloat32x3CameraMaterial,
            uniform.descriptor().buffer.size,
            uniform.allowed_usage(),
        )?;
        validate_texture_pack_texture_desc(texture.descriptor().texture, texture.allowed_usage())
            .map_err(|_| RasterCreateError::BindingRecipeMismatch)?;
        let position_size = u64::from(vertex_count)
            .checked_mul(12)
            .ok_or(RasterCreateError::VertexStreamCountMismatch)?;
        let uv_size = u64::from(vertex_count)
            .checked_mul(8)
            .ok_or(RasterCreateError::VertexStreamCountMismatch)?;
        if vertex_count == 0 || positions.descriptor().buffer.size != position_size {
            return Err(RasterCreateError::InvalidPositionStreamRange);
        }
        if texture_coordinates.descriptor().buffer.size != uv_size {
            return Err(RasterCreateError::InvalidTextureCoordinateStreamRange);
        }
        if !positions.allowed_usage().contains(BufferUsageKind::Vertex)
            || !texture_coordinates
                .allowed_usage()
                .contains(BufferUsageKind::Vertex)
        {
            return Err(RasterCreateError::VertexUsageRequired);
        }
        let native = crate::imp::create_raster_uv_texture_bindings(
            &self.inner,
            pipeline.native(),
            uniform.native(),
            texture.native(),
            positions.identity(),
            position_size,
            texture_coordinates.identity(),
            uv_size,
        )
        .map_err(RasterCreateError::NativeFailure)?;
        Ok(RasterUvTextureBindings(Arc::new(
            RasterUvTextureBindingsShared {
                _native: native,
                pipeline: pipeline.clone(),
                _uniform: uniform.lease(),
                _texture: texture.lease(),
                _positions: positions.lease(),
                _texture_coordinates: texture_coordinates.lease(),
                position_identity: positions.identity(),
                texture_coordinate_identity: texture_coordinates.identity(),
                vertex_count,
                device: self.identity,
            },
        )))
    }

    /// Creates the closed explicit-UV linear-clamp sampled-texture binding.
    ///
    /// The sampler is owned privately by the returned opaque binding; callers
    /// cannot configure filter, address, comparison, or LOD behavior.
    #[allow(
        clippy::too_many_arguments,
        reason = "the closed recipe has independent uniform, texture, and two vertex role inputs"
    )]
    pub fn create_raster_uv_linear_clamp_texture_bindings(
        &self,
        pipeline: &RasterPipeline,
        uniform: &Buffer,
        texture: &Texture,
        positions: &Buffer,
        texture_coordinates: &Buffer,
        vertex_count: u32,
    ) -> Result<RasterUvLinearClampTextureBindings, RasterCreateError> {
        if !self.capabilities().rgba8_unorm_filterable {
            return Err(RasterCreateError::TextureFormatNotFilterable {
                format: TextureFormat::Rgba8Unorm,
            });
        }
        if pipeline.kernel()
            != RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
        {
            return Err(RasterCreateError::BindingRecipeMismatch);
        }
        if pipeline.device_identity() != self.identity
            || uniform.device_identity() != self.identity
            || texture.device_identity() != self.identity
            || positions.device_identity() != self.identity
            || texture_coordinates.device_identity() != self.identity
        {
            return Err(RasterCreateError::ForeignDevice);
        }
        validate_raster_uniform_contract(
            RasterKernel::IndexedPositionFloat32x3CameraMaterial,
            uniform.descriptor().buffer.size,
            uniform.allowed_usage(),
        )?;
        validate_texture_pack_texture_desc(texture.descriptor().texture, texture.allowed_usage())
            .map_err(|_| RasterCreateError::BindingRecipeMismatch)?;
        let position_size = u64::from(vertex_count)
            .checked_mul(12)
            .ok_or(RasterCreateError::VertexStreamCountMismatch)?;
        let uv_size = u64::from(vertex_count)
            .checked_mul(8)
            .ok_or(RasterCreateError::VertexStreamCountMismatch)?;
        if vertex_count == 0 || positions.descriptor().buffer.size != position_size {
            return Err(RasterCreateError::InvalidPositionStreamRange);
        }
        if texture_coordinates.descriptor().buffer.size != uv_size {
            return Err(RasterCreateError::InvalidTextureCoordinateStreamRange);
        }
        if !positions.allowed_usage().contains(BufferUsageKind::Vertex)
            || !texture_coordinates
                .allowed_usage()
                .contains(BufferUsageKind::Vertex)
        {
            return Err(RasterCreateError::VertexUsageRequired);
        }
        let native = crate::imp::create_raster_uv_linear_clamp_texture_bindings(
            &self.inner,
            pipeline.native(),
            uniform.native(),
            texture.native(),
            positions.identity(),
            position_size,
            texture_coordinates.identity(),
            uv_size,
        )
        .map_err(RasterCreateError::NativeFailure)?;
        Ok(RasterUvLinearClampTextureBindings(Arc::new(
            RasterUvLinearClampTextureBindingsShared {
                _native: native,
                pipeline: pipeline.clone(),
                _uniform: uniform.lease(),
                _texture: texture.lease(),
                _positions: positions.lease(),
                _texture_coordinates: texture_coordinates.lease(),
                position_identity: positions.identity(),
                texture_coordinate_identity: texture_coordinates.identity(),
                vertex_count,
                device: self.identity,
            },
        )))
    }

    /// Creates the closed explicit-UV linear-clamp sRGB base-color binding.
    ///
    /// The encoded source bytes remain in the sRGB texture. Its conversion to
    /// linear happens in fixed native sampling, before filtering and material
    /// modulation; this API never performs a CPU-side conversion.
    #[allow(
        clippy::too_many_arguments,
        reason = "the closed recipe has independent uniform, texture, and two vertex role inputs"
    )]
    pub fn create_raster_uv_linear_clamp_srgb_texture_bindings(
        &self,
        pipeline: &RasterPipeline,
        uniform: &Buffer,
        texture: &Texture,
        positions: &Buffer,
        texture_coordinates: &Buffer,
        vertex_count: u32,
    ) -> Result<RasterUvLinearClampTextureBindings, RasterCreateError> {
        if !self.capabilities().rgba8_unorm_srgb_filterable {
            return Err(RasterCreateError::TextureFormatNotFilterable {
                format: TextureFormat::Rgba8UnormSrgb,
            });
        }
        if pipeline.kernel()
            != RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb
        {
            return Err(RasterCreateError::BindingRecipeMismatch);
        }
        if pipeline.device_identity() != self.identity
            || uniform.device_identity() != self.identity
            || texture.device_identity() != self.identity
            || positions.device_identity() != self.identity
            || texture_coordinates.device_identity() != self.identity
        {
            return Err(RasterCreateError::ForeignDevice);
        }
        validate_raster_uniform_contract(
            RasterKernel::IndexedPositionFloat32x3CameraMaterial,
            uniform.descriptor().buffer.size,
            uniform.allowed_usage(),
        )?;
        validate_raster_srgb_texture_desc(texture.descriptor().texture, texture.allowed_usage())?;
        let position_size = u64::from(vertex_count)
            .checked_mul(12)
            .ok_or(RasterCreateError::VertexStreamCountMismatch)?;
        let uv_size = u64::from(vertex_count)
            .checked_mul(8)
            .ok_or(RasterCreateError::VertexStreamCountMismatch)?;
        if vertex_count == 0 || positions.descriptor().buffer.size != position_size {
            return Err(RasterCreateError::InvalidPositionStreamRange);
        }
        if texture_coordinates.descriptor().buffer.size != uv_size {
            return Err(RasterCreateError::InvalidTextureCoordinateStreamRange);
        }
        if !positions.allowed_usage().contains(BufferUsageKind::Vertex)
            || !texture_coordinates
                .allowed_usage()
                .contains(BufferUsageKind::Vertex)
        {
            return Err(RasterCreateError::VertexUsageRequired);
        }
        let native = crate::imp::create_raster_uv_linear_clamp_srgb_texture_bindings(
            &self.inner,
            pipeline.native(),
            uniform.native(),
            texture.native(),
            positions.identity(),
            position_size,
            texture_coordinates.identity(),
            uv_size,
        )
        .map_err(RasterCreateError::NativeFailure)?;
        Ok(RasterUvLinearClampTextureBindings(Arc::new(
            RasterUvLinearClampTextureBindingsShared {
                _native: native,
                pipeline: pipeline.clone(),
                _uniform: uniform.lease(),
                _texture: texture.lease(),
                _positions: positions.lease(),
                _texture_coordinates: texture_coordinates.lease(),
                position_identity: positions.identity(),
                texture_coordinate_identity: texture_coordinates.identity(),
                vertex_count,
                device: self.identity,
            },
        )))
    }
}
