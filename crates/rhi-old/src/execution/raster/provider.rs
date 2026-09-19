//! RenderGraph object registry and lowering for the fixed raster profile.
//!
//! Registrations are device-affine recipes, not general bind-group
//! descriptions. Lowering validates the graph-declared resource shape before
//! constructing RHI bindings, so a graph identity cannot be reused with a
//! different pipeline, resource role, or device.

use super::*;
use crate::execution::helpers::require_device;
use crate::execution::provider::{compute_binding_error_kind, provider_error};

/// Registry for the fixed raster artifacts and the two fixed compute recipes
/// required by the 0.1.4 Raster→Compute→Copy vertical slice.
pub struct RasterObjectProvider {
    owner: Device,
    device: fluxel_rendergraph::DeviceIdentity,
    raster: HashMap<fluxel_rendergraph::RasterPipelineId, RasterPipeline>,
    compute: HashMap<fluxel_rendergraph::ComputePipelineId, ComputePipeline>,
    bindings: HashMap<
        fluxel_rendergraph::BindingSetId,
        (fluxel_rendergraph::ComputePipelineId, ComputePipeline),
    >,
    raster_bindings: HashMap<
        fluxel_rendergraph::BindingSetId,
        (fluxel_rendergraph::RasterPipelineId, RasterPipeline),
    >,
    raster_uv_bindings: HashMap<
        fluxel_rendergraph::BindingSetId,
        (
            fluxel_rendergraph::RasterPipelineId,
            RasterPipeline,
            Buffer,
            Buffer,
            u32,
        ),
    >,
    raster_uv_linear_clamp_bindings: HashMap<
        fluxel_rendergraph::BindingSetId,
        (
            fluxel_rendergraph::RasterPipelineId,
            RasterPipeline,
            Buffer,
            Buffer,
            u32,
        ),
    >,
    raster_uv_linear_clamp_srgb_bindings: HashMap<
        fluxel_rendergraph::BindingSetId,
        (
            fluxel_rendergraph::RasterPipelineId,
            RasterPipeline,
            Buffer,
            Buffer,
            u32,
        ),
    >,
    raster_normal_bindings: HashMap<
        fluxel_rendergraph::BindingSetId,
        (
            fluxel_rendergraph::RasterPipelineId,
            RasterPipeline,
            Buffer,
            Buffer,
            u32,
        ),
    >,
    raster_vertex_color_bindings: HashMap<
        fluxel_rendergraph::BindingSetId,
        (
            fluxel_rendergraph::RasterPipelineId,
            RasterPipeline,
            Buffer,
            Buffer,
            u32,
        ),
    >,
}

impl RasterObjectProvider {
    /// Creates an empty registry for one device-affine raster backend.
    pub fn new(device: &Device) -> Self {
        Self {
            owner: device.clone(),
            device: device.identity(),
            raster: HashMap::new(),
            compute: HashMap::new(),
            bindings: HashMap::new(),
            raster_bindings: HashMap::new(),
            raster_uv_bindings: HashMap::new(),
            raster_uv_linear_clamp_bindings: HashMap::new(),
            raster_uv_linear_clamp_srgb_bindings: HashMap::new(),
            raster_normal_bindings: HashMap::new(),
            raster_vertex_color_bindings: HashMap::new(),
        }
    }

    /// Registers one fixed raster pipeline under a graph identity.
    pub fn register_raster_pipeline(
        &mut self,
        id: fluxel_rendergraph::RasterPipelineId,
        pipeline: RasterPipeline,
    ) -> Result<(), NativeExecutionError> {
        require_device(
            pipeline.device_identity(),
            self.device,
            NativeExecutionError::ForeignResource,
        )?;
        self.raster.insert(id, pipeline);
        Ok(())
    }

    /// Registers one fixed compute pipeline under a graph identity.
    pub fn register_compute_pipeline(
        &mut self,
        id: fluxel_rendergraph::ComputePipelineId,
        pipeline: ComputePipeline,
    ) -> Result<(), NativeExecutionError> {
        require_device(
            pipeline.device_identity(),
            self.device,
            NativeExecutionError::ForeignResource,
        )?;
        self.compute.insert(id, pipeline);
        Ok(())
    }

    /// Registers the fixed binding recipe expected by a compute pipeline.
    pub fn register_bindings(
        &mut self,
        id: fluxel_rendergraph::BindingSetId,
        expected_pipeline: fluxel_rendergraph::ComputePipelineId,
    ) -> Result<(), NativeExecutionError> {
        let pipeline = self
            .compute
            .get(&expected_pipeline)
            .ok_or(NativeExecutionError::ComputeBindingMismatch)?;
        self.bindings
            .insert(id, (expected_pipeline, pipeline.clone()));
        Ok(())
    }

    /// Registers the closed frame-uniform recipe expected by a raster pipeline.
    pub fn register_raster_uniform_bindings(
        &mut self,
        id: fluxel_rendergraph::BindingSetId,
        expected_pipeline: fluxel_rendergraph::RasterPipelineId,
    ) -> Result<(), NativeExecutionError> {
        let pipeline = self
            .raster
            .get(&expected_pipeline)
            .ok_or(NativeExecutionError::RasterStateMismatch)?;
        if pipeline.kernel() != crate::RasterKernel::IndexedPositionFloat32x3CameraMaterial {
            return Err(NativeExecutionError::RasterStateMismatch);
        }
        self.raster_bindings
            .insert(id, (expected_pipeline, pipeline.clone()));
        Ok(())
    }

    /// Registers the closed uniform-plus-texture recipe expected by a textured raster pipeline.
    pub fn register_raster_textured_bindings(
        &mut self,
        id: fluxel_rendergraph::BindingSetId,
        expected_pipeline: fluxel_rendergraph::RasterPipelineId,
    ) -> Result<(), NativeExecutionError> {
        let pipeline = self
            .raster
            .get(&expected_pipeline)
            .ok_or(NativeExecutionError::RasterStateMismatch)?;
        if pipeline.kernel() != crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTexture {
            return Err(NativeExecutionError::RasterStateMismatch);
        }
        self.raster_bindings
            .insert(id, (expected_pipeline, pipeline.clone()));
        Ok(())
    }

    /// Registers the closed two-vertex-stream UV textured raster recipe.
    pub fn register_raster_uv_textured_bindings(
        &mut self,
        id: fluxel_rendergraph::BindingSetId,
        expected_pipeline: fluxel_rendergraph::RasterPipelineId,
        positions: &Buffer,
        texture_coordinates: &Buffer,
        vertex_count: u32,
    ) -> Result<(), NativeExecutionError> {
        let pipeline = self
            .raster
            .get(&expected_pipeline)
            .ok_or(NativeExecutionError::RasterStateMismatch)?;
        if pipeline.kernel() != crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUv
        {
            return Err(NativeExecutionError::RasterStateMismatch);
        }
        require_device(
            positions.device_identity(),
            self.device,
            NativeExecutionError::ForeignResource,
        )?;
        require_device(
            texture_coordinates.device_identity(),
            self.device,
            NativeExecutionError::ForeignResource,
        )?;
        self.raster_uv_bindings.insert(
            id,
            (
                expected_pipeline,
                pipeline.clone(),
                positions.clone(),
                texture_coordinates.clone(),
                vertex_count,
            ),
        );
        Ok(())
    }

    /// Registers the closed explicit-UV linear-clamp sampler recipe. The
    /// sampler remains internal to RHI; graph resources contain only the
    /// uniform, sampled texture, and two vertex streams.
    pub fn register_raster_uv_linear_clamp_textured_bindings(
        &mut self,
        id: fluxel_rendergraph::BindingSetId,
        expected_pipeline: fluxel_rendergraph::RasterPipelineId,
        positions: &Buffer,
        texture_coordinates: &Buffer,
        vertex_count: u32,
    ) -> Result<(), NativeExecutionError> {
        let pipeline = self
            .raster
            .get(&expected_pipeline)
            .ok_or(NativeExecutionError::RasterStateMismatch)?;
        if pipeline.kernel()
            != crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
        {
            return Err(NativeExecutionError::RasterStateMismatch);
        }
        require_device(
            positions.device_identity(),
            self.device,
            NativeExecutionError::ForeignResource,
        )?;
        require_device(
            texture_coordinates.device_identity(),
            self.device,
            NativeExecutionError::ForeignResource,
        )?;
        self.raster_uv_linear_clamp_bindings.insert(
            id,
            (
                expected_pipeline,
                pipeline.clone(),
                positions.clone(),
                texture_coordinates.clone(),
                vertex_count,
            ),
        );
        Ok(())
    }

    /// Registers the closed explicit-UV linear-clamp sRGB base-color recipe.
    pub fn register_raster_uv_linear_clamp_srgb_textured_bindings(
        &mut self,
        id: fluxel_rendergraph::BindingSetId,
        expected_pipeline: fluxel_rendergraph::RasterPipelineId,
        positions: &Buffer,
        texture_coordinates: &Buffer,
        vertex_count: u32,
    ) -> Result<(), NativeExecutionError> {
        let pipeline = self
            .raster
            .get(&expected_pipeline)
            .ok_or(NativeExecutionError::RasterStateMismatch)?;
        if pipeline.kernel()
            != crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb
        {
            return Err(NativeExecutionError::RasterStateMismatch);
        }
        require_device(
            positions.device_identity(),
            self.device,
            NativeExecutionError::ForeignResource,
        )?;
        require_device(
            texture_coordinates.device_identity(),
            self.device,
            NativeExecutionError::ForeignResource,
        )?;
        self.raster_uv_linear_clamp_srgb_bindings.insert(
            id,
            (
                expected_pipeline,
                pipeline.clone(),
                positions.clone(),
                texture_coordinates.clone(),
                vertex_count,
            ),
        );
        Ok(())
    }

    /// Registers the closed position-and-normal fixed Lambert recipe.
    pub fn register_raster_normal_bindings(
        &mut self,
        id: fluxel_rendergraph::BindingSetId,
        expected_pipeline: fluxel_rendergraph::RasterPipelineId,
        positions: &Buffer,
        normals: &Buffer,
        vertex_count: u32,
    ) -> Result<(), NativeExecutionError> {
        let pipeline = self
            .raster
            .get(&expected_pipeline)
            .ok_or(NativeExecutionError::RasterStateMismatch)?;
        if pipeline.kernel()
            != crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialNormalLambert
            || vertex_count == 0
        {
            return Err(NativeExecutionError::RasterStateMismatch);
        }
        require_device(
            positions.device_identity(),
            self.device,
            NativeExecutionError::ForeignResource,
        )?;
        require_device(
            normals.device_identity(),
            self.device,
            NativeExecutionError::ForeignResource,
        )?;
        self.raster_normal_bindings.insert(
            id,
            (
                expected_pipeline,
                pipeline.clone(),
                positions.clone(),
                normals.clone(),
                vertex_count,
            ),
        );
        Ok(())
    }

    /// Registers the closed position-and-RGBA8 vertex-color raster recipe.
    pub fn register_raster_vertex_color_bindings(
        &mut self,
        id: fluxel_rendergraph::BindingSetId,
        expected_pipeline: fluxel_rendergraph::RasterPipelineId,
        positions: &Buffer,
        colors: &Buffer,
        vertex_count: u32,
    ) -> Result<(), NativeExecutionError> {
        let pipeline = self
            .raster
            .get(&expected_pipeline)
            .ok_or(NativeExecutionError::RasterStateMismatch)?;
        if pipeline.kernel()
            != crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialVertexColor
            || vertex_count == 0
        {
            return Err(NativeExecutionError::RasterStateMismatch);
        }
        require_device(
            positions.device_identity(),
            self.device,
            NativeExecutionError::ForeignResource,
        )?;
        require_device(
            colors.device_identity(),
            self.device,
            NativeExecutionError::ForeignResource,
        )?;
        self.raster_vertex_color_bindings.insert(
            id,
            (
                expected_pipeline,
                pipeline.clone(),
                positions.clone(),
                colors.clone(),
                vertex_count,
            ),
        );
        Ok(())
    }
}

impl fluxel_rendergraph::RenderObjectProvider<RasterBackend> for RasterObjectProvider {
    fn raster_pipeline(
        &self,
        id: fluxel_rendergraph::RasterPipelineId,
    ) -> Result<
        fluxel_rendergraph::BoundRasterPipeline<RasterPipeline, ResourceLease>,
        fluxel_rendergraph::RecordingError,
    > {
        let pipeline = self.raster.get(&id).ok_or_else(|| {
            provider_error(
                fluxel_rendergraph::RecordingErrorKind::MissingFrameBinding,
                "unknown raster pipeline",
            )
        })?;
        if pipeline.device_identity() != self.device {
            return Err(provider_error(
                fluxel_rendergraph::RecordingErrorKind::IncompatibleBindingRecipe,
                "foreign raster pipeline",
            ));
        }
        Ok(fluxel_rendergraph::BoundRasterPipeline {
            device: self.device,
            physical: pipeline.clone(),
            lease: pipeline.lease().into(),
        })
    }

    fn compute_pipeline(
        &self,
        id: fluxel_rendergraph::ComputePipelineId,
    ) -> Result<
        fluxel_rendergraph::BoundComputePipeline<ComputePipeline, ResourceLease>,
        fluxel_rendergraph::RecordingError,
    > {
        let pipeline = self.compute.get(&id).ok_or_else(|| {
            provider_error(
                fluxel_rendergraph::RecordingErrorKind::MissingFrameBinding,
                "unknown compute pipeline",
            )
        })?;
        if pipeline.device_identity() != self.device {
            return Err(provider_error(
                fluxel_rendergraph::RecordingErrorKind::IncompatibleBindingRecipe,
                "foreign compute pipeline",
            ));
        }
        Ok(fluxel_rendergraph::BoundComputePipeline {
            device: self.device,
            physical: pipeline.clone(),
            lease: pipeline.lease().into(),
        })
    }

    fn bindings(
        &self,
        id: fluxel_rendergraph::BindingSetId,
        resources: &[fluxel_rendergraph::ResolvedBindingResource<'_, Texture, Buffer>],
        dynamic_offsets: &[u32],
    ) -> Result<
        fluxel_rendergraph::BoundBindings<RasterBindings, ResourceLease>,
        fluxel_rendergraph::RecordingError,
    > {
        if !dynamic_offsets.is_empty() {
            return Err(provider_error(
                fluxel_rendergraph::RecordingErrorKind::IncompatibleBindingRecipe,
                "fixed bindings do not accept dynamic offsets",
            ));
        }
        if let Some((_, pipeline, positions, colors, vertex_count)) =
            self.raster_vertex_color_bindings.get(&id)
        {
            let [
                fluxel_rendergraph::ResolvedBindingResource::Buffer {
                    physical: uniform,
                    range: BufferRange::Whole,
                    semantic:
                        fluxel_rendergraph::BindingResourceSemantic::BufferRead(
                            fluxel_rendergraph::BufferReadUse::Uniform,
                        ),
                },
            ] = resources
            else {
                return Err(provider_error(
                    fluxel_rendergraph::RecordingErrorKind::IncompatibleBindingRecipe,
                    "vertex-color raster binding requires one whole Uniform buffer",
                ));
            };
            let value = self
                .owner
                .create_raster_vertex_color_bindings(
                    pipeline,
                    uniform,
                    positions,
                    colors,
                    *vertex_count,
                )
                .map_err(|error| {
                    provider_error(
                        fluxel_rendergraph::RecordingErrorKind::IncompatibleBindingRecipe,
                        error.to_string(),
                    )
                })?;
            return Ok(fluxel_rendergraph::BoundBindings {
                device: self.device,
                lease: value.lease().into(),
                physical: RasterBindings::RasterVertexColor(value),
            });
        }
        if let Some((_, pipeline, positions, normals, vertex_count)) =
            self.raster_normal_bindings.get(&id)
        {
            let [
                fluxel_rendergraph::ResolvedBindingResource::Buffer {
                    physical: uniform,
                    range: BufferRange::Whole,
                    semantic:
                        fluxel_rendergraph::BindingResourceSemantic::BufferRead(
                            fluxel_rendergraph::BufferReadUse::Uniform,
                        ),
                },
            ] = resources
            else {
                return Err(provider_error(
                    fluxel_rendergraph::RecordingErrorKind::IncompatibleBindingRecipe,
                    "normal-Lambert raster binding requires one whole Uniform buffer",
                ));
            };
            let value = self
                .owner
                .create_raster_normal_bindings(pipeline, uniform, positions, normals, *vertex_count)
                .map_err(|error| {
                    provider_error(
                        fluxel_rendergraph::RecordingErrorKind::IncompatibleBindingRecipe,
                        error.to_string(),
                    )
                })?;
            return Ok(fluxel_rendergraph::BoundBindings {
                device: self.device,
                lease: value.lease().into(),
                physical: RasterBindings::RasterNormal(value),
            });
        }
        if let Some((_, pipeline, positions, texture_coordinates, vertex_count)) =
            self.raster_uv_linear_clamp_srgb_bindings.get(&id)
        {
            let [
                fluxel_rendergraph::ResolvedBindingResource::Buffer {
                    physical: uniform,
                    range: BufferRange::Whole,
                    semantic:
                        fluxel_rendergraph::BindingResourceSemantic::BufferRead(
                            fluxel_rendergraph::BufferReadUse::Uniform,
                        ),
                },
                fluxel_rendergraph::ResolvedBindingResource::Texture {
                    physical: texture,
                    range: fluxel_rendergraph::TextureRange::Whole,
                    semantic:
                        fluxel_rendergraph::BindingResourceSemantic::TextureRead(
                            fluxel_rendergraph::TextureReadUse::Sampled,
                        ),
                },
            ] = resources
            else {
                return Err(provider_error(
                    fluxel_rendergraph::RecordingErrorKind::IncompatibleBindingRecipe,
                    "UV linear-clamp sRGB raster binding requires whole Uniform then whole Sampled texture",
                ));
            };
            let value = self
                .owner
                .create_raster_uv_linear_clamp_srgb_texture_bindings(
                    pipeline,
                    uniform,
                    texture,
                    positions,
                    texture_coordinates,
                    *vertex_count,
                )
                .map_err(|error| {
                    provider_error(
                        fluxel_rendergraph::RecordingErrorKind::IncompatibleBindingRecipe,
                        error.to_string(),
                    )
                })?;
            return Ok(fluxel_rendergraph::BoundBindings {
                device: self.device,
                lease: value.lease().into(),
                physical: RasterBindings::RasterUvLinearClampSrgbTexture(value),
            });
        }
        if let Some((_, pipeline, positions, texture_coordinates, vertex_count)) =
            self.raster_uv_linear_clamp_bindings.get(&id)
        {
            let [
                fluxel_rendergraph::ResolvedBindingResource::Buffer {
                    physical: uniform,
                    range: BufferRange::Whole,
                    semantic:
                        fluxel_rendergraph::BindingResourceSemantic::BufferRead(
                            fluxel_rendergraph::BufferReadUse::Uniform,
                        ),
                },
                fluxel_rendergraph::ResolvedBindingResource::Texture {
                    physical: texture,
                    range: fluxel_rendergraph::TextureRange::Whole,
                    semantic:
                        fluxel_rendergraph::BindingResourceSemantic::TextureRead(
                            fluxel_rendergraph::TextureReadUse::Sampled,
                        ),
                },
            ] = resources
            else {
                return Err(provider_error(
                    fluxel_rendergraph::RecordingErrorKind::IncompatibleBindingRecipe,
                    "UV linear-clamp raster binding requires whole Uniform then whole Sampled texture",
                ));
            };
            let value = self
                .owner
                .create_raster_uv_linear_clamp_texture_bindings(
                    pipeline,
                    uniform,
                    texture,
                    positions,
                    texture_coordinates,
                    *vertex_count,
                )
                .map_err(|error| {
                    provider_error(
                        fluxel_rendergraph::RecordingErrorKind::IncompatibleBindingRecipe,
                        error.to_string(),
                    )
                })?;
            return Ok(fluxel_rendergraph::BoundBindings {
                device: self.device,
                lease: value.lease().into(),
                physical: RasterBindings::RasterUvLinearClampTexture(value),
            });
        }
        if let Some((_, pipeline, positions, texture_coordinates, vertex_count)) =
            self.raster_uv_bindings.get(&id)
        {
            let [
                fluxel_rendergraph::ResolvedBindingResource::Buffer {
                    physical: uniform,
                    range: BufferRange::Whole,
                    semantic:
                        fluxel_rendergraph::BindingResourceSemantic::BufferRead(
                            fluxel_rendergraph::BufferReadUse::Uniform,
                        ),
                },
                fluxel_rendergraph::ResolvedBindingResource::Texture {
                    physical: texture,
                    range: fluxel_rendergraph::TextureRange::Whole,
                    semantic:
                        fluxel_rendergraph::BindingResourceSemantic::TextureRead(
                            fluxel_rendergraph::TextureReadUse::Sampled,
                        ),
                },
            ] = resources
            else {
                return Err(provider_error(
                    fluxel_rendergraph::RecordingErrorKind::IncompatibleBindingRecipe,
                    "UV textured raster binding requires whole Uniform then whole Sampled texture",
                ));
            };
            let value = self
                .owner
                .create_raster_uv_texture_bindings(
                    pipeline,
                    uniform,
                    texture,
                    positions,
                    texture_coordinates,
                    *vertex_count,
                )
                .map_err(|error| {
                    provider_error(
                        fluxel_rendergraph::RecordingErrorKind::IncompatibleBindingRecipe,
                        error.to_string(),
                    )
                })?;
            return Ok(fluxel_rendergraph::BoundBindings {
                device: self.device,
                lease: value.lease().into(),
                physical: RasterBindings::RasterUvTexture(value),
            });
        }
        if let Some((_, pipeline)) = self.raster_bindings.get(&id) {
            if pipeline.kernel()
                == crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTexture
            {
                let [
                    fluxel_rendergraph::ResolvedBindingResource::Buffer {
                        physical: uniform,
                        range: BufferRange::Whole,
                        semantic:
                            fluxel_rendergraph::BindingResourceSemantic::BufferRead(
                                fluxel_rendergraph::BufferReadUse::Uniform,
                            ),
                    },
                    fluxel_rendergraph::ResolvedBindingResource::Texture {
                        physical: texture,
                        range: fluxel_rendergraph::TextureRange::Whole,
                        semantic:
                            fluxel_rendergraph::BindingResourceSemantic::TextureRead(
                                fluxel_rendergraph::TextureReadUse::Sampled,
                            ),
                    },
                ] = resources
                else {
                    return Err(provider_error(
                        fluxel_rendergraph::RecordingErrorKind::IncompatibleBindingRecipe,
                        "textured raster binding requires whole Uniform then whole Sampled texture",
                    ));
                };
                let value = self
                    .owner
                    .create_raster_texture_bindings(pipeline, uniform, texture)
                    .map_err(|error| {
                        provider_error(
                            fluxel_rendergraph::RecordingErrorKind::IncompatibleBindingRecipe,
                            error.to_string(),
                        )
                    })?;
                return Ok(fluxel_rendergraph::BoundBindings {
                    device: self.device,
                    lease: value.lease().into(),
                    physical: RasterBindings::RasterTexture(value),
                });
            }
            let [
                fluxel_rendergraph::ResolvedBindingResource::Buffer {
                    physical: buffer,
                    range: BufferRange::Whole,
                    semantic:
                        fluxel_rendergraph::BindingResourceSemantic::BufferRead(
                            fluxel_rendergraph::BufferReadUse::Uniform,
                        ),
                },
            ] = resources
            else {
                return Err(provider_error(
                    fluxel_rendergraph::RecordingErrorKind::IncompatibleBindingRecipe,
                    "camera/material raster binding requires one whole Uniform buffer",
                ));
            };
            let value = self
                .owner
                .create_raster_uniform_bindings(pipeline, buffer)
                .map_err(|error| {
                    provider_error(
                        fluxel_rendergraph::RecordingErrorKind::IncompatibleBindingRecipe,
                        error.to_string(),
                    )
                })?;
            return Ok(fluxel_rendergraph::BoundBindings {
                device: self.device,
                lease: value.lease().into(),
                physical: RasterBindings::RasterUniform(value),
            });
        }
        let (_, pipeline) = self.bindings.get(&id).ok_or_else(|| {
            provider_error(
                fluxel_rendergraph::RecordingErrorKind::MissingFrameBinding,
                "unknown fixed binding recipe",
            )
        })?;
        let bindings = match pipeline.kernel() {
            crate::ComputeKernel::TexturePackRgba8 => {
                let [
                    fluxel_rendergraph::ResolvedBindingResource::Texture {
                        physical: texture,
                        range: fluxel_rendergraph::TextureRange::Whole,
                        semantic:
                            fluxel_rendergraph::BindingResourceSemantic::TextureRead(
                                fluxel_rendergraph::TextureReadUse::Sampled,
                            ),
                    },
                    fluxel_rendergraph::ResolvedBindingResource::Buffer {
                        physical: buffer,
                        range,
                        semantic,
                    },
                ] = resources
                else {
                    return Err(provider_error(
                        fluxel_rendergraph::RecordingErrorKind::IncompatibleBindingRecipe,
                        "texture-pack requires a complete sampled texture and one RW storage buffer",
                    ));
                };
                if !matches!(
                    semantic,
                    fluxel_rendergraph::BindingResourceSemantic::BufferReadWrite(
                        fluxel_rendergraph::BufferReadWriteUse::Storage
                    ) | fluxel_rendergraph::BindingResourceSemantic::BufferWrite(
                        fluxel_rendergraph::BufferWriteUse::Storage
                    )
                ) {
                    return Err(provider_error(
                        fluxel_rendergraph::RecordingErrorKind::DeclaredUseMismatch,
                        "texture-pack requires Storage output",
                    ));
                }
                if texture.device_identity() != self.device
                    || buffer.device_identity() != self.device
                {
                    return Err(provider_error(
                        fluxel_rendergraph::RecordingErrorKind::IncompatibleBindingRecipe,
                        "texture-pack binding is foreign to provider device",
                    ));
                }
                let (offset, size) = match *range {
                    BufferRange::Whole => (0, buffer.descriptor().buffer.size),
                    BufferRange::Bytes { offset, size } => (offset, size),
                };
                let value = self
                    .owner
                    .create_texture_pack_bindings(pipeline, texture, buffer, offset, size)
                    .map_err(|error| {
                        provider_error(compute_binding_error_kind(&error), error.to_string())
                    })?;
                RasterBindings::TexturePack(value)
            }
            _ => {
                let [
                    fluxel_rendergraph::ResolvedBindingResource::Buffer {
                        physical: buffer,
                        range,
                        semantic:
                            fluxel_rendergraph::BindingResourceSemantic::BufferReadWrite(
                                fluxel_rendergraph::BufferReadWriteUse::Storage,
                            ),
                    },
                ] = resources
                else {
                    return Err(provider_error(
                        fluxel_rendergraph::RecordingErrorKind::IncompatibleBindingRecipe,
                        "fixed compute requires one RW storage buffer",
                    ));
                };
                if buffer.device_identity() != self.device {
                    return Err(provider_error(
                        fluxel_rendergraph::RecordingErrorKind::IncompatibleBindingRecipe,
                        "compute binding is foreign to provider device",
                    ));
                }
                let (offset, size) = match *range {
                    BufferRange::Whole => (0, buffer.descriptor().buffer.size),
                    BufferRange::Bytes { offset, size } => (offset, size),
                };
                let value = self
                    .owner
                    .create_compute_bindings(pipeline, buffer, offset, size)
                    .map_err(|error| {
                        provider_error(compute_binding_error_kind(&error), error.to_string())
                    })?;
                RasterBindings::Compute(value)
            }
        };
        let lease = match &bindings {
            RasterBindings::Compute(value) => value.lease().into(),
            RasterBindings::TexturePack(value) => value.lease().into(),
            RasterBindings::RasterUniform(value) => value.lease().into(),
            RasterBindings::RasterTexture(value) => value.lease().into(),
            RasterBindings::RasterUvTexture(value) => value.lease().into(),
            RasterBindings::RasterUvLinearClampTexture(value) => value.lease().into(),
            RasterBindings::RasterUvLinearClampSrgbTexture(value) => value.lease().into(),
            RasterBindings::RasterNormal(value) => value.lease().into(),
            RasterBindings::RasterVertexColor(value) => value.lease().into(),
        };
        Ok(fluxel_rendergraph::BoundBindings {
            device: self.device,
            physical: bindings,
            lease,
        })
    }
}
