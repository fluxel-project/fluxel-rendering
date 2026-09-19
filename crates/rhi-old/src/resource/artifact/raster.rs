//! Portable identities and WGSL definitions for fixed raster artifacts.
use super::super::*;
impl RasterKernel {
    /// Returns the fixed vertex entry point.
    pub const fn vertex_entry_point(self) -> &'static str {
        match self {
            Self::Triangle => "triangle_vertex",
            Self::IndexedPositionColor => "position_color_vertex",
            Self::IndexedPositionFloat32x3 => "position_f32x3_vertex",
            Self::IndexedPositionFloat32x3CameraMaterial => "camera_material_vertex",
            Self::IndexedPositionFloat32x3CameraMaterialTexture => "camera_material_texture_vertex",
            Self::IndexedPositionFloat32x3CameraMaterialTextureUv => {
                "camera_material_texture_uv_vertex"
            }
            Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp => {
                "camera_material_texture_uv_linear_clamp_vertex"
            }
            Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb => {
                "camera_material_texture_uv_linear_clamp_srgb_vertex"
            }
            Self::IndexedPositionFloat32x3CameraMaterialNormalLambert => {
                "camera_material_normal_lambert_vertex"
            }
            Self::IndexedPositionFloat32x3CameraMaterialVertexColor => {
                "camera_material_vertex_color_vertex"
            }
        }
    }

    /// Returns the shared fixed fragment entry point.
    pub const fn fragment_entry_point(self) -> &'static str {
        match self {
            Self::IndexedPositionFloat32x3CameraMaterialTexture
            | Self::IndexedPositionFloat32x3CameraMaterialTextureUv => "texture_color_fragment",
            Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp => {
                "linear_clamp_texture_color_fragment"
            }
            Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb => {
                "linear_clamp_srgb_texture_color_fragment"
            }
            Self::IndexedPositionFloat32x3CameraMaterialNormalLambert => "normal_lambert_fragment",
            Self::IndexedPositionFloat32x3CameraMaterialVertexColor => "vertex_color_fragment",
            _ => "color_fragment",
        }
    }

    /// Returns the target format supported by every fixed artifact.
    pub const fn target_format(self) -> TextureFormat {
        TextureFormat::Rgba8Unorm
    }

    /// Returns the byte stride of the fixed vertex recipe.
    pub const fn vertex_stride(self) -> u32 {
        match self {
            Self::Triangle => 0,
            Self::IndexedPositionColor => 12,
            Self::IndexedPositionFloat32x3 => 12,
            Self::IndexedPositionFloat32x3CameraMaterial => 12,
            Self::IndexedPositionFloat32x3CameraMaterialTexture => 12,
            Self::IndexedPositionFloat32x3CameraMaterialTextureUv => 12,
            Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp => 12,
            Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb => 12,
            Self::IndexedPositionFloat32x3CameraMaterialNormalLambert => 12,
            Self::IndexedPositionFloat32x3CameraMaterialVertexColor => 12,
        }
    }

    /// Returns the required index element format for this closed recipe.
    pub const fn index_format(self) -> Option<IndexFormat> {
        match self {
            Self::Triangle => None,
            Self::IndexedPositionColor => Some(IndexFormat::Uint16),
            Self::IndexedPositionFloat32x3 => Some(IndexFormat::Uint32),
            Self::IndexedPositionFloat32x3CameraMaterial => Some(IndexFormat::Uint32),
            Self::IndexedPositionFloat32x3CameraMaterialTexture => Some(IndexFormat::Uint32),
            Self::IndexedPositionFloat32x3CameraMaterialTextureUv => Some(IndexFormat::Uint32),
            Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp => {
                Some(IndexFormat::Uint32)
            }
            Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb => {
                Some(IndexFormat::Uint32)
            }
            Self::IndexedPositionFloat32x3CameraMaterialNormalLambert => Some(IndexFormat::Uint32),
            Self::IndexedPositionFloat32x3CameraMaterialVertexColor => Some(IndexFormat::Uint32),
        }
    }

    /// Returns the complete non-configurable vertex layout.
    pub const fn vertex_layout(self) -> RasterVertexLayout {
        match self {
            Self::Triangle => RasterVertexLayout::None,
            Self::IndexedPositionColor => RasterVertexLayout::PositionFloat32x2ColorUnorm8x4,
            Self::IndexedPositionFloat32x3 => RasterVertexLayout::PositionFloat32x3,
            Self::IndexedPositionFloat32x3CameraMaterial => RasterVertexLayout::PositionFloat32x3,
            Self::IndexedPositionFloat32x3CameraMaterialTexture => {
                RasterVertexLayout::PositionFloat32x3
            }
            Self::IndexedPositionFloat32x3CameraMaterialTextureUv => {
                RasterVertexLayout::PositionFloat32x3AndTextureCoordinateFloat32x2
            }
            Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp => {
                RasterVertexLayout::PositionFloat32x3AndTextureCoordinateFloat32x2
            }
            Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb => {
                RasterVertexLayout::PositionFloat32x3AndTextureCoordinateFloat32x2
            }
            Self::IndexedPositionFloat32x3CameraMaterialNormalLambert => {
                RasterVertexLayout::PositionFloat32x3AndNormalFloat32x3
            }
            Self::IndexedPositionFloat32x3CameraMaterialVertexColor => {
                RasterVertexLayout::PositionFloat32x3AndColorUnorm8x4
            }
        }
    }

    /// Returns the portable identity recorded by conformance fixtures.
    pub const fn portable_identity(self) -> RasterArtifactIdentity {
        RasterArtifactIdentity {
            module_source_hash: fnv1a64(self.wgsl_source().as_bytes()),
            vertex_entry_point: self.vertex_entry_point(),
            fragment_entry_point: self.fragment_entry_point(),
            target_format: self.target_format(),
            vertex_layout: self.vertex_layout(),
            vertex_stride: self.vertex_stride(),
            index_format: self.index_format(),
            binding_count: if self.has_frame_uniform() {
                if matches!(
                    self,
                    Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
                        | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb
                ) {
                    3
                } else if matches!(
                    self,
                    Self::IndexedPositionFloat32x3CameraMaterialTexture
                        | Self::IndexedPositionFloat32x3CameraMaterialTextureUv
                ) {
                    2
                } else {
                    1
                }
            } else {
                0
            },
            uniform_binding_size: if self.has_frame_uniform() { 80 } else { 0 },
            uniform_binding: if self.has_frame_uniform() {
                Some(0)
            } else {
                None
            },
            uniform_visibility: if self.has_frame_uniform() {
                Some(RasterBindingVisibility::VertexFragment)
            } else {
                None
            },
            binding_recipe_version: if self.has_frame_uniform() { 1 } else { 0 },
            texture_binding_recipe_version: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialTexture
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUv
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb
            ) {
                1
            } else {
                0
            },
            texture_binding: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialTexture
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUv
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb
            ) {
                Some(1)
            } else {
                None
            },
            texture_dimension: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialTexture
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUv
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb
            ) {
                Some(TextureDimension::D2)
            } else {
                None
            },
            texture_format: match self {
                Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb => {
                    Some(TextureFormat::Rgba8UnormSrgb)
                }
                Self::IndexedPositionFloat32x3CameraMaterialTexture
                | Self::IndexedPositionFloat32x3CameraMaterialTextureUv
                | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp => {
                    Some(TextureFormat::Rgba8Unorm)
                }
                _ => None,
            },
            texture_visibility: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialTexture
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUv
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb
            ) {
                Some(RasterBindingVisibility::Fragment)
            } else {
                None
            },
            texture_sample_type: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialTexture
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUv
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb
            ) {
                Some(RasterTextureSampleType::Float)
            } else {
                None
            },
            texture_mip_level: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialTexture
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUv
            ) {
                Some(0)
            } else {
                None
            },
            texture_mapping_recipe_version: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialTexture
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUv
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb
            ) {
                if matches!(
                    self,
                    Self::IndexedPositionFloat32x3CameraMaterialTextureUv
                        | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
                        | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb
                ) {
                    2
                } else {
                    1
                }
            } else {
                0
            },
            texture_coordinate_vertex_layout: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialTextureUv
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb
            ) {
                Some(RasterVertexLayout::PositionFloat32x3AndTextureCoordinateFloat32x2)
            } else {
                None
            },
            texture_coordinate_vertex_stride: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialTextureUv
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb
            ) {
                Some(8)
            } else {
                None
            },
            texture_coordinate_shader_location: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialTextureUv
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb
            ) {
                Some(1)
            } else {
                None
            },
            sampler_binding: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb
            ) {
                Some(2)
            } else {
                None
            },
            sampler_binding_recipe_version: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb
            ) {
                1
            } else {
                0
            },
            sampler_binding_type: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb
            ) {
                Some(RasterSamplerBindingType::Filtering)
            } else {
                None
            },
            sampler_min_filter: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb
            ) {
                Some(RasterSamplerFilter::Linear)
            } else {
                None
            },
            sampler_mag_filter: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb
            ) {
                Some(RasterSamplerFilter::Linear)
            } else {
                None
            },
            sampler_mipmap_filter: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb
            ) {
                Some(RasterSamplerFilter::Nearest)
            } else {
                None
            },
            sampler_address_mode_u: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb
            ) {
                Some(RasterSamplerAddressMode::ClampToEdge)
            } else {
                None
            },
            sampler_address_mode_v: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb
            ) {
                Some(RasterSamplerAddressMode::ClampToEdge)
            } else {
                None
            },
            sampler_address_mode_w: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb
            ) {
                Some(RasterSamplerAddressMode::ClampToEdge)
            } else {
                None
            },
            sampler_explicit_mip_level: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb
            ) {
                Some(0)
            } else {
                None
            },
            lighting_recipe_version: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialNormalLambert
            ) {
                1
            } else {
                0
            },
            normal_interpolation: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialNormalLambert
            ) {
                Some(RasterNormalInterpolation::PerspectiveCenter)
            } else {
                None
            },
            normal_normalization: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialNormalLambert
            ) {
                Some(RasterNormalNormalization::ExactPositiveUnitAfterInterpolationZeroLambert)
            } else {
                None
            },
            lighting_space: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialNormalLambert
            ) {
                Some(RasterLightingSpace::Object)
            } else {
                None
            },
            fixed_light_direction: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialNormalLambert
            ) {
                Some(RasterFixedLightDirection::PositiveZ)
            } else {
                None
            },
            lighting_model: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialNormalLambert
            ) {
                Some(RasterLightingModel::LambertRgbAlphaPassthrough)
            } else {
                None
            },
            position_vertex_slot: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialNormalLambert
            ) {
                Some(0)
            } else {
                None
            },
            position_shader_location: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialNormalLambert
            ) {
                Some(0)
            } else {
                None
            },
            normal_vertex_slot: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialNormalLambert
            ) {
                Some(1)
            } else {
                None
            },
            normal_vertex_stride: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialNormalLambert
            ) {
                Some(12)
            } else {
                None
            },
            normal_shader_location: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialNormalLambert
            ) {
                Some(1)
            } else {
                None
            },
            vertex_color_recipe_version: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialVertexColor
            ) {
                1
            } else {
                0
            },
            vertex_color_position_slot: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialVertexColor
            ) {
                Some(0)
            } else {
                None
            },
            vertex_color_position_shader_location: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialVertexColor
            ) {
                Some(0)
            } else {
                None
            },
            vertex_color_slot: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialVertexColor
            ) {
                Some(1)
            } else {
                None
            },
            vertex_color_stride: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialVertexColor
            ) {
                Some(4)
            } else {
                None
            },
            vertex_color_shader_location: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialVertexColor
            ) {
                Some(1)
            } else {
                None
            },
            recipe_version: 1,
        }
    }

    pub(crate) const fn wgsl_source(self) -> &'static str {
        match self {
            Self::Triangle => {
                "struct VertexOutput {\n\
         @builtin(position) position: vec4<f32>,\n\
         @location(0) color: vec4<f32>,\n\
         };\n\
         @vertex fn triangle_vertex(@builtin(vertex_index) index: u32) -> VertexOutput {\n\
             if (index == 0u) {\n\
                 return VertexOutput(vec4(-0.70, -0.60, 0.0, 1.0), vec4(64.0 / 255.0, 160.0 / 255.0, 1.0, 1.0));\n\
             } else if (index == 1u) {\n\
                 return VertexOutput(vec4(0.70, -0.60, 0.0, 1.0), vec4(64.0 / 255.0, 160.0 / 255.0, 1.0, 1.0));\n\
             } else {\n\
                 return VertexOutput(vec4(0.0, 0.70, 0.0, 1.0), vec4(64.0 / 255.0, 160.0 / 255.0, 1.0, 1.0));\n\
             }\n\
         }\n\
         @fragment fn color_fragment(input: VertexOutput) -> @location(0) vec4<f32> { return input.color; }"
            }
            Self::IndexedPositionColor => {
                "struct VertexInput {\n\
         @location(0) position: vec2<f32>,\n\
         @location(1) color: vec4<f32>,\n\
         };\n\
         struct VertexOutput {\n\
         @builtin(position) position: vec4<f32>,\n\
         @location(0) color: vec4<f32>,\n\
         };\n\
         @vertex fn position_color_vertex(input: VertexInput) -> VertexOutput {\n\
             return VertexOutput(vec4(input.position, 0.0, 1.0), input.color);\n\
         }\n\
         @fragment fn color_fragment(input: VertexOutput) -> @location(0) vec4<f32> { return input.color; }"
            }
            Self::IndexedPositionFloat32x3 => {
                "struct VertexInput {\n\
         @location(0) position: vec3<f32>,\n\
         };\n\
         struct VertexOutput {\n\
         @builtin(position) position: vec4<f32>,\n\
         };\n\
         @vertex fn position_f32x3_vertex(input: VertexInput) -> VertexOutput {\n\
             return VertexOutput(vec4(input.position, 1.0));\n\
         }\n\
         @fragment fn color_fragment(input: VertexOutput) -> @location(0) vec4<f32> {\n\
             _ = input;\n\
             return vec4(48.0 / 255.0, 176.0 / 255.0, 112.0 / 255.0, 1.0);\n\
         }"
            }
            Self::IndexedPositionFloat32x3CameraMaterial => {
                "struct FrameUniforms {\n\
         view_projection: mat4x4<f32>,\n\
         base_color: vec4<f32>,\n\
         };\n\
         @group(0) @binding(0) var<uniform> frame: FrameUniforms;\n\
         struct VertexInput { @location(0) position: vec3<f32>, };\n\
         struct VertexOutput { @builtin(position) position: vec4<f32>, };\n\
         @vertex fn camera_material_vertex(input: VertexInput) -> VertexOutput {\n\
             return VertexOutput(frame.view_projection * vec4(input.position, 1.0));\n\
         }\n\
         @fragment fn color_fragment(input: VertexOutput) -> @location(0) vec4<f32> {\n\
             _ = input; return frame.base_color;\n\
         }"
            }
            Self::IndexedPositionFloat32x3CameraMaterialTexture => {
                "struct FrameUniforms { view_projection: mat4x4<f32>, base_color: vec4<f32>, };\n\
         @group(0) @binding(0) var<uniform> frame: FrameUniforms;\n\
         @group(0) @binding(1) var tex: texture_2d<f32>;\n\
         struct VertexInput { @location(0) position: vec3<f32>, };\n\
         struct VertexOutput { @builtin(position) position: vec4<f32>, @location(0) @interpolate(perspective, center) uv: vec2<f32>, };\n\
         @vertex fn camera_material_texture_vertex(input: VertexInput) -> VertexOutput { return VertexOutput(frame.view_projection * vec4(input.position, 1.0), input.position.xy * vec2(0.5, -0.5) + vec2(0.5)); }\n\
         @fragment fn texture_color_fragment(input: VertexOutput) -> @location(0) vec4<f32> { let dimensions = textureDimensions(tex, 0); let texel = min(vec2<u32>(floor(clamp(input.uv, vec2<f32>(0.0), vec2<f32>(1.0)) * vec2<f32>(dimensions))), dimensions - vec2<u32>(1)); return frame.base_color * textureLoad(tex, vec2<i32>(texel), 0); }"
            }
            Self::IndexedPositionFloat32x3CameraMaterialTextureUv => {
                "struct FrameUniforms { view_projection: mat4x4<f32>, base_color: vec4<f32>, };\n\
         @group(0) @binding(0) var<uniform> frame: FrameUniforms;\n\
         @group(0) @binding(1) var tex: texture_2d<f32>;\n\
         struct VertexInput { @location(0) position: vec3<f32>, @location(1) uv: vec2<f32>, };\n\
         struct VertexOutput { @builtin(position) position: vec4<f32>, @location(0) @interpolate(perspective, center) uv: vec2<f32>, };\n\
         @vertex fn camera_material_texture_uv_vertex(input: VertexInput) -> VertexOutput { return VertexOutput(frame.view_projection * vec4(input.position, 1.0), input.uv); }\n\
         @fragment fn texture_color_fragment(input: VertexOutput) -> @location(0) vec4<f32> { let dimensions = textureDimensions(tex, 0); let texel = min(vec2<u32>(floor(clamp(input.uv, vec2<f32>(0.0), vec2<f32>(1.0)) * vec2<f32>(dimensions))), dimensions - vec2<u32>(1)); return frame.base_color * textureLoad(tex, vec2<i32>(texel), 0); }"
            }
            Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp => {
                "struct FrameUniforms { view_projection: mat4x4<f32>, base_color: vec4<f32>, };\n\
         @group(0) @binding(0) var<uniform> frame: FrameUniforms;\n\
         @group(0) @binding(1) var tex: texture_2d<f32>;\n\
         @group(0) @binding(2) var linear_clamp_sampler: sampler;\n\
         struct VertexInput { @location(0) position: vec3<f32>, @location(1) uv: vec2<f32>, };\n\
         struct VertexOutput { @builtin(position) position: vec4<f32>, @location(0) @interpolate(perspective, center) uv: vec2<f32>, };\n\
         @vertex fn camera_material_texture_uv_linear_clamp_vertex(input: VertexInput) -> VertexOutput { return VertexOutput(frame.view_projection * vec4(input.position, 1.0), input.uv); }\n\
         @fragment fn linear_clamp_texture_color_fragment(input: VertexOutput) -> @location(0) vec4<f32> { return frame.base_color * textureSampleLevel(tex, linear_clamp_sampler, input.uv, 0.0); }"
            }
            Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb => {
                "struct FrameUniforms { view_projection: mat4x4<f32>, base_color: vec4<f32>, };\n\
         @group(0) @binding(0) var<uniform> frame: FrameUniforms;\n\
         @group(0) @binding(1) var tex: texture_2d<f32>;\n\
         @group(0) @binding(2) var linear_clamp_sampler: sampler;\n\
         struct VertexInput { @location(0) position: vec3<f32>, @location(1) uv: vec2<f32>, };\n\
         struct VertexOutput { @builtin(position) position: vec4<f32>, @location(0) @interpolate(perspective, center) uv: vec2<f32>, };\n\
         @vertex fn camera_material_texture_uv_linear_clamp_srgb_vertex(input: VertexInput) -> VertexOutput { return VertexOutput(frame.view_projection * vec4(input.position, 1.0), input.uv); }\n\
         @fragment fn linear_clamp_srgb_texture_color_fragment(input: VertexOutput) -> @location(0) vec4<f32> { return frame.base_color * textureSampleLevel(tex, linear_clamp_sampler, input.uv, 0.0); }"
            }
            Self::IndexedPositionFloat32x3CameraMaterialNormalLambert => {
                "struct FrameUniforms { view_projection: mat4x4<f32>, base_color: vec4<f32>, };\n\
         @group(0) @binding(0) var<uniform> frame: FrameUniforms;\n\
         struct VertexInput { @location(0) position: vec3<f32>, @location(1) normal: vec3<f32>, };\n\
         struct VertexOutput { @builtin(position) position: vec4<f32>, @location(0) @interpolate(perspective, center) normal: vec3<f32>, };\n\
         @vertex fn camera_material_normal_lambert_vertex(input: VertexInput) -> VertexOutput { return VertexOutput(frame.view_projection * vec4(input.position, 1.0), input.normal); }\n\
         @fragment fn normal_lambert_fragment(input: VertexOutput) -> @location(0) vec4<f32> { let len2 = dot(input.normal, input.normal); if (len2 > 0.0) { let lambert = max(dot(input.normal * inverseSqrt(len2), vec3<f32>(0.0, 0.0, 1.0)), 0.0); return vec4<f32>(frame.base_color.rgb * lambert, frame.base_color.a); } return vec4<f32>(vec3<f32>(0.0), frame.base_color.a); }"
            }
            Self::IndexedPositionFloat32x3CameraMaterialVertexColor => {
                "struct FrameUniforms { view_projection: mat4x4<f32>, base_color: vec4<f32>, };\n\
         @group(0) @binding(0) var<uniform> frame: FrameUniforms;\n\
         struct VertexInput { @location(0) position: vec3<f32>, @location(1) color: vec4<f32>, };\n\
         struct VertexOutput { @builtin(position) position: vec4<f32>, @location(0) @interpolate(perspective, center) color: vec4<f32>, };\n\
         @vertex fn camera_material_vertex_color_vertex(input: VertexInput) -> VertexOutput { return VertexOutput(frame.view_projection * vec4(input.position, 1.0), input.color); }\n\
         @fragment fn vertex_color_fragment(input: VertexOutput) -> @location(0) vec4<f32> { return input.color * frame.base_color; }"
            }
        }
    }

    const fn has_frame_uniform(self) -> bool {
        matches!(
            self,
            Self::IndexedPositionFloat32x3CameraMaterial
                | Self::IndexedPositionFloat32x3CameraMaterialTexture
                | Self::IndexedPositionFloat32x3CameraMaterialTextureUv
                | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
                | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb
                | Self::IndexedPositionFloat32x3CameraMaterialNormalLambert
                | Self::IndexedPositionFloat32x3CameraMaterialVertexColor
        )
    }
}
