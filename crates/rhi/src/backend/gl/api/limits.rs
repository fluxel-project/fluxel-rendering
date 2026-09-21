//! Immutable numerical facts queried from one GL-family context.

use super::GlFamilyProfile;

/// A finite floating point value retained as its exact IEEE-754 bits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlFiniteF32(u32);
impl GlFiniteF32 {
    /// Rejects NaN and infinities while preserving every bit of a finite result.
    pub(crate) fn new(value: f32) -> Option<Self> {
        value.is_finite().then_some(Self(value.to_bits()))
    }
    /// Returns the queried bits.
    pub(crate) const fn bits(self) -> u32 {
        self.0
    }
    /// Returns the finite queried value.
    pub(crate) const fn get(self) -> f32 {
        f32::from_bits(self.0)
    }
}

/// All numerical inputs used by the profile and optional-domain resolver.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlLimits {
    /// Texture and attachment limits.
    pub max_texture_size: u32,
    pub max_3d_texture_size: u32,
    pub max_array_texture_layers: u32,
    pub max_cube_map_texture_size: u32,
    pub max_renderbuffer_size: u32,
    pub max_color_attachments: u32,
    pub max_draw_buffers: u32,
    /// Vertex, viewport and texture-unit limits.
    pub max_vertex_attributes: u32,
    pub max_viewport_dimensions: [u32; 2],
    pub max_viewports: u32,
    pub max_vertex_texture_image_units: u32,
    pub max_fragment_texture_image_units: u32,
    pub max_combined_texture_image_units: u32,
    /// UBO binding, range, alignment and stage limits.
    pub max_uniform_buffer_bindings: u32,
    pub max_uniform_block_size: u64,
    pub uniform_buffer_offset_alignment: u64,
    pub max_vertex_uniform_blocks: u32,
    pub max_fragment_uniform_blocks: u32,
    pub max_compute_uniform_blocks: u32,
    pub max_combined_uniform_blocks: u32,
    /// SSBO binding, range, alignment and stage limits.
    pub max_storage_buffer_bindings: u32,
    pub max_storage_block_size: u64,
    pub storage_buffer_offset_alignment: u64,
    pub max_vertex_storage_blocks: u32,
    pub max_fragment_storage_blocks: u32,
    pub max_compute_storage_blocks: u32,
    pub max_combined_storage_blocks: u32,
    /// Image-unit limits, independent from storage-buffer limits.
    pub max_image_units: u32,
    pub max_combined_image_units: u32,
    /// Sample and compute limits.
    pub max_samples: u32,
    pub max_color_texture_samples: u32,
    pub max_depth_texture_samples: u32,
    pub max_integer_samples: u32,
    pub max_compute_work_group_count: [u32; 3],
    pub max_compute_work_group_size: [u32; 3],
    pub max_compute_work_group_invocations: u32,
    /// Views one attachment may serve in a single pass; 0 means not queried.
    ///
    /// This is the numeric half of the multiview capability. WebGPU's
    /// `maxMultiviewViewCount` defaults to one view, which is the plain
    /// single-view attachment every profile already supports; a context that
    /// never answered the query records 0 here and therefore satisfies no
    /// multiview floor at all.
    pub max_multiview_view_count: u32,
    /// Only count/multi-draw has a count limit; single indirect commands do not.
    pub max_multi_draw_indirect_count: Option<u32>,
    /// Query and anisotropy facts.
    pub query_counter_bits: u32,
    pub max_texture_anisotropy: Option<GlFiniteF32>,
}

impl GlLimits {
    /// Explicit unavailable facts. They satisfy neither profile floors nor optional domains.
    pub const fn unavailable() -> Self {
        Self {
            max_texture_size: 0,
            max_3d_texture_size: 0,
            max_array_texture_layers: 0,
            max_cube_map_texture_size: 0,
            max_renderbuffer_size: 0,
            max_color_attachments: 0,
            max_draw_buffers: 0,
            max_vertex_attributes: 0,
            max_viewport_dimensions: [0; 2],
            max_viewports: 0,
            max_vertex_texture_image_units: 0,
            max_fragment_texture_image_units: 0,
            max_combined_texture_image_units: 0,
            max_uniform_buffer_bindings: 0,
            max_uniform_block_size: 0,
            uniform_buffer_offset_alignment: 0,
            max_vertex_uniform_blocks: 0,
            max_fragment_uniform_blocks: 0,
            max_compute_uniform_blocks: 0,
            max_combined_uniform_blocks: 0,
            max_storage_buffer_bindings: 0,
            max_storage_block_size: 0,
            storage_buffer_offset_alignment: 0,
            max_vertex_storage_blocks: 0,
            max_fragment_storage_blocks: 0,
            max_compute_storage_blocks: 0,
            max_combined_storage_blocks: 0,
            max_image_units: 0,
            max_combined_image_units: 0,
            max_samples: 0,
            max_color_texture_samples: 0,
            max_depth_texture_samples: 0,
            max_integer_samples: 0,
            max_compute_work_group_count: [0; 3],
            max_compute_work_group_size: [0; 3],
            max_compute_work_group_invocations: 0,
            max_multiview_view_count: 0,
            max_multi_draw_indirect_count: None,
            query_counter_bits: 0,
            max_texture_anisotropy: None,
        }
    }
    /// Validates the exact core profile's raster/resource floor.
    ///
    /// Everything checked here is a capacity the profile's fixed artifacts need.
    /// The two offset alignments are deliberately **not** checked, and the note
    /// on `requirements` says why.
    pub(crate) fn validate_profile_minimums(
        &self,
        profile: GlFamilyProfile,
    ) -> Result<(), GlLimitViolation> {
        let f = GlProfileMinimums::for_profile(profile);
        for (name, actual, required) in f.requirements(self) {
            if actual < required {
                return Err(GlLimitViolation {
                    name,
                    actual,
                    required,
                });
            }
        }
        Ok(())
    }
    /// Returns whether compute workgroup limits are complete.
    pub(crate) const fn supports_compute(&self) -> bool {
        self.max_compute_work_group_count[0] != 0
            && self.max_compute_work_group_count[1] != 0
            && self.max_compute_work_group_count[2] != 0
            && self.max_compute_work_group_size[0] != 0
            && self.max_compute_work_group_size[1] != 0
            && self.max_compute_work_group_size[2] != 0
            && self.max_compute_work_group_invocations != 0
    }
    /// Returns whether SSBO limits are complete.
    pub(crate) const fn supports_storage_buffers(&self) -> bool {
        self.max_storage_buffer_bindings != 0
            && self.max_storage_block_size != 0
            && self.storage_buffer_offset_alignment != 0
    }
    /// Returns whether image-unit limits are complete; exact format facts remain required.
    pub(crate) const fn supports_storage_images(&self) -> bool {
        self.max_image_units != 0 && self.max_combined_image_units != 0
    }
    /// Single indirect support is established by core/extension evidence and a real probe.
    pub(crate) const fn supports_single_indirect(&self) -> bool {
        true
    }
    /// Multi-draw/count needs a specifically queried count limit.
    ///
    /// That limit is a recorded fact rather than a driver property on this
    /// family: no GL or WebGL2 context exposes a portable query for how many
    /// draws one multi-draw command may issue, so every real route records
    /// `None` and this answers false there. The mock route is the only one that
    /// records a count, which is what makes this the domain's narrowing decision
    /// rather than a note beside it (plan P2-15).
    pub(crate) const fn supports_multi_draw_indirect(&self) -> bool {
        matches!(self.max_multi_draw_indirect_count, Some(value) if value != 0)
    }
    /// Multiview needs a queried view count of at least two views.
    ///
    /// One view is the plain single-view attachment every profile has without
    /// multiview, so a queried 1 or an unqueried 0 both fail here.
    pub(crate) const fn supports_multiview(&self) -> bool {
        self.max_multiview_view_count >= 2
    }
}

/// A failed portable core-limit comparison.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlLimitViolation {
    /// Stable limit name.
    pub name: &'static str,
    /// Observed value.
    pub actual: u64,
    /// Required floor.
    pub required: u64,
}

#[derive(Clone, Copy, Debug)]
struct GlProfileMinimums {
    texture: u64,
    three_d: u64,
    layers: u64,
    cube: u64,
    renderbuffer: u64,
    attachments: u64,
    draw_buffers: u64,
    viewport: u64,
    texture_units: u64,
    uniform_bindings: u64,
    samples: u64,
}
impl GlProfileMinimums {
    const fn for_profile(profile: GlFamilyProfile) -> Self {
        match profile {
            GlFamilyProfile::Desktop { .. } => Self {
                texture: 16_384,
                three_d: 2_048,
                layers: 2_048,
                cube: 16_384,
                renderbuffer: 16_384,
                attachments: 8,
                draw_buffers: 8,
                viewport: 16_384,
                texture_units: 16,
                uniform_bindings: 36,
                samples: 4,
            },
            GlFamilyProfile::Embedded { .. } | GlFamilyProfile::WebGl2 => Self {
                texture: 2_048,
                three_d: 256,
                layers: 256,
                cube: 2_048,
                renderbuffer: 4_096,
                attachments: 4,
                draw_buffers: 4,
                viewport: 2_048,
                texture_units: 16,
                uniform_bindings: 24,
                samples: 4,
            },
        }
    }
    /// The capacity floors this profile's fixed artifacts depend on.
    ///
    /// Offset alignment is deliberately absent, and it is worth saying why
    /// rather than leaving a gap a later reader would fill back in. Both
    /// `uniform_buffer_offset_alignment` and `storage_buffer_offset_alignment`
    /// are moduli a buffer offset has to land on, not capacities: the binding
    /// half of this layer enforces `offset % alignment`, and the native and
    /// common binding validators take the same discovered value as the
    /// alignment they check a caller's offset against. Every consumer therefore
    /// *honours* whatever the context reported instead of assuming a small
    /// value, so a coarse answer narrows which offsets a caller may choose and a
    /// fine one widens it -- neither is a reason to refuse the context.
    ///
    /// This table used to carry `uniform_buffer_offset_alignment` as
    /// `actual < 256`, which is the comparison backwards, and an inverted
    /// alignment row can only ever reject the *more* capable context: a driver
    /// whose required alignment is 16 accepts every 256-byte offset, while one
    /// demanding 512 accepts fewer. No test caught it because every fixture
    /// answered exactly 256 -- the single value an inverted comparison accepts
    /// -- so the first real desktop GL context this repository ever opened was
    /// refused by it.
    fn requirements(self, l: &GlLimits) -> [(&'static str, u64, u64); 14] {
        [
            (
                "max_texture_size",
                u64::from(l.max_texture_size),
                self.texture,
            ),
            (
                "max_3d_texture_size",
                u64::from(l.max_3d_texture_size),
                self.three_d,
            ),
            (
                "max_array_texture_layers",
                u64::from(l.max_array_texture_layers),
                self.layers,
            ),
            (
                "max_cube_map_texture_size",
                u64::from(l.max_cube_map_texture_size),
                self.cube,
            ),
            (
                "max_renderbuffer_size",
                u64::from(l.max_renderbuffer_size),
                self.renderbuffer,
            ),
            (
                "max_color_attachments",
                u64::from(l.max_color_attachments),
                self.attachments,
            ),
            (
                "max_draw_buffers",
                u64::from(l.max_draw_buffers),
                self.draw_buffers,
            ),
            (
                "max_viewport_width",
                u64::from(l.max_viewport_dimensions[0]),
                self.viewport,
            ),
            (
                "max_viewport_height",
                u64::from(l.max_viewport_dimensions[1]),
                self.viewport,
            ),
            (
                "max_combined_texture_image_units",
                u64::from(l.max_combined_texture_image_units),
                self.texture_units,
            ),
            (
                "max_uniform_buffer_bindings",
                u64::from(l.max_uniform_buffer_bindings),
                self.uniform_bindings,
            ),
            ("max_uniform_block_size", l.max_uniform_block_size, 16_384),
            ("max_samples", u64::from(l.max_samples), self.samples),
            (
                "max_vertex_attributes",
                u64::from(l.max_vertex_attributes),
                16,
            ),
        ]
    }
}
