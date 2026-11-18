//! Fixed raster state shared by realized Deferred programs.
use fluxel_rhi::api::*;

pub(in crate::renderer::deferred) fn color_target_with_format(
    format: TextureFormat,
) -> ColorTargetState {
    ColorTargetState {
        format,
        blend: None,
        write_mask: 0xf,
    }
}
pub(in crate::renderer::deferred) fn opaque_triangles() -> PrimitiveState {
    PrimitiveState {
        topology: PrimitiveTopology::TriangleList,
        strip_index_format: None,
        front_face: FrontFace::Ccw,
        cull_mode: CullMode::None,
        unclipped_depth: false,
        polygon_mode: PolygonMode::Fill,
        conservative: false,
    }
}

/// Triangle state for one material draw after its transform-specific winding
/// and Material-owned cull policy have been selected.
pub(super) fn surface_triangles(surface: crate::renderer::RasterSurfaceState) -> PrimitiveState {
    let mut primitive = opaque_triangles();
    surface.apply_to(&mut primitive);
    primitive
}
pub(in crate::renderer::deferred) fn single_sample() -> MultisampleState {
    multisample(1)
}

/// Fixed multisample state for Deferred geometry attachments.
pub(super) fn multisample(count: u32) -> MultisampleState {
    MultisampleState {
        count,
        mask: if count > 32 {
            u64::MAX
        } else {
            u64::from(u32::MAX)
        },
        alpha_to_coverage_enabled: false,
    }
}
pub(in crate::renderer::deferred) fn depth_state() -> DepthStencilState {
    let stencil = StencilFaceState {
        compare: CompareFunction::Always,
        fail: StencilOperation::Keep,
        depth_fail: StencilOperation::Keep,
        pass: StencilOperation::Keep,
    };
    DepthStencilState {
        format: TextureFormat::Depth32Float,
        depth_write_enabled: true,
        depth_compare: CompareFunction::LessEqual,
        stencil_front: stencil,
        stencil_back: stencil,
        stencil_read_mask: 0,
        stencil_write_mask: 0,
        depth_bias_constant: 0,
        depth_bias_slope_scale: 0.0,
        depth_bias_clamp: 0.0,
    }
}
