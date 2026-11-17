//! Bounded opaque Forward depth selection and device pipeline state.

use fluxel_rhi::{api::*, backend::ResourceBackend};

pub(super) fn allocate_depth<B: ResourceBackend>(
    device: &Device<B>,
    mode: crate::ForwardDepthMode,
    topology: crate::FrameTopology,
    extent: [u32; 2],
) -> Result<Option<crate::residency::ForwardDepthAttachment<B>>> {
    match mode {
        crate::ForwardDepthMode::Disabled => Ok(None),
        crate::ForwardDepthMode::Depth32LessEqualWrite => {
            if topology != crate::FrameTopology::OpaqueForward {
                return Err(Error::InvalidState);
            }
            crate::residency::ForwardDepthAttachment::allocate(device, extent).map(Some)
        }
    }
}

pub(super) fn pipeline_depth(mode: crate::ForwardDepthMode) -> Option<DepthStencilState> {
    if mode == crate::ForwardDepthMode::Disabled {
        return None;
    }
    let stencil = StencilFaceState {
        compare: CompareFunction::Always,
        fail: StencilOperation::Keep,
        depth_fail: StencilOperation::Keep,
        pass: StencilOperation::Keep,
    };
    Some(DepthStencilState {
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
    })
}
