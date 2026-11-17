//! Frame-local allocation choices after Forward residency assembly.

use super::*;

pub(super) struct PreparedFrameBuffers<B: ResourceBackend> {
    pub(super) offscreen: Option<crate::residency::ForwardFrameBuffers<B>>,
    pub(super) surface: Option<crate::residency::SurfaceForwardFrameBuffers<B>>,
    pub(super) uniform_upload: crate::residency::ForwardUniformUpload,
    pub(super) buffers: u64,
    pub(super) bytes: u64,
    pub(super) textures: u64,
}

impl<B> Renderer<B>
where
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + RasterBackend
        + PipelineBackend
        + ShaderBackend,
{
    #[allow(clippy::too_many_arguments)]
    pub(super) fn allocate_prepared_forward_frame(
        &mut self,
        slot: usize,
        extent: [u32; 2],
        uniform_len: u64,
        row_pitch: u32,
        alignment: u64,
        surface: bool,
        sampled_target: bool,
        attachments: Option<&BorrowedForwardAttachments<B>>,
    ) -> std::result::Result<PreparedFrameBuffers<B>, RendererError> {
        if let Some(attachments) = attachments {
            return self.allocate_borrowed_frame(
                attachments,
                extent,
                uniform_len,
                row_pitch,
                alignment,
            );
        }
        if surface {
            let plan = crate::residency::plan_surface_forward_frame(
                uniform_len,
                uniform_len,
                0..uniform_len,
                alignment,
            )
            .map_err(|error| self.observe_forward_error("surface frame planning", error))?;
            let bytes = plan.upload_desc.size.saturating_add(plan.uniform_desc.size);
            let (surface, uniform_upload) =
                crate::residency::SurfaceForwardFrameBuffers::allocate_planned(&self.device, plan)
                    .map_err(|error| {
                        self.observe_forward_error("surface frame allocation", error)
                    })?;
            return Ok(PreparedFrameBuffers {
                offscreen: None,
                surface: Some(surface),
                uniform_upload,
                buffers: 2,
                bytes,
                textures: 0,
            });
        }
        if !sampled_target {
            if let Some(offscreen) = self.forward_frames[slot].take().filter(|buffers| {
                crate::frame_data::pool::compatible(
                    buffers,
                    self.device.id(),
                    extent,
                    uniform_len,
                    row_pitch,
                )
            }) {
                self.forward_stats.frame_reuses = self.forward_stats.frame_reuses.saturating_add(1);
                return Ok(PreparedFrameBuffers {
                    offscreen: Some(offscreen),
                    surface: None,
                    uniform_upload: crate::residency::ForwardUniformUpload {
                        source: 0..uniform_len,
                        destination: 0..uniform_len,
                    },
                    buffers: 0,
                    bytes: 0,
                    textures: 0,
                });
            }
        }
        let plan = crate::residency::plan_forward_frame(
            extent,
            uniform_len,
            uniform_len,
            0..uniform_len,
            row_pitch,
            alignment,
            sampled_target,
        )
        .map_err(|error| self.observe_forward_error("frame planning", error))?;
        let bytes = plan
            .upload_desc
            .size
            .saturating_add(plan.uniform_desc.size)
            .saturating_add(plan.readback_desc.size);
        let (offscreen, uniform_upload) =
            crate::residency::ForwardFrameBuffers::allocate_planned(&self.device, plan)
                .map_err(|error| self.observe_forward_error("frame allocation", error))?;
        Ok(PreparedFrameBuffers {
            offscreen: Some(offscreen),
            surface: None,
            uniform_upload,
            buffers: 3,
            bytes,
            textures: 1,
        })
    }

    fn allocate_borrowed_frame(
        &mut self,
        attachments: &BorrowedForwardAttachments<B>,
        extent: [u32; 2],
        uniform_len: u64,
        row_pitch: u32,
        alignment: u64,
    ) -> std::result::Result<PreparedFrameBuffers<B>, RendererError> {
        let plan = crate::residency::plan_forward_frame(
            extent,
            uniform_len,
            uniform_len,
            0..uniform_len,
            row_pitch,
            alignment,
            false,
        )
        .map_err(|error| self.observe_forward_error("borrowed frame planning", error))?;
        let bytes = plan
            .upload_desc
            .size
            .saturating_add(plan.uniform_desc.size)
            .saturating_add(plan.readback_desc.size);
        let (offscreen, uniform_upload) =
            crate::residency::ForwardFrameBuffers::allocate_with_attachments(
                &self.device,
                plan,
                &attachments.color,
                &attachments.depth,
            )
            .map_err(|error| self.observe_forward_error("borrowed frame allocation", error))?;
        Ok(PreparedFrameBuffers {
            offscreen: Some(offscreen),
            surface: None,
            uniform_upload,
            buffers: 3,
            bytes,
            textures: 0,
        })
    }
}
