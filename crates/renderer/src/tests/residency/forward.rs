use super::*;

impl<B: ResourceBackend> ForwardResidentBuffers<B> {
    /// Allocates static device resources and returns exact staging copy facts.
    pub(crate) fn allocate(
        device: &Device<B>,
        streams: Vec<ForwardStreamInput>,
        textures: Vec<ForwardTextureInput>,
    ) -> Result<(Arc<Self>, ForwardResidentUpload)> {
        Self::allocate_planned(device, plan_forward_resident(streams, textures)?)
    }
}

impl<B: ResourceBackend> ForwardFrameBuffers<B> {
    /// Allocates per-frame owners; `uniform_bytes` are copied into `uniform_range`.
    pub(crate) fn allocate(
        device: &Device<B>,
        extent: [u32; 2],
        uniform_capacity: u64,
        uniform_bytes: Vec<u8>,
        uniform_range: Range<u64>,
        row_pitch: u32,
    ) -> Result<(Self, ForwardUniformUpload)> {
        Self::allocate_planned(
            device,
            plan_forward_frame(
                extent,
                uniform_capacity,
                u64::try_from(uniform_bytes.len()).map_err(|_| Error::InvalidRange)?,
                uniform_range,
                row_pitch,
                device.capabilities().limits.uniform_offset_alignment,
            )?,
        )
    }
}

impl<B: ResourceBackend> SurfaceForwardFrameBuffers<B> {
    pub(crate) fn allocate(
        device: &Device<B>,
        uniform_capacity: u64,
        uniform_bytes: Vec<u8>,
        uniform_range: Range<u64>,
    ) -> Result<(Self, ForwardUniformUpload)> {
        Self::allocate_planned(
            device,
            plan_surface_forward_frame(
                uniform_capacity,
                u64::try_from(uniform_bytes.len()).map_err(|_| Error::InvalidRange)?,
                uniform_range,
                device.capabilities().limits.uniform_offset_alignment,
            )?,
        )
    }
}
