//! DXGI presentation lowering. Native windows remain registered by opaque RHI IDs.

mod surface;

pub(crate) use surface::{Dx12FrameAttachment, Dx12Presentation};
