//! Surface declaration delegates to the shared capture OIT authoring path.

use super::*;

pub(super) fn declare_capture_oit_surface<B: ResourceBackend>(
    device: &Device<B>,
    capture: &ForwardFrame<B>,
    receiver: &ForwardFrame<B>,
    capture_oit: Option<&ForwardOitInput<'_, B>>,
    receiver_oit: Option<&ForwardOitInput<'_, B>>,
    image: &SurfaceImage<B>,
) -> std::result::Result<CaptureOitDeclaration, ForwardPlanBuildError> {
    super::capture_oit_declaration::declare_capture_oit_inner(
        device,
        capture,
        receiver,
        capture_oit,
        receiver_oit,
        Some(image),
    )
}
