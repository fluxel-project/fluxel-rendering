//! Surface declaration delegates to the shared capture OIT authoring path.

use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn declare_capture_oit_surface_with_capture_contribution<B, N, C>(
    device: &Device<B>,
    capture: &ForwardFrame<B>,
    receiver: &ForwardFrame<B>,
    capture_oit: Option<&ForwardOitInput<'_, B>>,
    receiver_oit: Option<&ForwardOitInput<'_, B>>,
    scene_refraction: bool,
    image: &SurfaceImage<B>,
    capture_ordinary: N,
    capture_contribute: C,
) -> std::result::Result<CaptureOitDeclaration, ForwardPlanBuildError>
where
    B: ResourceBackend,
    N: FnOnce(&mut ForwardPhaseContext<'_, '_, '_, B>) -> std::result::Result<(), PipelineError>,
    C: FnMut(&mut ForwardOitReadContext<'_, '_>) -> std::result::Result<(), PipelineError>,
{
    super::capture_oit_declaration::declare_capture_oit_inner(
        device,
        capture,
        receiver,
        capture_oit,
        receiver_oit,
        scene_refraction,
        Some(image),
        capture_ordinary,
        capture_contribute,
    )
}
