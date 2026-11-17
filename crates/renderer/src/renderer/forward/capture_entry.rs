//! Public submission entry for one same-Graph Forward capture receiver.

use super::*;
use std::time::Instant;

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
    /// Submits capture and receiver work to one acquired Surface lease.
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_forward_capture_surface<
        P: crate::FramePipeline<crate::ForwardResources>,
    >(
        &mut self,
        pipeline: &P,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: crate::RenderBounds,
        assets: &crate::ForwardAssets,
        producer_slot: CaptureTextureSlot,
        receiver: crate::PreparedCaptureReceiver,
        image: SurfaceImage<B>,
    ) -> std::result::Result<super::Submission, RendererError> {
        self.ensure_open()?;
        if image.device() != self.device.id() {
            return Err(Error::WrongDevice.into());
        }
        if !image.is_current() {
            return Err(Error::SurfaceOutdated.into());
        }
        self.free_slot().ok_or(RendererError::Backpressure)?;
        let preparation_started = Instant::now();
        let selected = pipeline
            .prepare(&snapshot, view, visibility)
            .map_err(RendererError::ForwardPipeline)?;
        selected
            .validate_for(&snapshot, view)
            .map_err(RendererError::ForwardPipeline)?;
        let depth_mode = pipeline.forward_depth_mode(&selected);
        let topology = pipeline
            .topology(&selected)
            .map_err(RendererError::ForwardPipeline)?;
        if !matches!(
            topology,
            crate::FrameTopology::OpaqueForward | crate::FrameTopology::SortedForward
        ) {
            return Err(RendererError::ForwardPipeline(
                crate::PipelineError::InvalidPhaseContribution,
            ));
        }
        let extent = selected.view_record().extent;
        if extent != image.extent() {
            return Err(Error::InvalidRange.into());
        }
        let receiver_input = receiver
            .with_lighting(
                &mut fluxel_shader::ShaderComposer::new(),
                self.environment_irradiance.is_some(),
                self.environment_specular.is_some(),
            )
            .map_err(RendererError::ForwardPreparation)?
            .into_draw_input();
        validate_receiver_sources(&receiver_input, &producer_slot)?;
        let receiver_topology = receiver_topology(&receiver_input)?;
        let retained_assets = capture_assets(assets, &receiver_input);
        if receiver_input.extent() != extent {
            return Err(Error::InvalidRange.into());
        }
        let capture_inputs = crate::preparation::prepare_forward_with_lighting(
            &selected,
            assets,
            &mut fluxel_shader::ShaderComposer::new(),
            self.environment_irradiance.is_some(),
            self.environment_specular.is_some(),
        )
        .map_err(RendererError::ForwardPreparation)?;
        let capture_inputs =
            super::composed::order_draws(pipeline, &selected, capture_inputs, topology)
                .map_err(RendererError::ForwardPipeline)?;
        let environment = self.environment_irradiance.clone().filter(|_| {
            capture_inputs
                .iter()
                .any(|input| input.prepared().uses_environment_irradiance())
        });
        self.submit_prepared_forward_capture(
            snapshot.clone(),
            capture_inputs,
            receiver_input,
            producer_slot,
            extent,
            topology,
            receiver_topology,
            depth_mode,
            &retained_assets,
            Some(image),
            preparation_started,
            environment,
            |context| pipeline.contribute(&selected, context),
        )
        .await
    }

    /// Submits a capture producer and its material receiver in one Graph.
    ///
    /// The receiver has already resolved its exact geometry and material
    /// generations. Capture bindings consume the independently named producer's
    /// target, alongside asset bindings; no CPU capture texture is synthesized.
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_forward_capture<P: crate::FramePipeline<crate::ForwardResources>>(
        &mut self,
        pipeline: &P,
        snapshot: crate::FrameSnapshot<crate::ForwardResources>,
        view: crate::RenderViewId,
        visibility: crate::RenderBounds,
        assets: &crate::ForwardAssets,
        producer_slot: CaptureTextureSlot,
        receiver: crate::PreparedCaptureReceiver,
    ) -> std::result::Result<super::Submission, RendererError> {
        self.ensure_open()?;
        self.free_slot().ok_or(RendererError::Backpressure)?;
        let preparation_started = Instant::now();
        let selected = pipeline
            .prepare(&snapshot, view, visibility)
            .map_err(RendererError::ForwardPipeline)?;
        selected
            .validate_for(&snapshot, view)
            .map_err(RendererError::ForwardPipeline)?;
        let depth_mode = pipeline.forward_depth_mode(&selected);
        let topology = pipeline
            .topology(&selected)
            .map_err(RendererError::ForwardPipeline)?;
        if !matches!(
            topology,
            crate::FrameTopology::OpaqueForward | crate::FrameTopology::SortedForward
        ) {
            return Err(RendererError::ForwardPipeline(
                crate::PipelineError::InvalidPhaseContribution,
            ));
        }
        let capture_inputs = crate::preparation::prepare_forward_with_lighting(
            &selected,
            assets,
            &mut fluxel_shader::ShaderComposer::new(),
            self.environment_irradiance.is_some(),
            self.environment_specular.is_some(),
        )
        .map_err(RendererError::ForwardPreparation)?;
        let extent = selected.view_record().extent;
        let receiver_input = receiver
            .with_lighting(
                &mut fluxel_shader::ShaderComposer::new(),
                self.environment_irradiance.is_some(),
                self.environment_specular.is_some(),
            )
            .map_err(RendererError::ForwardPreparation)?
            .into_draw_input();
        validate_receiver_sources(&receiver_input, &producer_slot)?;
        let receiver_topology = receiver_topology(&receiver_input)?;
        let retained_assets = capture_assets(assets, &receiver_input);
        if receiver_input.extent() != extent {
            return Err(Error::InvalidRange.into());
        }
        let capture_inputs =
            super::composed::order_draws(pipeline, &selected, capture_inputs, topology)
                .map_err(RendererError::ForwardPipeline)?;
        let capture_environment = self.environment_irradiance.clone().filter(|_| {
            capture_inputs
                .iter()
                .any(|input| input.prepared().uses_environment_irradiance())
        });
        self.submit_prepared_forward_capture(
            snapshot.clone(),
            capture_inputs,
            receiver_input,
            producer_slot,
            extent,
            topology,
            receiver_topology,
            depth_mode,
            &retained_assets,
            None,
            preparation_started,
            capture_environment,
            |context| pipeline.contribute(&selected, context),
        )
        .await
    }
}

fn receiver_topology(
    receiver: &crate::ForwardDrawInput,
) -> std::result::Result<crate::FrameTopology, RendererError> {
    super::composed::validate_transparent_input(receiver)
        .map_err(RendererError::ForwardPipeline)?;
    Ok(
        if matches!(
            receiver.prepared().material().surface_policy().alpha_mode,
            fluxel_material::SurfaceAlphaMode::Blend | fluxel_material::SurfaceAlphaMode::MaskBlend
        ) {
            crate::FrameTopology::SortedForward
        } else {
            crate::FrameTopology::OpaqueForward
        },
    )
}

/// Resolves receiver references against an independently named producer before
/// allocation or upload. Logical material names may alias the same product.
pub(in crate::renderer) fn validate_receiver_sources(
    receiver: &crate::ForwardDrawInput,
    producer: &CaptureTextureSlot,
) -> std::result::Result<(), RendererError> {
    let mut captures = 0;
    for index in 0..receiver.resolved_textures().len() {
        if let Some(slot) = receiver.capture_texture_slot(index) {
            if slot != producer.name() {
                return Err(RendererError::ForwardPreparation(
                    crate::ForwardPreparationError::MissingCaptureSlot {
                        source: receiver.source(),
                        slot: slot.to_owned(),
                    },
                ));
            }
            captures += 1;
        }
    }
    if captures == 0 {
        return Err(RendererError::ForwardPreparation(
            crate::ForwardPreparationError::CaptureReceiverRequiresSlot {
                source: receiver.source(),
            },
        ));
    }
    Ok(())
}

/// Cache retention includes the receiver's immutable snapshots even when the
/// producer scene did not select them. Pending owners remain independent.
pub(in crate::renderer) fn capture_assets(
    producer: &crate::ForwardAssets,
    receiver: &crate::ForwardDrawInput,
) -> crate::ForwardAssets {
    let mut assets = producer.clone();
    let geometry = receiver.geometry();
    if !assets
        .geometry
        .iter()
        .any(|entry| entry.id() == geometry.id() && entry.generation() == geometry.generation())
    {
        assets.geometry.push(geometry.clone());
    }
    for index in 0..receiver.resolved_textures().len() {
        if let Some(texture) = receiver.asset_texture(index) {
            if !assets.textures.iter().any(|entry| {
                entry.id() == texture.id() && entry.generation() == texture.generation()
            }) {
                assets.textures.push(texture.clone());
            }
        }
    }
    assets
}
