//! Namespace projections over the one compiled Deferred capture Graph.

use super::*;

impl<B: ResourceBackend> DeferredPlan<B> {
    pub(super) fn capture_completion_all(
        &self,
        outcome: &Outcome<B>,
        present: Option<StepId>,
    ) -> Result<CompletionPoint<B>>
    where
        B: ExecutionBackend,
    {
        let mut expected = self
            .execution
            .logical()
            .passes()
            .iter()
            .enumerate()
            .map(|(index, _)| StepId {
                plan: self.id,
                index: u32::try_from(index).expect("graph pass index"),
            })
            .collect::<Vec<_>>();
        if let Some(step) = present {
            expected.push(step);
        }
        let facts = outcome.facts();
        if facts.status != ExecutionStatus::Accepted
            || facts.accepted != Some(&expected)
            || present.is_some_and(|step| {
                !facts.presents.is_some_and(|items| {
                    items.len() == 1 && items[0].accepted && items[0].step == Some(step)
                })
            })
        {
            return Err(Error::InvalidState);
        }
        let terminal =
            present.unwrap_or_else(|| *expected.last().expect("compiled capture has a pass"));
        facts
            .checkpoints
            .and_then(|points| {
                points.iter().find(|point| {
                    point.point.device() == self.id.device
                        && point
                            .covers
                            .lane_frontiers
                            .iter()
                            .any(|&(lane, step)| lane == self.lane && step == terminal)
                })
            })
            .map(|point| point.point.clone())
            .ok_or(Error::InvalidState)
    }

    pub(super) fn capture_namespace(
        device: &Device<B>,
        id: PlanId,
        execution: CompiledExecution,
        submission: SubmissionPlan<B>,
        declaration: DeferredDeclaration,
        present_step: Option<StepId>,
    ) -> Result<Self> {
        if id.device != device.id() || declaration.lane.device != device.id() {
            return Err(Error::WrongDevice);
        }
        let output = match declaration.output {
            DeferredOutputDeclaration::Capture => DeferredOutputPlan::Capture,
            DeferredOutputDeclaration::Offscreen { readback_pass } => {
                DeferredOutputPlan::Offscreen { readback_pass }
            }
            DeferredOutputDeclaration::Surface => DeferredOutputPlan::Surface {
                present_step: present_step.ok_or(Error::InvalidState)?,
            },
        };
        let terminal = !matches!(output, DeferredOutputPlan::Capture);
        Ok(Self {
            device: device.clone(),
            id,
            lane: declaration.lane,
            submission,
            execution,
            declaration: declaration.graph,
            upload_pass: declaration.upload_pass,
            mip_generation: declaration.mip_generation,
            indirect: declaration.indirect,
            deformation_pass: declaration.deformation_pass,
            geometry_pass: declaration.geometry_pass,
            resolve_pass: declaration.resolve_pass,
            lighting_pass: declaration.lighting_pass,
            composition_passes: Vec::new(),
            composite_pass: terminal.then_some(declaration.composite_pass).flatten(),
            output,
            resident_upload_slot: declaration.resident_upload_slot,
            frame_upload_slot: declaration.frame_upload_slot,
            uniform_slot: declaration.uniform_slot,
            stream_slots: declaration.stream_slots,
            texture_slots: declaration.texture_slots,
            environment_upload_slot: declaration.environment_upload_slot,
            environment_slot: declaration.environment_slot,
            specular_upload_slot: declaration.specular_upload_slot,
            specular_view_uniform_slot: declaration.specular_view_uniform_slot,
            specular_radiance_slot: declaration.specular_radiance_slot,
            specular_brdf_lut_slot: declaration.specular_brdf_lut_slot,
            gbuffer_slots: declaration.gbuffer_slots,
            effects_slot: declaration.effects_slot,
            diffuse_roughness_slot: declaration.diffuse_roughness_slot,
            ambient_occlusion_slot: declaration.ambient_occlusion_slot,
            msaa_gbuffer_slots: declaration.msaa_gbuffer_slots,
            msaa_effects_slot: declaration.msaa_effects_slot,
            msaa_diffuse_roughness_slot: declaration.msaa_diffuse_roughness_slot,
            msaa_ambient_occlusion_slot: declaration.msaa_ambient_occlusion_slot,
            scene_color_slot: declaration.scene_color_slot,
            surface_target_slot: declaration.surface_target_slot,
            target_slot: terminal.then_some(declaration.target_slot).flatten(),
            readback_slot: terminal.then_some(declaration.readback_slot).flatten(),
            deformation_slots: declaration.deformation_slots,
            extent: declaration.extent,
            uniform_alignment: declaration.uniform_alignment,
            transform_bytes: declaration.transform_bytes,
        })
    }
}
