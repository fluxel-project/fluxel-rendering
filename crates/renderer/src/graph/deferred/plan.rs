//! Compiled-plan and concrete-owner validation for the Deferred probe.

use super::declaration_validation::{
    matches_final_target_template, matches_gbuffer_template, validate_deformation,
    validate_draw_uniforms, validate_uploads,
};
use super::*;

impl<B: ResourceBackend> DeferredPlan<B> {
    /// Compiles the one already-authored Graph declaration into a device plan.
    ///
    /// Callers that cache this plan must author a fresh declaration for every
    /// frame and call [`Self::validate_declaration`] before selecting a hit.
    pub(crate) fn compile_declared(
        device: &Device<B>,
        id: PlanId,
        declaration: DeferredDeclaration,
    ) -> Result<Self> {
        if id.device != device.id() || declaration.lane.device != device.id() {
            return Err(Error::WrongDevice);
        }
        let execution = CompiledExecution::compile(&declaration.graph)?;
        let mut expected = vec![declaration.upload_pass];
        if let Some(pass) = declaration.deformation_pass {
            expected.push(pass);
        }
        expected.extend([declaration.geometry_pass, declaration.lighting_pass]);
        if let Some(pass) = declaration.composite_pass {
            expected.push(pass);
        }
        if let DeferredOutputDeclaration::Offscreen { readback_pass } = declaration.output {
            expected.push(readback_pass);
        }
        if execution.logical().passes() != expected {
            return Err(Error::InvalidState);
        }
        let (submission, output) = match declaration.output {
            DeferredOutputDeclaration::Offscreen { readback_pass } => (
                execution.submission(device, id, declaration.lane)?,
                DeferredOutputPlan::Offscreen { readback_pass },
            ),
            DeferredOutputDeclaration::Surface => {
                let index = u32::try_from(expected.len()).map_err(|_| Error::InvalidRange)?;
                let present_step = StepId { plan: id, index };
                let mut steps = execution.steps(device, id, declaration.lane)?;
                steps.push(fluxel_rhi::api::StepDesc {
                    lane: declaration.lane,
                    kind: fluxel_rhi::api::StepKind::Present,
                    waits: vec![fluxel_rhi::api::StepWait {
                        previous: StepId {
                            plan: id,
                            index: index.checked_sub(1).ok_or(Error::InvalidState)?,
                        },
                        destination: fluxel_rhi::api::AccessScope {
                            stages: StageMask::ALL_COMMANDS,
                            accesses: AccessMask::ALL,
                        },
                    }],
                });
                // SAFETY: the common Graph steps precede a single presentation
                // step explicitly waiting for the final lighting operation.
                let submission =
                    unsafe { SubmissionPlan::from_compiled(id, std::sync::Arc::from(steps)) };
                submission.validate_for_execution(device)?;
                (submission, DeferredOutputPlan::Surface { present_step })
            }
        };
        Ok(Self {
            device: device.clone(),
            id,
            lane: declaration.lane,
            submission,
            execution,
            declaration: declaration.graph.clone(),
            upload_pass: declaration.upload_pass,
            deformation_pass: declaration.deformation_pass,
            geometry_pass: declaration.geometry_pass,
            lighting_pass: declaration.lighting_pass,
            composite_pass: declaration.composite_pass,
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
            scene_color_slot: declaration.scene_color_slot,
            surface_target_slot: declaration.surface_target_slot,
            target_slot: declaration.target_slot,
            readback_slot: declaration.readback_slot,
            deformation_slots: declaration.deformation_slots,
            extent: declaration.extent,
            uniform_alignment: declaration.uniform_alignment,
            transform_bytes: declaration.transform_bytes,
        })
    }
    /// Checks whether a freshly authored, handle-free Graph declaration can
    /// share this plan. Concrete owners are intentionally excluded here and
    /// are checked again by [`Self::validate_frame`] before recording.
    pub(crate) fn validate_declaration(&self, declaration: &DeferredDeclaration) -> Result<()> {
        let same_buffer_slots = |left: &[BufferSlot], right: &[BufferSlot]| {
            left.len() == right.len()
                && left
                    .iter()
                    .zip(right)
                    .all(|(left, right)| left.resource == right.resource)
        };
        let same_deformation_slots =
            |left: Option<[BufferSlot; 4]>, right: Option<[BufferSlot; 4]>| {
                matches!((left, right), (None, None))
                    || matches!((left, right), (Some(left), Some(right)) if left
                    .iter()
                    .zip(right)
                    .all(|(left, right)| left.resource == right.resource))
            };
        if !self.declaration.same_declaration(&declaration.graph)
            || self.lane != declaration.lane
            || self.upload_pass.1 != declaration.upload_pass.1
            || self.deformation_pass.map(|pass| pass.1)
                != declaration.deformation_pass.map(|pass| pass.1)
            || self.geometry_pass.1 != declaration.geometry_pass.1
            || self.lighting_pass.1 != declaration.lighting_pass.1
            || self.composite_pass.map(|pass| pass.1)
                != declaration.composite_pass.map(|pass| pass.1)
            || !match (self.output, declaration.output) {
                (
                    DeferredOutputPlan::Offscreen {
                        readback_pass: left,
                    },
                    DeferredOutputDeclaration::Offscreen {
                        readback_pass: right,
                    },
                ) => left.1 == right.1,
                (DeferredOutputPlan::Surface { .. }, DeferredOutputDeclaration::Surface) => true,
                _ => false,
            }
            || self.resident_upload_slot.resource != declaration.resident_upload_slot.resource
            || self.frame_upload_slot.resource != declaration.frame_upload_slot.resource
            || self.uniform_slot.resource != declaration.uniform_slot.resource
            || !same_buffer_slots(&self.stream_slots, &declaration.stream_slots)
            || self.texture_slots.len() != declaration.texture_slots.len()
            || self
                .texture_slots
                .iter()
                .zip(&declaration.texture_slots)
                .any(|(left, right)| left.resource != right.resource)
            || self.environment_upload_slot.map(|slot| slot.resource)
                != declaration
                    .environment_upload_slot
                    .map(|slot| slot.resource)
            || self.environment_slot.map(|slot| slot.resource)
                != declaration.environment_slot.map(|slot| slot.resource)
            || self.specular_upload_slot.map(|slot| slot.resource)
                != declaration.specular_upload_slot.map(|slot| slot.resource)
            || self.specular_view_uniform_slot.map(|slot| slot.resource)
                != declaration
                    .specular_view_uniform_slot
                    .map(|slot| slot.resource)
            || self.specular_radiance_slot.map(|slot| slot.resource)
                != declaration.specular_radiance_slot.map(|slot| slot.resource)
            || self.specular_brdf_lut_slot.map(|slot| slot.resource)
                != declaration.specular_brdf_lut_slot.map(|slot| slot.resource)
            || self.gbuffer_slots.map(|slot| slot.resource)
                != declaration.gbuffer_slots.map(|slot| slot.resource)
            || self.effects_slot.map(|slot| slot.resource)
                != declaration.effects_slot.map(|slot| slot.resource)
            || self.scene_color_slot.map(|slot| slot.resource)
                != declaration.scene_color_slot.map(|slot| slot.resource)
            || self.surface_target_slot.map(|slot| slot.resource)
                != declaration.surface_target_slot.map(|slot| slot.resource)
            || self.target_slot.map(|slot| slot.resource)
                != declaration.target_slot.map(|slot| slot.resource)
            || self.readback_slot.map(|slot| slot.resource)
                != declaration.readback_slot.map(|slot| slot.resource)
            || !same_deformation_slots(self.deformation_slots, declaration.deformation_slots)
            || self.extent != declaration.extent
            || self.uniform_alignment != declaration.uniform_alignment
            || self.transform_bytes != declaration.transform_bytes
        {
            return Err(Error::InvalidState);
        }
        Ok(())
    }

    pub(crate) fn id(&self) -> PlanId {
        self.id
    }

    pub(crate) fn submission(&self) -> &SubmissionPlan<B> {
        &self.submission
    }

    pub(crate) fn upload_step(&self) -> StepId {
        StepId {
            plan: self.id,
            index: 0,
        }
    }

    pub(crate) fn geometry_step(&self) -> StepId {
        StepId {
            plan: self.id,
            index: if self.deformation_pass.is_some() {
                2
            } else {
                1
            },
        }
    }

    pub(crate) fn deformation_step(&self) -> Option<StepId> {
        self.deformation_pass.map(|_| StepId {
            plan: self.id,
            index: 1,
        })
    }

    pub(crate) fn lighting_step(&self) -> StepId {
        StepId {
            plan: self.id,
            index: if self.deformation_pass.is_some() {
                3
            } else {
                2
            },
        }
    }

    pub(crate) fn readback_step(&self) -> StepId {
        if !matches!(self.output, DeferredOutputPlan::Offscreen { .. }) {
            return self.lighting_step();
        }
        StepId {
            plan: self.id,
            index: self.composite_step().index + 1,
        }
    }

    pub(crate) fn composite_step(&self) -> StepId {
        StepId {
            plan: self.id,
            index: self.lighting_step().index + u32::from(self.composite_pass.is_some()),
        }
    }

    pub(crate) fn present_step(&self) -> Result<StepId> {
        match self.output {
            DeferredOutputPlan::Surface { present_step } => Ok(present_step),
            DeferredOutputPlan::Offscreen { .. } => Err(Error::InvalidState),
        }
    }

    /// Scalar facts derived from the compiled graph; no graph handle escapes.
    pub(crate) fn facts(&self) -> DeferredGraphFacts {
        DeferredGraphFacts {
            pass_count: self.execution.logical().passes().len(),
            buffer_binding_count: self
                .execution
                .logical()
                .buffer_bindings()
                .requirements()
                .len(),
            texture_binding_count: self
                .execution
                .logical()
                .texture_bindings()
                .requirements()
                .len(),
        }
    }

    pub(crate) fn pass_count(&self) -> usize {
        self.facts().pass_count
    }

    pub(crate) fn binding_counts(&self) -> (usize, usize) {
        let facts = self.facts();
        (facts.buffer_binding_count, facts.texture_binding_count)
    }

    /// Validates concrete owners immediately before recording.
    ///
    /// The selected plan may be cached, but its caller has just re-authored a
    /// declaration for this frame. This validates the concrete owners and
    /// ranges against that cached Graph before mapping or recording.
    pub(crate) fn validate_frame(&self, frame: &DeferredFrame<B>) -> Result<()> {
        if frame.draws.is_empty()
            || frame.resident.device != self.id.device
            || frame.draws.iter().any(|draw| {
                draw.pipeline.device() != self.id.device
                    || draw.material_group.device() != self.id.device
                    || draw.transform_group.device() != self.id.device
            })
            || frame.lighting_pipeline.device() != self.id.device
            || frame.lighting_group.device() != self.id.device
            || frame.composite_pipeline.device() != self.id.device
            || frame.composite_group.device() != self.id.device
            || frame.scene_color.texture.device() != self.id.device
            || frame.scene_color.view.device() != self.id.device
            || frame.scene_color_sampler.device() != self.id.device
            || frame.effects.as_ref().is_some_and(|effects| {
                effects.texture.device() != self.id.device
                    || effects.view.device() != self.id.device
            })
        {
            return Err(Error::WrongDevice);
        }
        if self.deformation_pass.is_some() != frame.deformation.is_some() {
            return Err(Error::InvalidState);
        }
        if self.specular_upload_slot.is_some() != frame.specular.is_some()
            || self.specular_view_uniform_slot.is_some() != frame.specular.is_some()
            || self.specular_radiance_slot.is_some() != frame.specular.is_some()
            || self.specular_brdf_lut_slot.is_some() != frame.specular.is_some()
        {
            return Err(Error::InvalidState);
        }
        if let Some(specular) = &frame.specular {
            specular.validate_for_sampling(self.id.device)?;
        }
        if !frame.scene_color.filterable {
            return Err(Error::Unsupported("filterable Deferred scene-color"));
        }
        crate::renderer::validate_ordinary_sampler(&frame.scene_color_sampler_desc)?;
        if let Some(deformation) = &frame.deformation {
            if frame.draws.len() != 1 {
                return Err(Error::InvalidState);
            }
            validate_deformation(&self.device, &frame.draws[0], deformation)?;
            if self.deformation_slots.is_none() {
                return Err(Error::InvalidState);
            }
        }
        if frame.extent != self.extent
            || frame
                .draws
                .iter()
                .map(|draw| draw.transform_bytes)
                .collect::<Vec<_>>()
                != self.transform_bytes
            || self.stream_slots.len() != frame.resident.streams.len()
            || self.texture_slots.len() != frame.resident.textures.len()
            || !matches_gbuffer_template(
                &frame.gbuffer[0].desc,
                self.extent,
                fluxel_rhi::api::TextureFormat::Rgba8Unorm,
            )
            || !matches_gbuffer_template(
                &frame.gbuffer[1].desc,
                self.extent,
                fluxel_rhi::api::TextureFormat::Rgba8Unorm,
            )
            || !matches_gbuffer_template(
                &frame.gbuffer[2].desc,
                self.extent,
                fluxel_rhi::api::TextureFormat::Depth32Float,
            )
            || frame.effects.as_ref().is_some_and(|effects| {
                !matches_gbuffer_template(
                    &effects.desc,
                    self.extent,
                    fluxel_rhi::api::TextureFormat::Rgba16Float,
                )
            })
            || self.effects_slot.is_some() != frame.effects.is_some()
            || !matches_final_target_template(&frame.frame.target_desc, self.extent)
            || !matches_gbuffer_template(
                &frame.scene_color.desc,
                self.extent,
                fluxel_rhi::api::TextureFormat::Rgba8Unorm,
            )
            || self.composite_pass.is_none()
            || self.scene_color_slot.is_none()
        {
            return Err(Error::InvalidState);
        }
        validate_draw_uniforms(frame, self.uniform_alignment)?;
        validate_uploads(frame)?;
        for draw in &frame.draws {
            draw_accesses(frame, draw)?;
        }
        self.bindings(frame).map(|_| ())
    }

    pub(crate) fn completion(&self, outcome: &Outcome<B>) -> Result<CompletionPoint<B>>
    where
        B: ExecutionBackend,
    {
        let mut expected = vec![self.upload_step()];
        if let Some(step) = self.deformation_step() {
            expected.push(step);
        }
        expected.extend([self.geometry_step(), self.lighting_step()]);
        if self.composite_pass.is_some() {
            expected.push(self.composite_step());
        }
        if matches!(self.output, DeferredOutputPlan::Offscreen { .. }) {
            expected.push(self.readback_step());
        }
        let facts = outcome.facts();
        if facts.status != ExecutionStatus::Accepted || facts.accepted != Some(&expected) {
            return Err(Error::InvalidState);
        }
        facts
            .checkpoints
            .and_then(|points| {
                points.iter().find(|point| {
                    point.point.device() == self.id.device
                        && point
                            .covers
                            .lane_frontiers
                            .iter()
                            .any(|&(lane, step)| lane == self.lane && step == self.readback_step())
                })
            })
            .map(|point| point.point.clone())
            .ok_or(Error::InvalidState)
    }

    pub(crate) fn surface_completion(&self, outcome: &Outcome<B>) -> Result<CompletionPoint<B>>
    where
        B: ExecutionBackend,
    {
        let present = self.present_step()?;
        let mut expected = vec![self.upload_step()];
        if let Some(step) = self.deformation_step() {
            expected.push(step);
        }
        expected.extend([
            self.geometry_step(),
            self.lighting_step(),
            self.composite_step(),
            present,
        ]);
        let facts = outcome.facts();
        if facts.status != ExecutionStatus::Accepted
            || facts.accepted != Some(&expected)
            || !facts.presents.is_some_and(|items| {
                items.len() == 1 && items[0].accepted && items[0].step == Some(present)
            })
        {
            return Err(Error::InvalidState);
        }
        facts
            .checkpoints
            .and_then(|points| {
                points.iter().find(|point| {
                    point.point.device() == self.id.device
                        && point
                            .covers
                            .lane_frontiers
                            .iter()
                            .any(|&(lane, step)| lane == self.lane && step == present)
                })
            })
            .map(|point| point.point.clone())
            .ok_or(Error::InvalidState)
    }
}

/// Read-only diagnostic evidence projected from the compiled graph.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DeferredGraphFacts {
    pub(crate) pass_count: usize,
    pub(crate) buffer_binding_count: usize,
    pub(crate) texture_binding_count: usize,
}
