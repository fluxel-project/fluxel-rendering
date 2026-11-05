//! Compiled-plan and concrete-owner validation for the Deferred probe.

use super::declaration::{
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
            output,
            resident_upload_slot: declaration.resident_upload_slot,
            frame_upload_slot: declaration.frame_upload_slot,
            uniform_slot: declaration.uniform_slot,
            stream_slots: declaration.stream_slots,
            texture_slot: declaration.texture_slot,
            gbuffer_slots: declaration.gbuffer_slots,
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
            || self.texture_slot.resource != declaration.texture_slot.resource
            || self.gbuffer_slots.map(|slot| slot.resource)
                != declaration.gbuffer_slots.map(|slot| slot.resource)
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
            index: if self.deformation_pass.is_some() {
                4
            } else {
                3
            },
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
        if frame.resident.device != self.id.device
            || frame.draw.pipeline.device() != self.id.device
            || frame.draw.material_group.device() != self.id.device
            || frame.draw.transform_group.device() != self.id.device
            || frame.lighting_pipeline.device() != self.id.device
            || frame.lighting_group.device() != self.id.device
        {
            return Err(Error::WrongDevice);
        }
        if self.deformation_pass.is_some() != frame.deformation.is_some() {
            return Err(Error::InvalidState);
        }
        if let Some(deformation) = &frame.deformation {
            validate_deformation(&self.device, &frame.draw, deformation)?;
            if self.deformation_slots.is_none() {
                return Err(Error::InvalidState);
            }
        }
        if frame.extent != self.extent
            || frame.draw.transform_bytes != self.transform_bytes
            || self.stream_slots.len() != frame.resident.streams.len()
            || frame.draw.vertex_layouts.len() != self.stream_slots.len()
            || frame.draw.vertex_ranges.len() != self.stream_slots.len()
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
            || !matches_final_target_template(&frame.frame.target_desc, self.extent)
        {
            return Err(Error::InvalidState);
        }
        validate_draw_uniforms(frame, self.uniform_alignment)?;
        validate_uploads(frame)?;
        draw_accesses(frame, &frame.draw)?;
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
        expected.extend([self.geometry_step(), self.lighting_step(), present]);
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

    pub(super) fn bindings<'a>(
        &'a self,
        frame: &'a DeferredFrame<B>,
    ) -> Result<FrameBindings<'a, B>> {
        let uses_buffer = |slot| {
            self.execution
                .logical()
                .buffer_bindings()
                .requirements()
                .iter()
                .any(|requirement| requirement.slot == slot)
        };
        let mut buffers = vec![
            BoundBuffer::new(
                self.frame_upload_slot,
                frame.frame.upload_instance,
                &frame.frame.upload,
                &frame.frame.upload_desc,
                frame.frame.upload_incoming,
            ),
            BoundBuffer::new(
                self.uniform_slot,
                frame.frame.uniform_instance,
                &frame.frame.uniform,
                &frame.frame.uniform_desc,
                frame.frame.uniform_incoming,
            ),
            BoundBuffer::new(
                self.readback_slot.ok_or(Error::InvalidState)?,
                frame.frame.readback_instance,
                &frame.frame.readback,
                &frame.frame.readback_desc,
                frame.frame.readback_incoming,
            ),
        ];
        if uses_buffer(self.resident_upload_slot) {
            buffers.push(BoundBuffer::new(
                self.resident_upload_slot,
                frame.resident.upload_instance,
                &frame.resident.upload,
                &frame.resident.upload_desc,
                frame.resident.upload_incoming,
            ));
        }
        for (slot, stream) in self
            .stream_slots
            .iter()
            .copied()
            .zip(&frame.resident.streams)
        {
            if uses_buffer(slot) {
                buffers.push(BoundBuffer::new(
                    slot,
                    stream.instance,
                    &stream.buffer,
                    &stream.desc,
                    stream.incoming,
                ));
            }
        }
        if let (Some(slots), Some(deformation)) = (self.deformation_slots, &frame.deformation) {
            for (slot, owner) in [
                (slots[0], &deformation.upload),
                (slots[1], &deformation.source),
                (slots[2], &deformation.product),
                (slots[3], &deformation.parameters),
            ] {
                if uses_buffer(slot) {
                    buffers.push(BoundBuffer::new(
                        slot,
                        owner.instance,
                        &owner.buffer,
                        &owner.desc,
                        owner.incoming,
                    ));
                }
            }
        }
        let textures = std::iter::once(BoundTexture::new(
            self.texture_slot,
            frame.resident.textures[0].instance,
            &frame.resident.textures[0].texture,
            &frame.resident.textures[0].desc,
            frame.resident.textures[0].incoming,
            frame.resident.textures[0].layout,
        ))
        .chain(
            self.gbuffer_slots
                .iter()
                .copied()
                .zip(&frame.gbuffer)
                .map(|(slot, texture)| {
                    BoundTexture::new(
                        slot,
                        texture.instance,
                        &texture.texture,
                        &texture.desc,
                        texture.incoming,
                        texture.layout,
                    )
                }),
        )
        .chain(std::iter::once(BoundTexture::new(
            self.target_slot.ok_or(Error::InvalidState)?,
            frame.frame.target_instance,
            &frame.frame.target,
            &frame.frame.target_desc,
            frame.frame.target_incoming,
            frame.frame.target_layout,
        )))
        .collect();
        FrameBindings::new(&self.execution, self.id.device, buffers, textures)
            .map_err(binding_error)
    }
}

/// Read-only diagnostic evidence projected from the compiled graph.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DeferredGraphFacts {
    pub(crate) pass_count: usize,
    pub(crate) buffer_binding_count: usize,
    pub(crate) texture_binding_count: usize,
}
