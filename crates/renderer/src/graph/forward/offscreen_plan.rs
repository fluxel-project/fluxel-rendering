//! Offscreen Plan for the bounded Forward product.
use super::*;

impl<B: ResourceBackend> ForwardPlan<B> {
    pub(crate) fn declare_with<F>(
        device: &Device<B>,
        frame: &ForwardFrame<B>,
        expected: FrameTopology,
        contribute: F,
    ) -> std::result::Result<OffscreenForwardDeclaration, ForwardPlanBuildError>
    where
        F: FnOnce(
            &mut ForwardPhaseContext<'_, '_, '_, B>,
        ) -> std::result::Result<(), PipelineError>,
    {
        declare_offscreen_with(device, frame, expected, contribute)
    }

    pub(crate) fn declare_hdr_with<F>(
        device: &Device<B>,
        frame: &ForwardFrame<B>,
        expected: FrameTopology,
        image: Option<&SurfaceImage<B>>,
        contribute: F,
    ) -> std::result::Result<OffscreenForwardDeclaration, ForwardPlanBuildError>
    where
        F: FnOnce(
            &mut ForwardPhaseContext<'_, '_, '_, B>,
        ) -> std::result::Result<(), PipelineError>,
    {
        declare_offscreen_hdr_with(device, frame, expected, image, contribute)
    }

    pub(crate) fn validate_declaration(
        &self,
        declaration: &OffscreenForwardDeclaration,
    ) -> Result<()> {
        if !self.declaration.same_declaration(&declaration.graph)
            || self.lane != declaration.lane
            || self.upload_pass.1 != declaration.upload_pass.1
            || !same_mip_passes(&self.mip_passes, &declaration.mip_passes)
            || !same_indirect(&self.indirect, &declaration.indirect)
            || !same_particles(&self.particles, &declaration.particles)
            || self.raster_pass.1 != declaration.raster_pass.1
            || self.readback_pass.1 != declaration.readback_pass.1
            || self.resident_upload_slot.resource != declaration.resident_upload_slot.resource
            || self.frame_upload_slot.resource != declaration.frame_upload_slot.resource
            || self.uniform_slot.resource != declaration.uniform_slot.resource
            || self.environment_upload_slot.map(|slot| slot.resource)
                != declaration
                    .environment_upload_slot
                    .map(|slot| slot.resource)
            || self.specular_upload_slot.map(|slot| slot.resource)
                != declaration.specular_upload_slot.map(|slot| slot.resource)
            || self.specular_view_uniform_slot.map(|slot| slot.resource)
                != declaration
                    .specular_view_uniform_slot
                    .map(|slot| slot.resource)
            || !same_buffer_slots(&self.stream_slots, &declaration.stream_slots)
            || !same_texture_slots(&self.texture_slots, &declaration.texture_slots)
            || self.environment_slot.map(|slot| slot.resource)
                != declaration.environment_slot.map(|slot| slot.resource)
            || self.specular_radiance_slot.map(|slot| slot.resource)
                != declaration.specular_radiance_slot.map(|slot| slot.resource)
            || self.specular_brdf_lut_slot.map(|slot| slot.resource)
                != declaration.specular_brdf_lut_slot.map(|slot| slot.resource)
            || self.depth_slot.map(|slot| slot.resource)
                != declaration.depth_slot.map(|slot| slot.resource)
            || self.target_slot.resource != declaration.target_slot.resource
            || self.multisample_color_slot.map(|slot| slot.resource)
                != declaration.multisample_color_slot.map(|slot| slot.resource)
            || self.readback_slot.resource != declaration.readback_slot.resource
            || self.draw_count != declaration.draw_count
            || self.stream_count != declaration.stream_count
            || self.texture_count != declaration.texture_count
            || self.extent != declaration.extent
            || self.shape != declaration.shape
            || self.uniform_alignment != declaration.uniform_alignment
            || self.topology != declaration.topology
        {
            return Err(Error::InvalidState);
        }
        Ok(())
    }

    pub(crate) fn compile_declared(
        device: &Device<B>,
        id: PlanId,
        declaration: OffscreenForwardDeclaration,
    ) -> std::result::Result<Self, ForwardPlanBuildError> {
        Self::compile_declared_inner(device, id, declaration, None)
    }

    pub(super) fn compile_declared_inner(
        device: &Device<B>,
        id: PlanId,
        declaration: OffscreenForwardDeclaration,
        oit: Option<super::oit_plan::ForwardOitPlan>,
    ) -> std::result::Result<Self, ForwardPlanBuildError> {
        if id.device != device.id() || declaration.lane.device != device.id() {
            return Err(Error::WrongDevice.into());
        }
        let execution = CompiledExecution::compile(&declaration.graph)?;
        let mut expected =
            Vec::with_capacity(declaration.mip_passes.len() + declaration.indirect.len() * 2 + 5);
        expected.push(declaration.upload_pass);
        expected.extend(declaration.mip_passes.iter().map(|pass| pass.pass));
        for item in &declaration.indirect {
            expected.extend([item.graph.upload_pass, item.graph.compute_pass]);
        }
        for item in &declaration.particles {
            expected.extend([item.graph.upload_pass, item.graph.update_pass]);
        }
        expected.push(declaration.raster_pass);
        if let Some(product) = oit.as_ref() {
            expected.extend(product.passes());
            expected.extend(product.fallback);
        }
        if let Some(terminal) = declaration.hdr_terminal.as_ref() {
            match terminal {
                HdrForwardTerminal::Offscreen {
                    composite_pass,
                    readback_pass,
                    ..
                } => expected.extend([*composite_pass, *readback_pass]),
                HdrForwardTerminal::Surface { composite_pass, .. } => {
                    expected.push(*composite_pass);
                }
            }
        } else if let Some(terminal) = oit.as_ref().and_then(|product| product.hdr_readback) {
            expected.extend([terminal.composite_pass, terminal.readback_pass]);
        } else {
            expected.push(declaration.readback_pass);
        }
        if execution.logical().passes() != expected {
            return Err(Error::InvalidState.into());
        }
        let hdr_surface = matches!(
            declaration.hdr_terminal,
            Some(HdrForwardTerminal::Surface { .. })
        );
        let submission =
            if oit.as_ref().and_then(|product| product.surface).is_some() || hdr_surface {
                let present = StepId {
                    plan: id,
                    index: u32::try_from(execution.logical().passes().len())
                        .map_err(|_| Error::InvalidRange)?,
                };
                let mut steps = execution.steps(device, id, declaration.lane)?;
                let previous = present.index.checked_sub(1).ok_or(Error::InvalidState)?;
                steps.push(StepDesc {
                    lane: declaration.lane,
                    kind: StepKind::Present,
                    waits: vec![StepWait {
                        previous: StepId {
                            plan: id,
                            index: previous,
                        },
                        destination: AccessScope {
                            stages: StageMask::ALL_COMMANDS,
                            accesses: AccessMask::ALL,
                        },
                    }],
                });
                let submission = unsafe { SubmissionPlan::from_compiled(id, Arc::from(steps)) };
                submission.validate_for_execution(device)?;
                submission
            } else {
                execution.submission(device, id, declaration.lane)?
            };
        Ok(Self {
            id,
            lane: declaration.lane,
            submission,
            execution,
            declaration: declaration.graph.clone(),
            upload_pass: declaration.upload_pass,
            mip_passes: declaration.mip_passes,
            indirect: declaration.indirect,
            particles: declaration.particles,
            raster_pass: declaration.raster_pass,
            readback_pass: declaration.readback_pass,
            resident_upload_slot: declaration.resident_upload_slot,
            frame_upload_slot: declaration.frame_upload_slot,
            uniform_slot: declaration.uniform_slot,
            environment_upload_slot: declaration.environment_upload_slot,
            specular_upload_slot: declaration.specular_upload_slot,
            specular_view_uniform_slot: declaration.specular_view_uniform_slot,
            stream_slots: declaration.stream_slots,
            texture_slots: declaration.texture_slots,
            environment_slot: declaration.environment_slot,
            specular_radiance_slot: declaration.specular_radiance_slot,
            specular_brdf_lut_slot: declaration.specular_brdf_lut_slot,
            depth_slot: declaration.depth_slot,
            target_slot: declaration.target_slot,
            multisample_color_slot: declaration.multisample_color_slot,
            readback_slot: declaration.readback_slot,
            draw_count: declaration.draw_count,
            stream_count: declaration.stream_count,
            texture_count: declaration.texture_count,
            extent: declaration.extent,
            shape: declaration.shape,
            uniform_alignment: declaration.uniform_alignment,
            topology: declaration.topology,
            oit,
            hdr_terminal: declaration.hdr_terminal,
        })
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
    pub(crate) fn raster_step(&self) -> StepId {
        self.step_for(self.raster_pass)
            .expect("compiled Forward raster pass belongs to its plan")
    }

    pub(crate) fn indirect_upload_step(&self, index: usize) -> Result<StepId> {
        self.step_for(
            self.indirect
                .get(index)
                .ok_or(Error::InvalidRange)?
                .graph
                .upload_pass,
        )
    }

    pub(crate) fn indirect_compute_step(&self, index: usize) -> Result<StepId> {
        self.step_for(
            self.indirect
                .get(index)
                .ok_or(Error::InvalidRange)?
                .graph
                .compute_pass,
        )
    }
    pub(crate) fn particle_upload_step(&self, index: usize) -> Result<StepId> {
        self.step_for(
            self.particles
                .get(index)
                .ok_or(Error::InvalidRange)?
                .graph
                .upload_pass,
        )
    }
    pub(crate) fn particle_update_step(&self, index: usize) -> Result<StepId> {
        self.step_for(
            self.particles
                .get(index)
                .ok_or(Error::InvalidRange)?
                .graph
                .update_pass,
        )
    }
    pub(crate) fn readback_step(&self) -> StepId {
        StepId {
            plan: self.id,
            index: u32::try_from(self.execution.logical().passes().len() - 1)
                .expect("compiled pass count fits u32"),
        }
    }

    pub(crate) fn hdr_composite_step(&self) -> Result<StepId> {
        let pass = match self.hdr_terminal.as_ref() {
            Some(HdrForwardTerminal::Offscreen { composite_pass, .. })
            | Some(HdrForwardTerminal::Surface { composite_pass, .. }) => *composite_pass,
            None => return Err(Error::InvalidState),
        };
        self.step_for(pass)
    }

    pub(crate) fn hdr_readback_step(&self) -> Result<StepId> {
        let pass = match self.hdr_terminal.as_ref() {
            Some(HdrForwardTerminal::Offscreen { readback_pass, .. }) => *readback_pass,
            _ => return Err(Error::InvalidState),
        };
        self.step_for(pass)
    }

    pub(crate) fn hdr_present_step(&self) -> Result<StepId> {
        match self.hdr_terminal {
            Some(HdrForwardTerminal::Surface { .. }) => Ok(StepId {
                plan: self.id,
                index: u32::try_from(self.execution.logical().passes().len())
                    .map_err(|_| Error::InvalidRange)?,
            }),
            _ => Err(Error::InvalidState),
        }
    }

    fn step_for(&self, wanted: GraphPass) -> Result<StepId> {
        let index = self
            .execution
            .logical()
            .passes()
            .iter()
            .position(|pass| *pass == wanted)
            .ok_or(Error::InvalidState)?;
        Ok(StepId {
            plan: self.id,
            index: u32::try_from(index).map_err(|_| Error::InvalidRange)?,
        })
    }
    pub(crate) fn mip_steps(&self) -> impl ExactSizeIterator<Item = StepId> + '_ {
        (0..self.mip_passes.len()).map(|index| StepId {
            plan: self.id,
            index: 1 + u32::try_from(index).expect("mip pass index fits u32"),
        })
    }

    /// Rejects a frame whose structural shape cannot use this plan.
    ///
    /// This bounded initial plan declares exact byte ranges, so callers must
    /// include all declared ranges and layout facts in their cache key. Resource
    /// identity itself is deliberately absent from that key; `FrameBindings`
    /// validates each incoming owner independently.
    pub(crate) fn validate_frame(&self, frame: &ForwardFrame<B>) -> Result<()> {
        self.validate_frame_base(frame)?;
        self.validate_bindings(frame)
    }

    pub(crate) fn validate_hdr_frame(
        &self,
        frame: &ForwardFrame<B>,
        image: Option<&SurfaceImage<B>>,
    ) -> Result<()> {
        self.validate_frame_base(frame)?;
        let standard_msaa_surface = frame.frame.multisample_color.is_some()
            && image.is_some()
            && frame.frame.output.is_none()
            && frame.frame.target_desc.format == TextureFormat::Rgba8Unorm;
        if frame.frame.target_desc.format != TextureFormat::Rgba16Float && !standard_msaa_surface {
            return Err(Error::InvalidState);
        }
        match (self.hdr_terminal.as_ref(), image) {
            (Some(HdrForwardTerminal::Offscreen { .. }), None) => self.bindings(frame).map(|_| ()),
            (Some(HdrForwardTerminal::Surface { extent, format, .. }), Some(image))
                if image.is_current()
                    && image.device() == self.id.device
                    && image.extent() == *extent
                    && image.format() == *format =>
            {
                self.bindings_with_hdr_surface(frame, image).map(|_| ())
            }
            _ => Err(Error::InvalidState),
        }
    }
    pub(super) fn validate_frame_base(&self, frame: &ForwardFrame<B>) -> Result<()> {
        validate_resident_texture_devices(&frame.resident, self.id.device)?;
        if frame
            .frame
            .depth
            .as_ref()
            .is_some_and(|owner| owner.device != self.id.device)
        {
            return Err(Error::WrongDevice);
        }
        if frame.resident.device != self.id.device
            || frame
                .environment
                .as_ref()
                .is_some_and(|owner| owner.device != self.id.device)
            || frame
                .specular
                .as_ref()
                .is_some_and(|owner| owner.device != self.id.device)
            || frame.extent != self.extent
            || frame.draws.len() != self.draw_count
            || frame.resident.streams.len() != self.stream_count
            || frame.resident.textures.len() != self.texture_count
            || self.shape != forward_shape(frame)
            || frame
                .draws
                .iter()
                .any(|draw| validate_draw(draw, self.stream_count).is_err())
        {
            return Err(Error::InvalidState);
        }
        if let Some(environment) = frame.environment.as_ref() {
            environment.validate_for_sampling(self.id.device)?;
        }
        if let Some(specular) = frame.specular.as_ref() {
            specular.validate_for_sampling(self.id.device)?;
        }
        validate_forward_topology(self.topology, frame.draws.len(), frame.draw_batches.len())?;
        crate::residency::validate_indirect_draws(&frame.draws, &frame.indirect)?;
        crate::residency::validate_particle_draws(&frame.draws, &frame.particles, &frame.indirect)?;
        validate_uploads(frame)?;
        validate_batches(frame)?;
        for draw in &frame.draws {
            validate_offsets(draw, self.uniform_alignment)?;
            if draw.capture_textures.iter().any(|capture| *capture)
                || draw
                    .textures
                    .iter()
                    .any(|texture| *texture >= self.texture_count)
            {
                return Err(Error::InvalidRange);
            }
            draw_accesses(frame, draw)?;
        }
        Ok(())
    }

    pub(crate) fn validate_bindings(&self, frame: &ForwardFrame<B>) -> Result<()> {
        self.bindings(frame).map(|_| ())
    }

    pub(crate) fn completion(&self, outcome: &Outcome<B>) -> Result<CompletionPoint<B>>
    where
        B: ExecutionBackend,
    {
        if self.oit.as_ref().and_then(|oit| oit.surface).is_some()
            || matches!(self.hdr_terminal, Some(HdrForwardTerminal::Surface { .. }))
        {
            return Err(Error::InvalidState);
        }
        let expected: Vec<_> = (0..self.execution.logical().passes().len())
            .map(|index| StepId {
                plan: self.id,
                index: u32::try_from(index).expect("compiled pass index fits u32"),
            })
            .collect();
        let facts = outcome.facts();
        if facts.status != ExecutionStatus::Accepted || facts.accepted != Some(&expected) {
            return Err(Error::InvalidState);
        }
        facts
            .checkpoints
            .and_then(|points| {
                points
                    .iter()
                    .find(|point| {
                        point.point.device() == self.id.device
                            && point.covers.lane_frontiers.iter().any(|&(lane, step)| {
                                lane == self.lane && step == self.readback_step()
                            })
                    })
                    .map(|point| point.point.clone())
            })
            .ok_or(Error::InvalidState)
    }

    pub(crate) fn hdr_surface_completion(&self, outcome: &Outcome<B>) -> Result<CompletionPoint<B>>
    where
        B: ExecutionBackend,
    {
        let present = self.hdr_present_step()?;
        let mut expected: Vec<_> = (0..self.execution.logical().passes().len())
            .map(|index| StepId {
                plan: self.id,
                index: u32::try_from(index).expect("compiled pass index fits u32"),
            })
            .collect();
        expected.push(present);
        let facts = outcome.facts();
        if facts.status != ExecutionStatus::Accepted
            || facts.accepted != Some(expected.as_slice())
            || !facts.presents.is_some_and(|items| {
                items.len() == 1 && items[0].accepted && items[0].step == Some(present)
            })
        {
            return Err(Error::InvalidState);
        }
        facts
            .checkpoints
            .and_then(|points| {
                points
                    .iter()
                    .find(|point| {
                        point.point.device() == self.id.device
                            && point
                                .covers
                                .lane_frontiers
                                .iter()
                                .any(|&(lane, step)| lane == self.lane && step == present)
                    })
                    .map(|point| point.point.clone())
            })
            .ok_or(Error::InvalidState)
    }
}

#[cfg(test)]
#[path = "../../tests/graph/forward_offscreen_plan.rs"]
mod test_helpers;
