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
        Self::compile_with_composition(device, id, declaration, &[])
    }

    /// Compiles the existing Deferred Graph with real composition passes
    /// authored between lighting and the final scene-color composite.
    pub(super) fn compile_with_composition(
        device: &Device<B>,
        id: PlanId,
        declaration: DeferredDeclaration,
        composition_passes: &[GraphPass],
    ) -> Result<Self> {
        if id.device != device.id() || declaration.lane.device != device.id() {
            return Err(Error::WrongDevice);
        }
        let execution = CompiledExecution::compile(&declaration.graph)?;
        let mut expected = vec![declaration.upload_pass];
        expected.extend(
            declaration
                .mip_generation
                .iter()
                .map(|generation| generation.pass),
        );
        for (_, indirect) in &declaration.indirect {
            expected.push(indirect.upload_pass);
            expected.push(indirect.compute_pass);
        }
        if let Some(pass) = declaration.deformation_pass {
            expected.push(pass);
        }
        expected.push(declaration.geometry_pass);
        if let Some(pass) = declaration.resolve_pass {
            expected.push(pass);
        }
        expected.push(declaration.lighting_pass);
        expected.extend_from_slice(composition_passes);
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
            DeferredOutputDeclaration::Capture => return Err(Error::InvalidState),
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
            mip_generation: declaration.mip_generation,
            indirect: declaration.indirect,
            deformation_pass: declaration.deformation_pass,
            geometry_pass: declaration.geometry_pass,
            resolve_pass: declaration.resolve_pass,
            lighting_pass: declaration.lighting_pass,
            composition_passes: composition_passes.to_vec(),
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
            msaa_gbuffer_slots: declaration.msaa_gbuffer_slots,
            msaa_effects_slot: declaration.msaa_effects_slot,
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
            || self
                .mip_generation
                .iter()
                .map(|generation| {
                    (
                        generation.pass.1,
                        generation.texture,
                        generation.source_mip,
                        generation.destination_mip,
                    )
                })
                .ne(declaration.mip_generation.iter().map(|generation| {
                    (
                        generation.pass.1,
                        generation.texture,
                        generation.source_mip,
                        generation.destination_mip,
                    )
                }))
            || self.deformation_pass.map(|pass| pass.1)
                != declaration.deformation_pass.map(|pass| pass.1)
            || self.indirect.len() != declaration.indirect.len()
            || self.indirect.iter().zip(&declaration.indirect).any(
                |((left_draw, left), (right_draw, right))| {
                    left_draw != right_draw
                        || left.upload_pass.1 != right.upload_pass.1
                        || left.compute_pass.1 != right.compute_pass.1
                },
            )
            || self.geometry_pass.1 != declaration.geometry_pass.1
            || self.resolve_pass.map(|pass| pass.1) != declaration.resolve_pass.map(|pass| pass.1)
            || self.lighting_pass.1 != declaration.lighting_pass.1
            || !self.composition_passes.is_empty()
            || self.composite_pass.map(|pass| pass.1)
                != declaration.composite_pass.map(|pass| pass.1)
            || !match (self.output, declaration.output) {
                (DeferredOutputPlan::Capture, DeferredOutputDeclaration::Capture) => true,
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
            || self
                .msaa_gbuffer_slots
                .map(|items| items.map(|slot| slot.resource))
                != declaration
                    .msaa_gbuffer_slots
                    .map(|items| items.map(|slot| slot.resource))
            || self.msaa_effects_slot.map(|slot| slot.resource)
                != declaration.msaa_effects_slot.map(|slot| slot.resource)
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
            index: u32::try_from(self.geometry_pass.1).expect("graph pass index fits StepId"),
        }
    }

    pub(crate) fn deformation_step(&self) -> Option<StepId> {
        self.deformation_pass.map(|pass| StepId {
            plan: self.id,
            index: u32::try_from(pass.1).expect("graph pass index fits StepId"),
        })
    }

    pub(crate) fn lighting_step(&self) -> StepId {
        StepId {
            plan: self.id,
            index: u32::try_from(self.lighting_pass.1).expect("graph pass index fits StepId"),
        }
    }

    pub(crate) fn readback_step(&self) -> StepId {
        match self.output {
            DeferredOutputPlan::Capture => self.lighting_step(),
            DeferredOutputPlan::Offscreen { readback_pass } => StepId {
                plan: self.id,
                index: u32::try_from(readback_pass.1).expect("graph pass index fits StepId"),
            },
            DeferredOutputPlan::Surface { .. } => self.lighting_step(),
        }
    }

    pub(crate) fn composite_step(&self) -> StepId {
        let pass = self
            .composite_pass
            .expect("compiled Deferred has composite pass");
        StepId {
            plan: self.id,
            index: u32::try_from(pass.1).expect("graph pass index fits StepId"),
        }
    }

    pub(crate) fn present_step(&self) -> Result<StepId> {
        match self.output {
            DeferredOutputPlan::Surface { present_step } => Ok(present_step),
            DeferredOutputPlan::Capture | DeferredOutputPlan::Offscreen { .. } => {
                Err(Error::InvalidState)
            }
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
        self.validate_frame_facts(frame)?;
        self.bindings(frame).map(|_| ())
    }

    /// Validates owner and descriptor facts without constructing bindings.
    /// Combined capture validates both namespaces before binding their single
    /// compiled Graph at the recording boundary.
    pub(super) fn validate_frame_facts(&self, frame: &DeferredFrame<B>) -> Result<()> {
        super::validate_resident_texture_devices(&frame.resident, self.id.device)?;
        if frame.draws.is_empty()
            || frame.resident.device != self.id.device
            || frame.draws.iter().any(|draw| {
                !draw.pipelines_on(self.id.device)
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
        if self.indirect.len() != frame.indirect.len()
            || self
                .indirect
                .iter()
                .zip(&frame.indirect)
                .any(|((draw, _), (product_draw, _))| draw != product_draw)
        {
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
                crate::residency::deferred_albedo_format(frame.pbr),
            )
            || !matches_gbuffer_template(
                &frame.gbuffer[1].desc,
                self.extent,
                crate::residency::deferred_normal_format(frame.pbr),
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
            || self.msaa_gbuffer_slots.is_some() != frame.msaa.is_some()
            || self.msaa_effects_slot.is_some()
                != frame
                    .msaa
                    .as_ref()
                    .and_then(|msaa| msaa.effects.as_ref())
                    .is_some()
            || (!matches!(self.output, DeferredOutputPlan::Capture)
                && !matches_final_target_template(&frame.frame.target_desc, self.extent))
            || !super::declaration_validation::matches_scene_color_template(
                &frame.scene_color.desc,
                self.extent,
            )
            || (!matches!(self.output, DeferredOutputPlan::Capture)
                && self.composite_pass.is_none())
            || self.scene_color_slot.is_none()
        {
            return Err(Error::InvalidState);
        }
        if let Some(msaa) = &frame.msaa {
            if msaa.pipeline.device() != self.id.device
                || msaa.group.device() != self.id.device
                || msaa
                    .gbuffer
                    .iter()
                    .zip(&frame.gbuffer)
                    .any(|(source, destination)| {
                        source.texture.device() != self.id.device
                            || source.view.device() != self.id.device
                            || source.desc.samples <= 1
                            || source.desc.extent != [self.extent[0], self.extent[1], 1]
                            || source.desc.format != destination.desc.format
                            || destination.desc.samples != 1
                    })
                || msaa
                    .effects
                    .as_ref()
                    .zip(frame.effects.as_ref())
                    .is_some_and(|(source, destination)| {
                        source.desc.format != destination.desc.format
                            || source.desc.extent != destination.desc.extent
                            || source.desc.samples != msaa.gbuffer[0].desc.samples
                            || destination.desc.samples != 1
                    })
            {
                return Err(Error::InvalidState);
            }
        }
        validate_draw_uniforms(frame, self.uniform_alignment)?;
        validate_uploads(frame)?;
        for draw in &frame.draws {
            draw_accesses(frame, draw)?;
        }
        Ok(())
    }
}

/// Read-only diagnostic evidence projected from the compiled graph.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DeferredGraphFacts {
    pub(crate) pass_count: usize,
    pub(crate) buffer_binding_count: usize,
    pub(crate) texture_binding_count: usize,
}
