//! Frame-local OIT programs and the actual textures sampled by their groups.

use super::oit_program::{OitFullscreenPrograms, OitMaterialProgram};
use super::*;
use crate::{
    graph::{
        ResourceInstanceId,
        oit_record::{
            OitColorTargetRef, OitDrawPrograms, OitFullscreenCommand, OitRecordCommands,
            OitSampledGroupRef,
        },
    },
    preparation::forward_fragment::{OitDepthPrecision, OitDrawStage},
    residency::{ForwardFrame, OitOwners, OitPlan, OitTextureOwner},
};

/// Facts are captured while creating the actual bind group, independently of
/// Graph declarations. Recording compares them with its selected resource side.
pub(super) struct OitSampledGroup<B: ResourceBackend> {
    group: BindGroup<B>,
    instances: Vec<ResourceInstanceId>,
}

impl<B: ResourceBackend> OitSampledGroup<B> {
    pub(super) fn borrowed(&self) -> OitSampledGroupRef<'_, B> {
        OitSampledGroupRef {
            group: &self.group,
            textures: &self.instances,
        }
    }
}

pub(super) struct OitDrawResources<B: ResourceBackend> {
    pub(super) draw: usize,
    pub(super) program: OitMaterialProgram<B>,
    pub(super) initial_group: OitSampledGroup<B>,
    pub(super) peel_groups: [OitSampledGroup<B>; 2],
}

/// All additional native owners needed by OIT. The eventual pending frame must
/// retain this entire value alongside the ordinary Forward resource owners.
pub(crate) struct OitFrameResources<B: ResourceBackend> {
    pub(in crate::renderer) owners: OitOwners<B>,
    pub(in crate::renderer) output: OitTextureOwner<B>,
    pub(super) draws: Vec<OitDrawResources<B>>,
    pub(super) fullscreen: OitFullscreenPrograms<B>,
    pub(super) initialize_back: OitSampledGroup<B>,
    pub(super) accumulate_back: [OitSampledGroup<B>; 2],
    pub(super) final_composite: [OitSampledGroup<B>; 2],
}

impl<B: ResourceBackend> OitFrameResources<B> {
    pub(in crate::renderer) fn draw_commands(&self) -> Vec<OitDrawPrograms<'_, B>> {
        self.draws
            .iter()
            .map(|draw| OitDrawPrograms {
                draw: draw.draw,
                initial: &draw.program.initial,
                peel: &draw.program.peel,
                preliminary_initial: draw.program.preliminary_initial.as_ref(),
                preliminary_peel: draw.program.preliminary_peel.as_ref(),
                initial_group3: draw.initial_group.borrowed(),
                peel_group3: draw.peel_groups.each_ref().map(OitSampledGroup::borrowed),
                empty_group2: draw.program.empty_environment_group.as_ref(),
            })
            .collect()
    }

    pub(in crate::renderer) fn commands_for_opaque<'a>(
        &'a self,
        frame: &'a ForwardFrame<B>,
        draws: &'a [OitDrawPrograms<'a, B>],
        opaque: &'a crate::residency::ForwardTexture<B>,
    ) -> OitRecordCommands<'a, B> {
        let mut commands = self.commands(frame, draws);
        commands.opaque_color = OitColorTargetRef {
            texture: &opaque.texture,
            view: &opaque.view,
            desc: &opaque.desc,
            instance: opaque.instance,
            device: opaque.device,
        };
        commands
    }

    pub(in crate::renderer) fn commands<'a>(
        &'a self,
        frame: &'a ForwardFrame<B>,
        draws: &'a [OitDrawPrograms<'a, B>],
    ) -> OitRecordCommands<'a, B> {
        OitRecordCommands {
            draws,
            init_back: OitFullscreenCommand {
                pipeline: &self.fullscreen.initialize_back.pipeline,
                bind_group: self.initialize_back.borrowed(),
            },
            back_accumulate: self
                .accumulate_back
                .each_ref()
                .map(|group| OitFullscreenCommand {
                    pipeline: &self.fullscreen.accumulate_back.pipeline,
                    bind_group: group.borrowed(),
                }),
            final_composite: self
                .final_composite
                .each_ref()
                .map(|group| OitFullscreenCommand {
                    pipeline: &self.fullscreen.final_composite.pipeline,
                    bind_group: group.borrowed(),
                }),
            opaque_color: OitColorTargetRef {
                texture: &frame.frame.target,
                view: &frame.frame.target_view,
                desc: &frame.frame.target_desc,
                instance: frame.frame.target_instance,
                device: frame.resident.device,
            },
            output: OitColorTargetRef {
                texture: &self.output.texture,
                view: &self.output.view,
                desc: &self.output.desc,
                instance: self.output.instance,
                device: self.output.device,
            },
        }
    }
}

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
    /// Reuses an assembled frame's real groups 0--2. Candidate indices refer
    /// to that same frame and its retained, identically ordered logical inputs.
    /// This creates resources only; it neither declares nor submits a Graph.
    pub(in crate::renderer) async fn assemble_oit_resources(
        &mut self,
        frame: &ForwardFrame<B>,
        inputs: &[crate::ForwardDrawInput],
        candidates: &[usize],
        composer: &mut fluxel_shader::ShaderComposer,
    ) -> std::result::Result<OitFrameResources<B>, RendererError> {
        self.ensure_open()?;
        validate_frame_inputs(frame, inputs, candidates, self.device.id())?;
        let plan = OitPlan::admit(&self.device, frame.extent)?;
        let precision = match plan.depth_precision {
            crate::residency::OitDepthPrecision::Float32 => OitDepthPrecision::Rg32Float,
            crate::residency::OitDepthPrecision::Float16 => OitDepthPrecision::Rg16Float,
        };
        let output_desc = output_descriptor(frame);
        admit_output(&self.device, &output_desc)?;
        // Finish logical composition for every candidate before creating its
        // native programs or allocating the OIT targets.
        let prepared = candidates
            .iter()
            .map(|&index| {
                let input = &inputs[index];
                Ok((
                    index,
                    crate::preparation::prepare_oit_material(
                        input,
                        OitDrawStage::InitialDepth,
                        precision,
                        composer,
                    )?,
                    crate::preparation::prepare_oit_material(
                        input,
                        OitDrawStage::Peel,
                        precision,
                        composer,
                    )?,
                ))
            })
            .collect::<std::result::Result<Vec<_>, crate::ForwardPreparationError>>()
            .map_err(RendererError::ForwardPreparation)?;
        let mut programs = Vec::with_capacity(prepared.len());
        for (index, initial, peel) in prepared {
            let input = &inputs[index];
            let base = self
                .forward_pipeline(
                    input.prepared(),
                    frame.frame.target_desc.format,
                    input.transform_uniform_bytes(),
                    crate::ForwardDepthMode::Depth32LessEqualWrite,
                    1,
                    crate::renderer::raster_surface_state(input),
                )
                .await?;
            let program = self
                .oit_material_program(
                    input.prepared(),
                    &base,
                    &initial,
                    &peel,
                    plan.min_max_desc.format,
                    input.transform_uniform_bytes(),
                )
                .await?;
            programs.push((index, program));
        }
        let fullscreen = self.oit_fullscreen_programs(output_desc.format).await?;
        let owners = OitOwners::allocate(&self.device, plan)
            .map_err(|error| self.observe_forward_error("OIT intermediate allocation", error))?;
        let output = allocate_output(&self.device, output_desc)
            .map_err(|error| self.observe_forward_error("OIT output allocation", error))?;
        let mut draws = Vec::with_capacity(programs.len());
        for (draw, program) in programs {
            let initial_group = sampled_group(
                &self.device,
                &program.initial_layout,
                &[(
                    &owners.ping_pong[1].min_max.view,
                    owners.ping_pong[1].min_max.instance,
                )],
            )
            .map_err(|error| self.observe_forward_error("OIT initial group", error))?;
            let peel_groups = make_pair(|side| {
                let owner = &owners.ping_pong[side];
                sampled_group(
                    &self.device,
                    &program.peel_layout,
                    &[
                        (&owner.min_max.view, owner.min_max.instance),
                        (&owner.front.view, owner.front.instance),
                    ],
                )
            })
            .map_err(|error| self.observe_forward_error("OIT peel groups", error))?;
            draws.push(OitDrawResources {
                draw,
                program,
                initial_group,
                peel_groups,
            });
        }
        let initialize_back = sampled_group(
            &self.device,
            &fullscreen.initialize_back.layout,
            &[(&frame.frame.target_view, frame.frame.target_instance)],
        )
        .map_err(|error| {
            self.observe_forward_error("OIT accumulator initialization group", error)
        })?;
        let accumulate_back = make_pair(|side| {
            let owner = &owners.ping_pong[side].back;
            sampled_group(
                &self.device,
                &fullscreen.accumulate_back.layout,
                &[(&owner.view, owner.instance)],
            )
        })
        .map_err(|error| self.observe_forward_error("OIT back accumulation groups", error))?;
        let final_composite = make_pair(|side| {
            let owner = &owners.ping_pong[side].front;
            sampled_group(
                &self.device,
                &fullscreen.final_composite.layout,
                &[
                    (&owner.view, owner.instance),
                    (
                        &owners.back_accumulator.view,
                        owners.back_accumulator.instance,
                    ),
                ],
            )
        })
        .map_err(|error| self.observe_forward_error("OIT final composite groups", error))?;
        Ok(OitFrameResources {
            owners,
            output,
            draws,
            fullscreen,
            initialize_back,
            accumulate_back,
            final_composite,
        })
    }
}

fn validate_frame_inputs<B: ResourceBackend>(
    frame: &ForwardFrame<B>,
    inputs: &[crate::ForwardDrawInput],
    candidates: &[usize],
    device: DeviceId,
) -> Result<()> {
    if frame.resident.device != device
        || frame.frame.target.device() != device
        || frame.frame.target_view.device() != device
    {
        return Err(Error::WrongDevice);
    }
    if inputs.len() != frame.draws.len()
        || !frame
            .frame
            .target_desc
            .usage
            .contains(TextureUsage::TEXTURE_BINDING)
        || frame.frame.target_desc.extent != [frame.extent[0], frame.extent[1], 1]
        || frame.frame.target_desc.dimension != TextureDimension::D2
        || frame.frame.target_desc.array_layers != 1
        || frame.frame.target_desc.mip_levels != 1
        || frame.frame.target_desc.samples != 1
        || !frame
            .frame
            .target_desc
            .usage
            .contains(TextureUsage::RENDER_ATTACHMENT)
    {
        return Err(Error::InvalidState);
    }
    let depth = frame.frame.depth.as_ref().ok_or(Error::InvalidState)?;
    if depth.device != device || depth.texture.device() != device || depth.view.device() != device {
        return Err(Error::WrongDevice);
    }
    if depth.desc.extent != [frame.extent[0], frame.extent[1], 1]
        || depth.desc.format != TextureFormat::Depth32Float
        || depth.desc.samples != 1
        || depth.desc.mip_levels != 1
        || depth.desc.array_layers != 1
        || depth.desc.dimension != TextureDimension::D2
        || !depth.desc.usage.contains(TextureUsage::RENDER_ATTACHMENT)
    {
        return Err(Error::InvalidState);
    }
    for (position, &index) in candidates.iter().enumerate() {
        let input = inputs.get(index).ok_or(Error::InvalidRange)?;
        if candidates[..position].contains(&index)
            || !matches!(
                input.effective_alpha_mode(),
                fluxel_material::SurfaceAlphaMode::Blend
                    | fluxel_material::SurfaceAlphaMode::MaskBlend
            )
            || frame.draws[index].compatibility
                != crate::preparation::ForwardCompatibilityKey::from_draw(input)
        {
            return Err(Error::InvalidState);
        }
    }
    Ok(())
}

fn sampled_group<B: ResourceBackend>(
    device: &Device<B>,
    layout: &BindGroupLayout<B>,
    textures: &[(&TextureView<B>, ResourceInstanceId)],
) -> Result<OitSampledGroup<B>> {
    let entries = textures
        .iter()
        .enumerate()
        .map(|(index, (view, _))| {
            Ok(BindGroupEntry {
                binding: u32::try_from(index).map_err(|_| Error::InvalidRange)?,
                resource: BindingResource::TextureView(view),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let group = device.create_bind_group(&BindGroupDesc {
        label: Some("OIT sampled owners"),
        layout,
        entries: &entries,
    })?;
    Ok(OitSampledGroup {
        group,
        instances: textures.iter().map(|(_, id)| *id).collect(),
    })
}

fn make_pair<T>(mut create: impl FnMut(usize) -> Result<T>) -> Result<[T; 2]> {
    Ok([create(0)?, create(1)?])
}

fn output_descriptor<B: ResourceBackend>(frame: &ForwardFrame<B>) -> TextureDesc {
    let mut desc = frame.frame.target_desc.clone();
    desc.label = Some("OIT final output".into());
    desc.usage =
        TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC | TextureUsage::TEXTURE_BINDING;
    desc
}

fn admit_output<B: ResourceBackend>(device: &Device<B>, desc: &TextureDesc) -> Result<()> {
    match device.texture_support(desc) {
        Support::Supported(support)
            if support
                .max_extent
                .iter()
                .zip(desc.extent)
                .all(|(&max, actual)| max >= actual)
                && support.max_array_layers >= desc.array_layers
                && support.max_mips >= desc.mip_levels
                && support.sample_counts.contains(&desc.samples) =>
        {
            Ok(())
        }
        Support::Supported(_) => Err(Error::Unsupported("OIT final output descriptor")),
        Support::Unsupported(reason) => Err(Error::Unsupported(reason)),
    }
}

fn allocate_output<B: ResourceBackend>(
    device: &Device<B>,
    desc: TextureDesc,
) -> Result<OitTextureOwner<B>> {
    let view_desc = TextureViewDesc {
        format: desc.format,
        dimension: TextureViewDimension::D2,
        aspects: TextureAspects::COLOR,
        mips: 0..1,
        layers: 0..1,
    };
    let texture = device.create_texture(&desc)?;
    let view = device.create_view(&texture, &view_desc)?;
    Ok(OitTextureOwner {
        texture,
        view,
        view_desc,
        desc,
        instance: ResourceInstanceId::fresh().map_err(|_| Error::InvalidState)?,
        incoming: TextureUses {
            stages: StageMask::EMPTY,
            accesses: AccessMask::EMPTY,
        },
        layout: ImageLayout::Undefined,
        device: device.id(),
    })
}
