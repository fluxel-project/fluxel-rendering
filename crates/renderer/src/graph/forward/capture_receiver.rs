//! A Forward capture receiver appended to an already-declared producer Graph.
//!
//! This is deliberately a namespace, not a plan: its caller owns compilation,
//! submission and the producer's actual native attachment owners.

use super::capture_declaration::{
    CaptureFrameSlots, FrameHandles, SharedResidentImports, capture_texture_slots,
    declare_mip_generations, declare_upload, import_frame_inner, validate_spec,
};
use super::*;
use crate::graph::definition::TextureHandle;
use crate::graph::oit_record::{OitDepthTargetRef, OitDrawSlots, OitRecordCommands};

/// Receiver passes and exact final Graph version appended after a producer.
pub(in crate::graph) struct ForwardReceiverComposition {
    pub(in crate::graph) upload: GraphPass,
    pub(in crate::graph) mip_generations: Vec<super::super::mip_generation::MipGenerationPass>,
    pub(in crate::graph) raster: GraphPass,
    oit: Option<super::oit_plan::ForwardOitPlan>,
    pub(in crate::graph) final_slot: TextureSlot,
    pub(in crate::graph) final_handle: TextureHandle,
    slots: CaptureFrameSlots,
    handles: FrameHandles,
    spec: ForwardGraphSpec,
    pub(in crate::graph) shape: ForwardShape,
    pub(in crate::graph) alignment: u64,
}

/// Terminal selected by the enclosing combined Graph, never a second plan.
pub(in crate::graph) enum ForwardReceiverTerminal {
    Offscreen {
        pass: GraphPass,
    },
    Surface {
        pass: GraphPass,
        slot: TextureSlot,
        extent: [u32; 2],
        format: TextureFormat,
    },
}

pub(in crate::graph) fn declare_receiver_terminal<B: ResourceBackend>(
    graph: &mut GraphBuilder,
    receiver: &ForwardReceiverComposition,
    image: Option<&SurfaceImage<B>>,
) -> std::result::Result<ForwardReceiverTerminal, ForwardPlanBuildError> {
    match image {
        None => {
            let length = u64::from(receiver.spec.readback_row_pitch)
                .checked_mul(u64::from(receiver.spec.extent[1]))
                .ok_or(Error::InvalidRange)?;
            let (pass, ()) = graph
                .add_pass("deferred-forward-capture-readback", |pass| {
                    pass.read_texture_with_use(
                        receiver.final_handle,
                        whole(),
                        texture_copy_read_uses().uses,
                        ImageLayout::CopySrc,
                        TextureUsage::COPY_SRC,
                    )?;
                    pass.write_with_use(
                        receiver.handles.readback,
                        0..length,
                        copy_write_uses().uses,
                        BufferUsage::COPY_DST,
                    )?;
                    pass.side_effect();
                    Ok(())
                })
                .map_err(|_| Error::InvalidState)?;
            Ok(ForwardReceiverTerminal::Offscreen { pass })
        }
        Some(image) => {
            if image.extent() != receiver.spec.extent {
                return Err(Error::InvalidRange.into());
            }
            let surface = graph
                .import_surface_texture(TextureDesc {
                    label: Some("fluxel deferred forward capture surface lease".into()),
                    dimension: fluxel_rhi::api::TextureDimension::D2,
                    extent: [receiver.spec.extent[0], receiver.spec.extent[1], 1],
                    array_layers: 1,
                    format: image.format(),
                    mip_levels: 1,
                    samples: 1,
                    usage: TextureUsage::RENDER_ATTACHMENT,
                    view_formats: Vec::new(),
                })
                .map_err(|_| Error::InvalidState)?;
            let (pass, ()) = graph
                .add_pass("deferred-forward-capture-surface-composite", |pass| {
                    pass.read_texture_with_use(
                        receiver.final_handle,
                        whole(),
                        texture_read_uses().uses,
                        ImageLayout::ShaderReadOnly,
                        TextureUsage::TEXTURE_BINDING,
                    )?;
                    pass.write_texture_with_use(
                        surface,
                        whole(),
                        color_write_uses().uses,
                        ImageLayout::ColorAttachment,
                        TextureUsage::RENDER_ATTACHMENT,
                    )?;
                    pass.side_effect();
                    Ok(())
                })
                .map_err(|_| Error::InvalidState)?;
            Ok(ForwardReceiverTerminal::Surface {
                pass,
                slot: surface.into(),
                extent: receiver.spec.extent,
                format: image.format(),
            })
        }
    }
}

/// Appends a receiver to the producer's real Graph product.
///
/// Every logical capture texture aliases `producer_final`; the producer is
/// never re-imported, copied, or represented by a synthetic frame.  The caller
/// must bind that exact final version alongside this namespace's bindings.
#[allow(clippy::too_many_arguments)]
pub(in crate::graph) fn declare_forward_capture_namespace<B>(
    device: &Device<B>,
    graph: &mut GraphBuilder,
    frame: &ForwardFrame<B>,
    capture_source: Option<TextureHandle>,
    oit_input: Option<&ForwardOitInput<'_, B>>,
    shared: &mut SharedResidentImports,
) -> std::result::Result<ForwardCaptureReceiver, ForwardPlanBuildError>
where
    B: ResourceBackend,
{
    validate_resident_texture_devices(&frame.resident, device.id())?;
    let spec = ForwardGraphSpec::from_frame(frame)?;
    if spec.device != device.id() || frame.extent.contains(&0) {
        return Err(Error::WrongDevice.into());
    }
    let receiver = capture_source.is_some();
    validate_spec(device, &spec, FrameTopology::SortedForward, receiver)?;
    super::capture_oit_declaration::validate_oit_input(device, frame, &spec, oit_input)?;
    let aliases = capture_source
        .map(|source| {
            capture_texture_slots(&spec).map(|slots| {
                slots
                    .iter()
                    .map(|capture| capture.then_some(source))
                    .collect::<Vec<_>>()
            })
        })
        .transpose()?;
    let (slots, handles) = import_frame_inner(graph, &spec, aliases.as_deref(), shared, None)?;
    let (upload, handles) = declare_upload(graph, &spec, handles)?;
    let (handles, mip_generations) = declare_mip_generations(graph, &spec, handles, shared)?;
    let dependency = mip_generations
        .last()
        .map_or(upload, |generation| generation.pass);
    let (raster, final_handle, oit) = super::capture_oit_declaration::declare_side(
        graph, &spec, &handles, upload, dependency, oit_input,
    )?;
    Ok(ForwardReceiverComposition {
        upload,
        mip_generations,
        raster,
        oit,
        final_slot: final_handle.into(),
        final_handle,
        slots,
        handles,
        shape: spec.shape.clone(),
        spec,
        alignment: device.capabilities().limits.uniform_offset_alignment,
    })
}

/// Appends a Forward capture receiver to an existing producer namespace.
pub(in crate::graph) fn declare_receiver_after_deferred<B>(
    device: &Device<B>,
    graph: &mut GraphBuilder,
    receiver: &ForwardFrame<B>,
    producer_final: TextureHandle,
    receiver_oit: Option<&ForwardOitInput<'_, B>>,
    shared: &mut SharedResidentImports,
) -> std::result::Result<ForwardCaptureReceiver, ForwardPlanBuildError>
where
    B: ResourceBackend,
{
    declare_forward_capture_namespace(
        device,
        graph,
        receiver,
        Some(producer_final),
        receiver_oit,
        shared,
    )
}

/// Compatibility name for call sites which treat the receiver as a namespace.
pub(in crate::graph) type ForwardCaptureReceiver = ForwardReceiverComposition;

impl ForwardReceiverComposition {
    pub(in crate::graph) fn append_passes(&self, passes: &mut Vec<GraphPass>) {
        passes.push(self.upload);
        passes.extend(
            self.mip_generations
                .iter()
                .map(|generation| generation.pass),
        );
        passes.push(self.raster);
        if let Some(oit) = &self.oit {
            passes.extend(oit.passes());
            if let Some(pass) = oit.fallback {
                passes.push(pass);
            }
        }
    }

    pub(in crate::graph) fn final_slot(&self) -> TextureSlot {
        self.final_slot
    }

    pub(in crate::graph) fn validate_frame<B: ResourceBackend>(
        &self,
        device: DeviceId,
        frame: &ForwardFrame<B>,
    ) -> Result<()> {
        validate_resident_texture_devices(&frame.resident, device)?;
        if frame.resident.device != device || forward_shape(frame) != self.shape {
            return Err(Error::InvalidState);
        }
        validate_forward_topology(
            FrameTopology::SortedForward,
            frame.draws.len(),
            frame.draw_batches.len(),
        )?;
        validate_uploads(frame)?;
        validate_batches(frame)?;
        for draw in &frame.draws {
            validate_draw(draw, frame.resident.streams.len())?;
            validate_offsets(draw, self.alignment)?;
            draw_accesses(frame, draw)?;
            if draw.pipeline.device() != device
                || draw
                    .bind_groups
                    .iter()
                    .any(|group| group.group.device() != device)
            {
                return Err(Error::WrongDevice);
            }
        }
        if let Some(environment) = frame.environment.as_ref() {
            environment.validate_for_sampling(device)?;
        }
        if let Some(specular) = frame.specular.as_ref() {
            specular.validate_for_sampling(device)?;
        }
        Ok(())
    }

    /// Adds only owners introduced by this namespace. Capture aliases are
    /// intentionally omitted: the caller binds the exact producer final owner.
    pub(in crate::graph) fn append_bindings<'a, B: ResourceBackend>(
        &self,
        execution: &CompiledExecution,
        frame: &'a ForwardFrame<B>,
        buffers: &mut Vec<BoundBuffer<'a, B>>,
        textures: &mut Vec<BoundTexture<'a, B>>,
    ) -> Result<()> {
        self.validate_frame(frame.resident.device, frame)?;
        let required = |slot| {
            execution
                .logical()
                .buffer_bindings()
                .requirements()
                .iter()
                .any(|r| r.slot == slot)
        };
        let mut buffer = |slot, instance, owner, desc, incoming| {
            if required(slot) {
                buffers.push(BoundBuffer::new(slot, instance, owner, desc, incoming));
            }
        };
        let s = &self.slots;
        buffer(
            s.resident_upload,
            frame.resident.upload_instance,
            &frame.resident.upload,
            &frame.resident.upload_desc,
            frame.resident.upload_incoming,
        );
        buffer(
            s.frame_upload,
            frame.frame.upload_instance,
            &frame.frame.upload,
            &frame.frame.upload_desc,
            frame.frame.upload_incoming,
        );
        buffer(
            s.uniform,
            frame.frame.uniform_instance,
            &frame.frame.uniform,
            &frame.frame.uniform_desc,
            frame.frame.uniform_incoming,
        );
        buffer(
            s.readback,
            frame.frame.readback_instance,
            &frame.frame.readback,
            &frame.frame.readback_desc,
            frame.frame.readback_incoming,
        );
        for (&slot, owner) in s.streams.iter().zip(&frame.resident.streams) {
            buffer(
                slot,
                owner.instance,
                &owner.buffer,
                &owner.desc,
                owner.incoming,
            );
        }
        if let (Some(slot), Some(owner)) = (s.environment_upload, frame.environment.as_ref()) {
            buffer(
                slot,
                owner.upload_instance,
                &owner.upload,
                &owner.upload_desc,
                owner.upload_incoming,
            );
        }
        let capture_slots = capture_texture_slots(&self.spec)?;
        for ((&slot, owner), capture) in s
            .textures
            .iter()
            .zip(&frame.resident.textures)
            .zip(&capture_slots)
        {
            if !*capture {
                textures.push(BoundTexture::new(
                    slot,
                    owner.instance,
                    &owner.texture,
                    &owner.desc,
                    owner.incoming,
                    owner.layout,
                ));
            }
        }
        if let (Some(slot), Some(owner)) = (s.environment, frame.environment.as_ref()) {
            textures.push(BoundTexture::new(
                slot,
                owner.texture_instance,
                &owner.texture,
                &owner.desc,
                owner.incoming,
                owner.layout,
            ));
        }
        textures.push(BoundTexture::new(
            s.target,
            frame.frame.target_instance,
            &frame.frame.target,
            &frame.frame.target_desc,
            frame.frame.target_incoming,
            frame.frame.target_layout,
        ));
        if let (Some(slot), Some(depth)) = (s.depth, frame.frame.depth.as_ref()) {
            textures.push(BoundTexture::new(
                slot,
                depth.instance,
                &depth.texture,
                &depth.desc,
                depth.incoming,
                depth.layout,
            ));
        }
        super::capture_specular_bindings::append(
            s,
            frame.specular.as_ref(),
            frame.resident.device,
            buffers,
            textures,
        )
    }

    /// Adds the eight exact OIT owners (seven peel intermediates plus final
    /// output) required by the OIT Graph slots. The combined-plan owner
    /// performs cross-namespace canonical de-duplication afterwards.
    pub(in crate::graph) fn append_oit_bindings<'a, B: ResourceBackend>(
        &self,
        input: Option<&'a ForwardOitInput<'a, B>>,
        textures: &mut Vec<BoundTexture<'a, B>>,
    ) -> Result<()> {
        let (oit, input) = match (&self.oit, input) {
            (None, None) => return Ok(()),
            (Some(oit), Some(input)) => (oit, input),
            _ => return Err(Error::InvalidState),
        };
        input.owners.validate(input.output.device)?;
        let owners = &input.owners;
        let entries = [
            (oit.slots.ping_pong[0].min_max, &owners.ping_pong[0].min_max),
            (oit.slots.ping_pong[0].front, &owners.ping_pong[0].front),
            (oit.slots.ping_pong[0].back, &owners.ping_pong[0].back),
            (oit.slots.ping_pong[1].min_max, &owners.ping_pong[1].min_max),
            (oit.slots.ping_pong[1].front, &owners.ping_pong[1].front),
            (oit.slots.ping_pong[1].back, &owners.ping_pong[1].back),
            (oit.slots.back_accumulator, &owners.back_accumulator),
            (oit.slots.output, input.output),
        ];
        for (slot, owner) in entries {
            textures.push(BoundTexture::new(
                slot,
                owner.instance,
                &owner.texture,
                &owner.desc,
                owner.incoming,
                owner.layout,
            ));
        }
        Ok(())
    }

    /// Revalidates only owner and declared OIT facts while staging a plan.
    pub(in crate::graph) fn validate_oit_input<B: ResourceBackend>(
        &self,
        device: DeviceId,
        frame: &ForwardFrame<B>,
        input: Option<&ForwardOitInput<'_, B>>,
    ) -> Result<()> {
        match (&self.oit, input) {
            (None, None) => Ok(()),
            (Some(oit), Some(input)) => {
                input.owners.validate(device)?;
                let end = input
                    .opaque_count
                    .checked_add(input.peel_count)
                    .ok_or(Error::InvalidRange)?;
                if frame.resident.device != device
                    || input.output.device != device
                    || input.output.texture.device() != device
                    || input.output.view.device() != device
                    || input.owners.extent != frame.extent
                    || end > frame.draws.len()
                    || input.peel_count == 0
                    || oit.opaque_count != input.opaque_count
                    || oit.peel_count != input.peel_count
                    || oit.rounds != input.rounds
                {
                    Err(Error::InvalidState)
                } else {
                    Ok(())
                }
            }
            _ => Err(Error::InvalidState),
        }
    }

    /// Validates the native OIT command set before any receiver pass records.
    /// Pointer checks intentionally use the original Forward frame owners:
    /// cloned RHI wrappers are not evidence for the OIT opaque target.
    pub(in crate::graph) fn validate_oit_input_commands<B: ResourceBackend>(
        &self,
        device: DeviceId,
        frame: &ForwardFrame<B>,
        input: Option<&ForwardOitInput<'_, B>>,
        commands: Option<&OitRecordCommands<'_, B>>,
    ) -> Result<()> {
        let (oit, input, commands) = match (&self.oit, input, commands) {
            (None, None, None) => return Ok(()),
            (Some(oit), Some(input), Some(commands)) => (oit, input, commands),
            _ => return Err(Error::InvalidState),
        };
        self.validate_oit_input(device, frame, Some(input))?;
        let end = oit
            .opaque_count
            .checked_add(oit.peel_count)
            .ok_or(Error::InvalidRange)?;
        if input.output.device != device
            || input.output.texture.device() != device
            || input.output.view.device() != device
            || frame.frame.target.device() != device
            || frame.frame.target_view.device() != device
            || end > frame.draws.len()
            || commands.draws.len() != oit.peel_count
            || commands
                .draws
                .iter()
                .enumerate()
                .any(|(offset, draw)| draw.draw != oit.opaque_count + offset)
            || commands.opaque_color.instance != frame.frame.target_instance
            || !std::ptr::eq(commands.opaque_color.texture, &frame.frame.target)
            || !std::ptr::eq(commands.opaque_color.view, &frame.frame.target_view)
            || commands.output.instance != input.output.instance
            || !std::ptr::eq(commands.output.texture, &input.output.texture)
            || !std::ptr::eq(commands.output.view, &input.output.view)
        {
            return Err(Error::InvalidState);
        }
        if commands.draws.iter().any(|draw| {
            draw.initial.device() != device
                || draw.peel.device() != device
                || draw.initial_group3.group.device() != device
                || draw
                    .peel_group3
                    .iter()
                    .any(|group| group.group.device() != device)
                || draw
                    .empty_group2
                    .is_some_and(|group| group.device() != device)
        }) || std::iter::once(&commands.init_back)
            .chain(commands.back_accumulate.iter())
            .chain(commands.final_composite.iter())
            .any(|command| {
                command.pipeline.device() != device || command.bind_group.group.device() != device
            })
        {
            return Err(Error::WrongDevice);
        }
        Ok(())
    }
}

#[path = "capture_receiver_record.rs"]
mod record;
