//! Shared actual-owner binding collection for the two Forward capture endings.
//!
//! Capture declarations may map the same completed resident stream or material
//! texture into both frame namespaces.  A Graph slot then has one binding, but
//! every owner presented for that slot must still prove the same saved facts.

use super::*;
use crate::graph::ResourceInstanceId;

type StreamOwner = (ResourceInstanceId, DeviceId, BufferDesc, BufferUses);
type TextureOwner = (
    ResourceInstanceId,
    DeviceId,
    DeviceId,
    TextureDesc,
    TextureUses,
    ImageLayout,
);

pub(super) fn collect<'a, B: ResourceBackend>(
    execution: &CompiledExecution,
    device: DeviceId,
    capture_slots: &'a CaptureFrameSlots,
    capture: &'a ForwardFrame<B>,
    receiver_slots: &'a CaptureFrameSlots,
    receiver: &'a ForwardFrame<B>,
) -> Result<(Vec<BoundBuffer<'a, B>>, Vec<BoundTexture<'a, B>>)> {
    let mut buffers = Vec::new();
    let mut textures = Vec::new();
    let mut resident_stream_slots = std::collections::HashMap::<BufferSlot, StreamOwner>::new();
    let mut resident_texture_slots = std::collections::HashMap::<TextureSlot, TextureOwner>::new();
    for (slots, frame, capture_side) in [
        (capture_slots, capture, true),
        (receiver_slots, receiver, false),
    ] {
        super::capture_specular_bindings::append(
            slots,
            frame.specular.as_ref(),
            device,
            &mut buffers,
            &mut textures,
        )?;
        let uses_buffer = |slot| {
            execution
                .logical()
                .buffer_bindings()
                .requirements()
                .iter()
                .any(|requirement| requirement.slot == slot)
        };
        let mut push_buffer = |slot, instance, buffer, desc, incoming| {
            if uses_buffer(slot) {
                buffers.push(BoundBuffer::new(slot, instance, buffer, desc, incoming));
            }
        };
        push_buffer(
            slots.resident_upload,
            frame.resident.upload_instance,
            &frame.resident.upload,
            &frame.resident.upload_desc,
            frame.resident.upload_incoming,
        );
        push_buffer(
            slots.frame_upload,
            frame.frame.upload_instance,
            &frame.frame.upload,
            &frame.frame.upload_desc,
            frame.frame.upload_incoming,
        );
        push_buffer(
            slots.uniform,
            frame.frame.uniform_instance,
            &frame.frame.uniform,
            &frame.frame.uniform_desc,
            frame.frame.uniform_incoming,
        );
        push_buffer(
            slots.readback,
            frame.frame.readback_instance,
            &frame.frame.readback,
            &frame.frame.readback_desc,
            frame.frame.readback_incoming,
        );
        for (slot, stream) in slots.streams.iter().copied().zip(&frame.resident.streams) {
            let owner = (
                stream.instance,
                stream.buffer.device(),
                stream.desc.clone(),
                stream.incoming,
            );
            match resident_stream_slots.get(&slot) {
                Some(previous) if !same_stream_owner(previous, &owner) => {
                    return Err(Error::InvalidState);
                }
                Some(_) => {}
                None => {
                    resident_stream_slots.insert(slot, owner);
                    push_buffer(
                        slot,
                        stream.instance,
                        &stream.buffer,
                        &stream.desc,
                        stream.incoming,
                    );
                }
            }
        }
        if let (Some(slot), Some(owner)) = (slots.environment_upload, frame.environment.as_ref()) {
            push_buffer(
                slot,
                owner.upload_instance,
                &owner.upload,
                &owner.upload_desc,
                owner.upload_incoming,
            );
        }
        let capture_slots = if capture_side {
            vec![false; frame.resident.textures.len()]
        } else {
            capture_declaration::capture_texture_slots(&ForwardGraphSpec::from_frame(frame)?)?
        };
        for ((slot, texture), capture_texture) in slots
            .textures
            .iter()
            .copied()
            .zip(&frame.resident.textures)
            .zip(capture_slots)
        {
            if capture_texture {
                continue;
            }
            let owner = (
                texture.instance,
                texture.texture.device(),
                texture.view.device(),
                texture.desc.clone(),
                texture.incoming,
                texture.layout,
            );
            match resident_texture_slots.get(&slot) {
                Some(previous) if !same_texture_owner(previous, &owner) => {
                    return Err(Error::InvalidState);
                }
                Some(_) => {}
                None => {
                    resident_texture_slots.insert(slot, owner);
                    textures.push(BoundTexture::new(
                        slot,
                        texture.instance,
                        &texture.texture,
                        &texture.desc,
                        texture.incoming,
                        texture.layout,
                    ));
                }
            }
        }
        textures.push(BoundTexture::new(
            slots.target,
            frame.frame.target_instance,
            &frame.frame.target,
            &frame.frame.target_desc,
            frame.frame.target_incoming,
            frame.frame.target_layout,
        ));
        if let (Some(slot), Some(owner)) = (slots.environment, frame.environment.as_ref()) {
            textures.push(BoundTexture::new(
                slot,
                owner.texture_instance,
                &owner.texture,
                &owner.desc,
                owner.incoming,
                owner.layout,
            ));
        }
        if let (Some(slot), Some(owner)) = (slots.depth, frame.frame.depth.as_ref()) {
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
    Ok((buffers, textures))
}

fn same_stream_owner(left: &StreamOwner, right: &StreamOwner) -> bool {
    left.0 == right.0
        && left.1 == right.1
        && capture_declaration::same_buffer_owner_desc(&left.2, &right.2)
        && left.3 == right.3
}

fn same_texture_owner(left: &TextureOwner, right: &TextureOwner) -> bool {
    left.0 == right.0
        && left.1 == right.1
        && left.2 == right.2
        && capture_declaration::same_texture_owner_desc(&left.3, &right.3)
        && left.4 == right.4
        && left.5 == right.5
}
