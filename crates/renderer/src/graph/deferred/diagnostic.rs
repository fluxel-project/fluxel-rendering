//! Read-only Deferred submission diagnostics retained with pending ownership.

use super::*;
use std::fmt::Write;

impl<B: ResourceBackend> DeferredPlan<B> {
    /// Projects the compiled contracts and concrete frame-owner facts to text.
    ///
    /// The text deliberately exposes no graph handles or recording authority.
    pub(crate) fn diagnostic(&self, frame: &DeferredFrame<B>) -> String {
        self.diagnostic_body(super::record_view::DeferredRecordView::offscreen(frame))
    }

    pub(crate) fn surface_diagnostic(&self, frame: &DeferredSurfaceFrame<B>) -> String {
        self.diagnostic_body(super::record_view::DeferredRecordView::surface(frame))
    }

    fn diagnostic_body(&self, frame: super::record_view::DeferredRecordView<'_, B>) -> String {
        let mut output = format!("compiled_execution={:#?}\n", self.execution);
        let _ = writeln!(output, "plan={:?} lane={:?}", self.id, self.lane);
        buffer(
            &mut output,
            "resident_upload",
            frame.resident.upload_instance,
            frame.resident.upload.device(),
            &frame.resident.upload_desc,
            frame.resident.upload_incoming,
        );
        let (
            upload,
            upload_instance,
            upload_desc,
            upload_incoming,
            uniform,
            uniform_instance,
            uniform_desc,
            uniform_incoming,
        ) = match frame.source {
            super::record_view::RecordSource::Offscreen(owner) => (
                &owner.frame.upload,
                owner.frame.upload_instance,
                &owner.frame.upload_desc,
                owner.frame.upload_incoming,
                &owner.frame.uniform,
                owner.frame.uniform_instance,
                &owner.frame.uniform_desc,
                owner.frame.uniform_incoming,
            ),
            super::record_view::RecordSource::Surface(owner) => (
                &owner.frame.upload,
                owner.frame.upload_instance,
                &owner.frame.upload_desc,
                owner.frame.upload_incoming,
                &owner.frame.uniform,
                owner.frame.uniform_instance,
                &owner.frame.uniform_desc,
                owner.frame.uniform_incoming,
            ),
        };
        buffer(
            &mut output,
            "frame_upload",
            upload_instance,
            upload.device(),
            upload_desc,
            upload_incoming,
        );
        buffer(
            &mut output,
            "uniform",
            uniform_instance,
            uniform.device(),
            uniform_desc,
            uniform_incoming,
        );
        if let super::record_view::RecordSource::Offscreen(owner) = frame.source {
            buffer(
                &mut output,
                "readback",
                owner.frame.readback_instance,
                owner.frame.readback.device(),
                &owner.frame.readback_desc,
                owner.frame.readback_incoming,
            );
        }
        for (index, stream) in frame.resident.streams.iter().enumerate() {
            buffer(
                &mut output,
                &format!("stream[{index}]"),
                stream.instance,
                stream.buffer.device(),
                &stream.desc,
                stream.incoming,
            );
        }
        if let Some(deformation) = &frame.deformation {
            for (label, owner) in [
                ("deformation_upload", &deformation.upload),
                ("deformation_source", &deformation.source),
                ("deformation_product", &deformation.product),
                ("deformation_parameters", &deformation.parameters),
            ] {
                buffer(
                    &mut output,
                    label,
                    owner.instance,
                    owner.buffer.device(),
                    &owner.desc,
                    owner.incoming,
                );
            }
            let _ = writeln!(
                output,
                "deformation={{source:{:?}, parameters:{:?}, groups:{:?}, vertices:{}}}",
                0..deformation.source_bytes,
                deformation.parameter_source,
                deformation.groups,
                deformation.vertex_count,
            );
        }
        texture(
            &mut output,
            "material[0]",
            frame.resident.textures[0].instance,
            frame.resident.textures[0].texture.device(),
            &frame.resident.textures[0].desc,
            frame.resident.textures[0].incoming,
            frame.resident.textures[0].layout,
        );
        for (index, texture_owner) in frame.gbuffer.iter().enumerate() {
            texture(
                &mut output,
                &format!("gbuffer[{index}]"),
                texture_owner.instance,
                texture_owner.texture.device(),
                &texture_owner.desc,
                texture_owner.incoming,
                texture_owner.layout,
            );
        }
        match frame.source {
            super::record_view::RecordSource::Offscreen(owner) => texture(
                &mut output,
                "final_target",
                owner.frame.target_instance,
                owner.frame.target.device(),
                &owner.frame.target_desc,
                owner.frame.target_incoming,
                owner.frame.target_layout,
            ),
            super::record_view::RecordSource::Surface(owner) => {
                let _ = writeln!(
                    output,
                    "surface={{format:{:?}, extent:{:?}, present_step:{:?}}}",
                    owner.color_format,
                    owner.extent,
                    self.present_step()
                );
            }
        }
        let _ = writeln!(
            output,
            "draw={{vertices:{:?}, vertex_ranges:{:?}, tint:{:?}, transform:{:?}}}",
            frame.draw.vertices,
            frame.draw.vertex_ranges,
            frame.draw.tint_range,
            frame.draw.transform_range,
        );
        output
    }
}

fn buffer(
    output: &mut String,
    label: &str,
    instance: super::super::ResourceInstanceId,
    device: fluxel_rhi::api::DeviceId,
    desc: &fluxel_rhi::api::BufferDesc,
    incoming: BufferUses,
) {
    let _ = writeln!(
        output,
        "buffer {label}: instance={instance:?} device={device:?} desc={desc:?} incoming={incoming:?}",
    );
}

fn texture(
    output: &mut String,
    label: &str,
    instance: super::super::ResourceInstanceId,
    device: fluxel_rhi::api::DeviceId,
    desc: &fluxel_rhi::api::TextureDesc,
    incoming: TextureUses,
    layout: ImageLayout,
) {
    let _ = writeln!(
        output,
        "texture {label}: instance={instance:?} device={device:?} desc={desc:?} incoming={incoming:?} layout={layout:?}",
    );
}
