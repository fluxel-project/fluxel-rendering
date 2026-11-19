//! The completed Deferred body before its output is selected.

use super::*;
use crate::graph::definition::{BufferHandle, TextureHandle};

/// One real Deferred Graph up through lighting.
///
/// The handles are versions produced by the same Graph transaction.  A later
/// in-module contribution can continue from `scene_color` and `gbuffer[2]`
/// without importing the concrete owners again or replaying the scheduler.
#[derive(Clone)]
pub(super) struct DeferredGraphPrefix {
    pub(super) graph: GraphBuilder,
    pub(super) lane: LaneId,
    pub(super) upload_pass: GraphPass,
    pub(super) mip_generation: Vec<super::super::mip_generation::MipGenerationPass>,
    pub(super) indirect: Vec<(usize, super::super::indirect::IndirectGraph)>,
    pub(super) deformation_pass: Option<GraphPass>,
    pub(super) geometry_pass: GraphPass,
    pub(super) resolve_pass: Option<GraphPass>,
    pub(super) lighting_pass: GraphPass,
    pub(super) scene_color: TextureHandle,
    pub(super) resident_upload: BufferHandle,
    pub(super) frame_upload: BufferHandle,
    pub(super) uniform: BufferHandle,
    pub(super) streams: Vec<BufferHandle>,
    pub(super) materials: Vec<TextureHandle>,
    pub(super) environment_upload: Option<BufferHandle>,
    pub(super) environment: Option<TextureHandle>,
    pub(super) specular_upload: Option<BufferHandle>,
    pub(super) specular_view_uniform: Option<BufferHandle>,
    pub(super) specular_radiance: Option<TextureHandle>,
    pub(super) specular_brdf_lut: Option<TextureHandle>,
    pub(super) gbuffer: [TextureHandle; 3],
    pub(super) effects: Option<TextureHandle>,
    pub(super) diffuse_roughness: Option<TextureHandle>,
    pub(super) ambient_occlusion: Option<TextureHandle>,
    pub(super) msaa_gbuffer: Option<[TextureHandle; 3]>,
    pub(super) msaa_effects: Option<TextureHandle>,
    pub(super) msaa_diffuse_roughness: Option<TextureHandle>,
    pub(super) msaa_ambient_occlusion: Option<TextureHandle>,
    pub(super) surface_target: Option<TextureHandle>,
    pub(super) target: Option<TextureHandle>,
    pub(super) readback: Option<BufferHandle>,
    pub(super) row_pitch: Option<u32>,
    pub(super) deformation_handles:
        Option<(BufferHandle, BufferHandle, BufferHandle, BufferHandle)>,
    pub(super) extent: [u32; 2],
    pub(super) uniform_alignment: u64,
    pub(super) transform_bytes: Vec<u64>,
}

impl DeferredGraphPrefix {
    /// Projects prefix metadata for a combined owner without authoring a terminal.
    pub(super) fn capture_namespace_declaration(self) -> super::DeferredDeclaration {
        super::DeferredDeclaration {
            graph: self.graph,
            lane: self.lane,
            upload_pass: self.upload_pass,
            mip_generation: self.mip_generation,
            indirect: self.indirect,
            deformation_pass: self.deformation_pass,
            geometry_pass: self.geometry_pass,
            resolve_pass: self.resolve_pass,
            lighting_pass: self.lighting_pass,
            composite_pass: None,
            output: super::DeferredOutputDeclaration::Capture,
            resident_upload_slot: self.resident_upload.into(),
            frame_upload_slot: self.frame_upload.into(),
            uniform_slot: self.uniform.into(),
            stream_slots: self.streams.into_iter().map(Into::into).collect(),
            texture_slots: self.materials.into_iter().map(Into::into).collect(),
            environment_upload_slot: self.environment_upload.map(Into::into),
            environment_slot: self.environment.map(Into::into),
            specular_upload_slot: self.specular_upload.map(Into::into),
            specular_view_uniform_slot: self.specular_view_uniform.map(Into::into),
            specular_radiance_slot: self.specular_radiance.map(Into::into),
            specular_brdf_lut_slot: self.specular_brdf_lut.map(Into::into),
            gbuffer_slots: self.gbuffer.map(Into::into),
            effects_slot: self.effects.map(Into::into),
            diffuse_roughness_slot: self.diffuse_roughness.map(Into::into),
            ambient_occlusion_slot: self.ambient_occlusion.map(Into::into),
            msaa_gbuffer_slots: self.msaa_gbuffer.map(|items| items.map(Into::into)),
            msaa_effects_slot: self.msaa_effects.map(Into::into),
            msaa_diffuse_roughness_slot: self.msaa_diffuse_roughness.map(Into::into),
            msaa_ambient_occlusion_slot: self.msaa_ambient_occlusion.map(Into::into),
            scene_color_slot: Some(self.scene_color.into()),
            surface_target_slot: self.surface_target.map(Into::into),
            target_slot: self.target.map(Into::into),
            readback_slot: self.readback.map(Into::into),
            deformation_slots: self
                .deformation_handles
                .map(|(a, b, c, d)| [a.into(), b.into(), c.into(), d.into()]),
            extent: self.extent,
            uniform_alignment: self.uniform_alignment,
            transform_bytes: self.transform_bytes,
        }
    }
}
