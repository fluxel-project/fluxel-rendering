//! Raw authored inputs and actual opened-device facts retained before rendering.
use fluxel_rhi::{api::Device, backend::ResourceBackend};
use renderer_09_mipmap_tunnel_material::scene::Input;
use std::{error::Error, fs, path::Path};

pub(super) fn write_device<B: ResourceBackend>(
    output: &Path,
    device: &Device<B>,
) -> Result<(), Box<dyn Error>> {
    let limits = &device.capabilities().limits;
    fs::write(
        output.join("device-limits.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "device": format!("{:?}", device.id()),
            "maxBufferBytes": limits.max_buffer_bytes,
            "maxTextureDimension2d": limits.max_texture_dimension_2d,
            "maxTextureArrayLayers": limits.max_texture_array_layers,
            "maxColorAttachments": limits.max_color_attachments,
            "maxVertexBuffers": limits.max_vertex_buffers,
            "maxVertexAttributes": limits.max_vertex_attributes,
            "maxBindGroups": limits.max_bind_groups,
            "maxBindingsPerGroup": limits.max_bindings_per_group,
            "uniformOffsetAlignment": limits.uniform_offset_alignment,
        }))?,
    )?;
    Ok(())
}

pub(super) fn write_inputs(output: &Path, input: &Input) -> Result<(), Box<dyn Error>> {
    fs::write(output.join("input-model.gltf"), &input.model_bytes)?;
    fs::write(output.join("input-asset.ktx"), &input.asset_bytes)?;
    fs::write(
        output.join("tunnel-geometry.bin"),
        &input.geometry.streams[0],
    )?;
    fs::write(output.join("mip0-source.rgba"), &input.image)?;
    fs::write(
        output.join("material.txt"),
        format!("{:#?}\n", input.material),
    )?;
    fs::write(
        output.join("Cargo.lock"),
        include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.lock")),
    )?;
    let sampler = &input.sampler;
    let values = serde_json::json!({
        "modelFnv1a64": format!("{:016x}", fnv1a64(&input.model_bytes)),
        "assetFnv1a64": format!("{:016x}", fnv1a64(&input.asset_bytes)),
        "extent": input.extent,
        "mipLevels": input.assets.textures[0].value().mip_levels,
        "basePayloadBytes": input.image.len(),
        "geometryVertices": input.geometry.vertices.end,
        "geometryLayout": format!("{:?}", input.geometry.layout),
        "viewProjection": input.vp,
        "worldFromLocal": fluxel_renderer::RenderTransform::identity().columns(),
        "tint": [0.0, 1.0, 1.0, 1.0],
        "geometryIdentity": format!("{:?}", fluxel_renderer::SceneAssetRef::from_snapshot(&input.assets.geometry[0])),
        "materialIdentity": format!("{:?}", fluxel_renderer::SceneAssetRef::from_snapshot(&input.assets.materials[0])),
        "textureIdentity": format!("{:?}", fluxel_renderer::SceneAssetRef::from_snapshot(&input.assets.textures[0])),
        "requestedSampler": {
            "address": sampler.address.map(|value| format!("{value:?}")),
            "magFilter": format!("{:?}", sampler.mag_filter),
            "minFilter": format!("{:?}", sampler.min_filter),
            "mipFilter": format!("{:?}", sampler.mip_filter),
            "lod": [sampler.lod.start, sampler.lod.end],
            "anisotropy": sampler.anisotropy,
            "compare": format!("{:?}", sampler.compare),
            "border": format!("{:?}", sampler.border),
        },
        "sampling": "textureSampleBias, implicit derivative LOD, opaque sampled RGB",
        "forwardOutput": "direct authored color",
        "deferredOutput": "legacy colored-albedo diagnostic lighting; separately modeled packed normal and depth",
    });
    fs::write(
        output.join("input.json"),
        serde_json::to_vec_pretty(&values)?,
    )?;
    Ok(())
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}
