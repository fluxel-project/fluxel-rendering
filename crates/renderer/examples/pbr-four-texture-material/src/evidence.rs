//! Retain exact sources, source mapping and independent predictions before use.
use renderer_pbr_four_texture_material::{oracle, scene};
use std::{error::Error, fs, path::Path};
pub(crate) fn write_input(
    output: &Path,
    input: &scene::Input,
    probes: &[oracle::Probe],
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(output)?;
    fs::write(output.join("geometry.bin"), &input.geometry_bytes)?;
    fs::write(
        output.join("material.txt"),
        format!("{:#?}\n", input.material),
    )?;
    fs::write(output.join("Cargo.lock"), include_bytes!("../Cargo.lock"))?;
    let geometry = &input.assets.geometry[0];
    let material = &input.assets.materials[0];
    fs::write(
        output.join("owner-identities.json"),
        format!(
            "{{\"geometry\":{{\"id\":{:?},\"generation\":{}}},\"material\":{{\"id\":{:?},\"generation\":{}}}}}\n",
            format!("{:?}", geometry.id()),
            geometry.generation().get(),
            format!("{:?}", material.id()),
            material.generation().get(),
        ),
    )?;
    fs::write(
        output.join("diffuse-irradiance.rgba"),
        &input.diffuse.value().rgba8,
    )?;
    let mut assets = Vec::new();
    for (index, asset) in input.assets.textures.iter().enumerate() {
        let data = asset.value();
        let name = format!("source-{index}.rgba");
        fs::write(output.join(&name), &data.rgba8)?;
        assets.push(format!("{{\"index\":{index},\"id\":{:?},\"generation\":{},\"extent\":{:?},\"encoding\":{:?},\"file\":{:?}}}",format!("{:?}",asset.id()),asset.generation().get(),data.extent,format!("{:?}",data.encoding),name));
    }
    let predictions=probes.iter().map(|p|format!("{{\"x\":{},\"y\":{},\"object\":{},\"uv\":{:?},\"forward\":{:?},\"deferred\":{:?}}}",p.x,p.y,p.object,p.uv,p.forward,p.deferred)).collect::<Vec<_>>().join(",");
    fs::write(
        output.join("input.json"),
        format!(
            "{{\"case\":{:?},\"extent\":{:?},\"roleNames\":{:?},\"objectRoleSources\":{:?},\"worldFromLocal\":{:?},\"tints\":{:?},\"vertexStride\":48,\"assets\":[{}],\"sampling\":\"linear clamp LOD0; source-format RGB decode before filter; alpha linear\",\"normal\":\"authored RGB*2-1 with actual plane TBN\",\"orm\":\"R=AO,G=perceptual roughness,B=metallic\"}}\n",
            input.case.name(),
            scene::EXTENT,
            scene::ROLE_NAMES,
            input.sources,
            input.models,
            scene::TINTS,
            assets.join(",")
        ),
    )?;
    fs::write(
        output.join("oracle.json"),
        format!(
            "{{\"schemaVersion\":1,\"rgbTolerance\":{},\"independent\":\"source bytes, projected plane UV, exact sRGB decode, bilinear, GGX+diffuse irradiance AO+emissive; separate Deferred G-buffer quantization\",\"probeSelection\":\"CPU-only stable across all controls and floor/ceil eight-fractional-bit filter coordinates plus neighboring UNORM8 normal/ORM samples; precision variation <=2 RGB codes; same coordinates for every case; separation >=16 pixels\",\"probes\":[{predictions}]}}\n",
            oracle::TOLERANCE
        ),
    )?;
    Ok(())
}
