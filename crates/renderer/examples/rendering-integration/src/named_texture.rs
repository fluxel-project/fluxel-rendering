use std::error::Error;

use fluxel_renderer::{ColorTextureAsset, ForwardResources, ForwardTextureSource, SceneAssetRef};

pub(crate) fn material_texture_reference(
    resources: &ForwardResources,
) -> Result<SceneAssetRef<ColorTextureAsset>, Box<dyn Error>> {
    let mut matches = resources
        .textures
        .iter()
        .filter(|binding| binding.resource == "material_texture");
    let binding = matches.next().ok_or("missing material_texture binding")?;
    if matches.next().is_some() {
        return Err("duplicate material_texture binding".into());
    }
    match &binding.source {
        ForwardTextureSource::Asset(asset) => Ok(*asset),
        _ => Err("material_texture must be an RGBA8 color asset binding".into()),
    }
}
