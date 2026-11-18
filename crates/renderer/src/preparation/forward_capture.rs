//! Capture-product preparation for the bounded Forward receiver.

use fluxel_material::{MaterialCompiler, MaterialResourceKind};

use super::*;
use crate::preparation::material::prepare_compiled_material;

#[derive(Clone)]
struct SceneRefractionBinding {
    texture: String,
    sampler: String,
}

impl SceneRefractionBinding {
    fn names(&self) -> (&str, &str) {
        (&self.texture, &self.sampler)
    }
}

/// The declared source for a Forward material's sampled color input.
#[derive(Clone, Debug)]
pub enum ForwardTextureSource {
    /// One exact immutable CPU color-texture generation.
    Asset(SceneAssetRef<ColorTextureAsset>),
    /// A renderer-owned same-frame capture product logical resource name.
    CaptureSlot(String),
}

impl From<SceneAssetRef<ColorTextureAsset>> for ForwardTextureSource {
    fn from(value: SceneAssetRef<ColorTextureAsset>) -> Self {
        Self::Asset(value)
    }
}

/// Resources for a receiver which samples an explicitly named capture product.
pub type ForwardCaptureResources = ForwardResources;

/// The concrete source for a prepared Forward material texture binding.
#[derive(Clone)]
pub(crate) enum ResolvedForwardTexture {
    Asset(AssetSnapshot<ColorTextureAsset, ColorTextureData>),
    CaptureSlot(String),
}

/// CPU-only receiver input supplied by a same-frame Graph capture product.
#[derive(Clone)]
pub struct PreparedCaptureReceiver {
    pub(crate) source: ForwardDrawSource,
    pub(crate) order: i64,
    pub(crate) geometry: AssetSnapshot<GeometryAsset, GeometryData>,
    pub(crate) material: AssetSnapshot<MaterialAsset, MaterialDefinition>,
    pub(crate) prepared: PreparedMaterial,
    pub(crate) sampler: Option<fluxel_rhi::api::SamplerDesc>,
    pub(crate) textures: Vec<ResolvedForwardTexture>,
    pub(crate) tint: [f32; 4],
    pub(crate) transform: RenderTransform,
    pub(crate) view: RenderView,
    pub(crate) clip_from_local: [[f32; 4]; 4],
    pub(crate) normal_from_local: Option<[[f32; 4]; 4]>,
    scene_refraction: Option<SceneRefractionBinding>,
}

impl PreparedCaptureReceiver {
    /// Converts a validated receiver without manufacturing a CPU texture asset.
    pub(crate) fn into_draw_input(self) -> ForwardDrawInput {
        ForwardDrawInput {
            source: self.source,
            order: self.order,
            geometry: self.geometry,
            material: self.material,
            textures: self.textures,
            prepared: self.prepared,
            sampler: self.sampler,
            tint: self.tint,
            transform: self.transform,
            view: self.view,
            clip_from_local: self.clip_from_local,
            normal_from_local: self.normal_from_local,
        }
    }

    /// Re-prepares this retained receiver for the selected lighting route.
    pub(crate) fn with_lighting(
        self,
        composer: &mut ShaderComposer,
        environment: bool,
        specular: bool,
    ) -> Result<Self, ForwardPreparationError> {
        if self.scene_refraction.is_some() {
            return Err(
                ForwardPreparationError::UnsupportedSceneRefractionReceiver {
                    source: self.source,
                },
            );
        }
        self.with_capture_lighting(composer, environment, specular)
    }

    pub(crate) fn with_capture_lighting(
        mut self,
        composer: &mut ShaderComposer,
        environment: bool,
        specular: bool,
    ) -> Result<Self, ForwardPreparationError> {
        let request = material_request(
            self.material.value(),
            &self.geometry.value().layout,
            &self.view,
            self.source,
            environment,
            specular,
            self.scene_refraction
                .as_ref()
                .map(SceneRefractionBinding::names),
        )?;
        let prepared = prepare_material(
            &self.material,
            &self.geometry.value().layout,
            composer,
            request,
        )
        .map_err(ForwardPreparationError::Material)?;
        let schema = validate_material_schema(&prepared, self.source)?;
        if self.prepared.binding_schema() != Some(&schema) {
            return Err(ForwardPreparationError::MaterialBindingSchema {
                source: self.source,
            });
        }
        validate_cpu_geometry(self.geometry.value(), prepared.geometry(), self.source)?;
        super::forward_transmission::validate_transform(
            self.material.value(),
            self.transform,
            self.source,
        )?;
        self.normal_from_local = normal_from_local(
            self.geometry.value(),
            prepared.geometry(),
            self.transform,
            crate::pbr::is_pbr(self.material.value().output()),
            matches!(
                self.material.value().output(),
                fluxel_material::SurfaceOutput::MetallicRoughnessUv0Normal { .. }
            ),
            self.source,
        )?;
        self.prepared = prepared.with_binding_schema(schema);
        Ok(self)
    }
}

/// Prepares one receiver which samples an explicitly named capture product.
#[allow(clippy::too_many_arguments)]
pub fn prepare_forward_capture_receiver(
    view: &RenderView,
    source: ForwardDrawSource,
    order: i64,
    resources: &ForwardCaptureResources,
    transform: RenderTransform,
    assets: &ForwardAssets,
    composer: &mut ShaderComposer,
) -> Result<PreparedCaptureReceiver, ForwardPreparationError> {
    prepare_capture_receiver(
        view, source, order, resources, transform, assets, composer, None,
    )
}

/// Prepares one PBR receiver that refracts an explicitly named capture texture.
///
/// Select a texture resource bound to a capture slot in `resources`. The
/// ordinary Forward capture producer supplies that slot. Transmission and an
/// ordinary material sampler must be declared, and the submitted renderer
/// must provide the specular BRDF environment used for surface reflectance.
#[allow(clippy::too_many_arguments)]
pub fn prepare_forward_scene_refraction_receiver(
    view: &RenderView,
    source: ForwardDrawSource,
    order: i64,
    resources: &ForwardCaptureResources,
    transform: RenderTransform,
    assets: &ForwardAssets,
    composer: &mut ShaderComposer,
    refraction_texture: &str,
) -> Result<PreparedCaptureReceiver, ForwardPreparationError> {
    prepare_capture_receiver(
        view,
        source,
        order,
        resources,
        transform,
        assets,
        composer,
        Some(refraction_texture),
    )
}

#[allow(clippy::too_many_arguments)]
fn prepare_capture_receiver(
    view: &RenderView,
    source: ForwardDrawSource,
    order: i64,
    resources: &ForwardCaptureResources,
    transform: RenderTransform,
    assets: &ForwardAssets,
    composer: &mut ShaderComposer,
    refraction_texture: Option<&str>,
) -> Result<PreparedCaptureReceiver, ForwardPreparationError> {
    if view.extent.contains(&0) || !finite_transform(view.transform) {
        return Err(ForwardPreparationError::InvalidView);
    }
    if !finite_transform(transform) {
        return Err(ForwardPreparationError::InvalidTransform { source });
    }
    if !resources.tint.iter().all(|value| value.is_finite()) {
        return Err(ForwardPreparationError::InvalidTint { source });
    }
    let geometry = resolve_asset(&assets.geometry, resources.geometry, "geometry", source)?;
    let material = resolve_asset(&assets.materials, resources.material, "material", source)?;
    let compiled = MaterialCompiler::compile_for(&material)
        .map_err(MaterialPreparationError::Material)
        .map_err(ForwardPreparationError::Material)?;
    let scene_refraction = refraction_texture
        .map(|name| scene_refraction_binding(&compiled, resources, name, source))
        .transpose()?;
    let request = material_request(
        material.value(),
        &geometry.value().layout,
        view,
        source,
        false,
        // The actual environment is validated again at submission.
        material.value().transmission_extensions().is_some(),
        scene_refraction.as_ref().map(SceneRefractionBinding::names),
    )?;
    let prepared = prepare_compiled_material(compiled, &geometry.value().layout, composer, request)
        .map_err(ForwardPreparationError::Material)?;
    super::forward_transmission::validate_transform(material.value(), transform, source)?;
    let schema = validate_material_schema(&prepared, source)?;
    let textures = ordered_texture_sources(&resources.textures, &schema.textures, source)?;
    let textures = resolve_capture_textures(textures, &schema.textures, assets, source)?;
    let sampler = super::super::forward_sampler::resolve_sampler_binding(
        resources.sampler.as_ref(),
        &schema,
        source,
    )?;
    let prepared = prepared.with_binding_schema(schema);
    validate_cpu_geometry(geometry.value(), prepared.geometry(), source)?;
    let normal_from_local = normal_from_local(
        geometry.value(),
        prepared.geometry(),
        transform,
        crate::pbr::is_pbr(material.value().output()),
        matches!(
            material.value().output(),
            fluxel_material::SurfaceOutput::MetallicRoughnessUv0Normal { .. }
        ),
        source,
    )?;
    let clip_from_local = multiply_columns(view.transform, transform);
    if !clip_from_local
        .iter()
        .flatten()
        .all(|value| value.is_finite())
    {
        return Err(ForwardPreparationError::InvalidTransform { source });
    }
    Ok(PreparedCaptureReceiver {
        source,
        order,
        geometry,
        material,
        prepared,
        sampler,
        textures,
        tint: resources.tint,
        transform,
        view: view.clone(),
        clip_from_local,
        normal_from_local,
        scene_refraction,
    })
}

fn scene_refraction_binding(
    material: &fluxel_material::CompiledMaterialVariant,
    resources: &ForwardCaptureResources,
    texture: &str,
    source: ForwardDrawSource,
) -> Result<SceneRefractionBinding, ForwardPreparationError> {
    let invalid = ForwardPreparationError::InvalidSceneRefractionSource { source };
    let bindings = &material.contribution().bindings;
    if material.transmission_extensions().is_none()
        || !bindings.iter().any(|binding| {
            binding.name == texture && binding.kind == MaterialResourceKind::Texture2dF32
        })
        || !resources.textures.iter().any(|binding| {
            binding.resource == texture
                && matches!(binding.source, ForwardTextureSource::CaptureSlot(_))
        })
    {
        return Err(invalid);
    }
    let sampler = bindings
        .iter()
        .find(|binding| binding.kind == MaterialResourceKind::Sampler { comparison: false })
        .ok_or(invalid)?;
    Ok(SceneRefractionBinding {
        texture: texture.to_owned(),
        sampler: sampler.name.clone(),
    })
}

fn resolve_capture_textures(
    textures: Vec<&ForwardTextureBinding>,
    expected: &[super::super::MaterialTextureBinding],
    assets: &ForwardAssets,
    source: ForwardDrawSource,
) -> Result<Vec<ResolvedForwardTexture>, ForwardPreparationError> {
    let mut captures = 0;
    let resolved = textures
        .into_iter()
        .zip(expected)
        .map(|(texture, expected)| match &texture.source {
            ForwardTextureSource::Asset(reference) => {
                let asset = resolve_asset(&assets.textures, *reference, "texture", source)?;
                validate_texture(asset.value(), source)?;
                super::validate_texture_encoding(expected, asset.value().encoding, source)?;
                Ok(ResolvedForwardTexture::Asset(asset))
            }
            ForwardTextureSource::CaptureSlot(slot) => {
                if slot.is_empty() {
                    return Err(ForwardPreparationError::MissingCaptureSlot {
                        source,
                        slot: slot.clone(),
                    });
                }
                super::validate_texture_encoding(expected, TextureEncoding::Linear, source)?;
                captures += 1;
                Ok(ResolvedForwardTexture::CaptureSlot(slot.clone()))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    if captures == 0 {
        return Err(ForwardPreparationError::CaptureReceiverRequiresSlot { source });
    }
    Ok(resolved)
}
