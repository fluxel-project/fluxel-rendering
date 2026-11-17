//! Capture-product preparation for the bounded Forward receiver.

use super::*;

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
    pub(crate) textures: Vec<ResolvedForwardTexture>,
    pub(crate) tint: [f32; 4],
    pub(crate) transform: RenderTransform,
    pub(crate) view: RenderView,
    pub(crate) clip_from_local: [[f32; 4]; 4],
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
            tint: self.tint,
            transform: self.transform,
            view: self.view,
            clip_from_local: self.clip_from_local,
            normal_from_local: None,
        }
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
    let request = match material.value().output() {
        fluxel_material::SurfaceOutput::Color => {
            forward_template(&geometry.value().layout, source)?.request()
        }
        fluxel_material::SurfaceOutput::ColorUv0 => {
            color_uv0_template(&geometry.value().layout, source)?.request()
        }
        _ => return Err(ForwardPreparationError::MaterialBindingSchema { source }),
    };
    let prepared = prepare_material(&material, &geometry.value().layout, composer, request)
        .map_err(ForwardPreparationError::Material)?;
    let schema = validate_material_schema(&prepared, source)?;
    let textures = ordered_texture_sources(&resources.textures, &schema.textures, source)?;
    if textures.len() != 1 {
        return Err(ForwardPreparationError::CaptureReceiverRequiresSlot { source });
    }
    let texture = textures[0];
    let ForwardTextureSource::CaptureSlot(capture_slot) = &texture.source else {
        return Err(ForwardPreparationError::CaptureReceiverRequiresSlot { source });
    };
    if capture_slot.is_empty() {
        return Err(ForwardPreparationError::MissingCaptureSlot {
            source,
            slot: capture_slot.clone(),
        });
    }
    // The current capture entry names one product and one receiver binding
    // with the same slot. General product-to-binding aliases belong to the
    // mixed-source capture consumer, rather than silently widening this path.
    if capture_slot != &texture.resource {
        return Err(ForwardPreparationError::InvalidCaptureSlot {
            source,
            slot: capture_slot.clone(),
        });
    }
    let prepared = prepared.with_binding_schema(schema);
    validate_cpu_geometry(geometry.value(), prepared.geometry(), source)?;
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
        textures: vec![ResolvedForwardTexture::CaptureSlot(capture_slot.clone())],
        tint: resources.tint,
        transform,
        view: view.clone(),
        clip_from_local,
    })
}
