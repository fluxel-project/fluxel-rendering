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

/// Asset references for a receiver which samples an explicitly named capture
/// product rather than a CPU texture asset.
#[derive(Clone, Debug)]
pub struct ForwardCaptureResources {
    /// Exact immutable geometry generation.
    pub geometry: SceneAssetRef<GeometryAsset>,
    /// Exact immutable Material generation.
    pub material: SceneAssetRef<MaterialAsset>,
    /// Renderer-owned capture product slot consumed by this material.
    pub texture: ForwardTextureSource,
    /// Per-record material tint uploaded through group 0 binding 0 later.
    pub tint: [f32; 4],
}

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
    pub(crate) capture_slot: String,
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
            texture: ResolvedForwardTexture::CaptureSlot(self.capture_slot),
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
    let ForwardTextureSource::CaptureSlot(capture_slot) = &resources.texture else {
        return Err(ForwardPreparationError::CaptureReceiverRequiresSlot { source });
    };
    if capture_slot.is_empty() {
        return Err(ForwardPreparationError::MissingCaptureSlot {
            source,
            slot: capture_slot.clone(),
        });
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
    validate_material_schema(&prepared, source)?;
    let valid_slot = prepared
        .material()
        .contribution()
        .bindings
        .iter()
        .any(|binding| {
            binding.name == *capture_slot
                && binding.kind == fluxel_material::MaterialResourceKind::Texture2dF32
                && binding.location.group == 0
                && binding.location.binding == 1
        });
    if !valid_slot {
        return Err(ForwardPreparationError::InvalidCaptureSlot {
            source,
            slot: capture_slot.clone(),
        });
    }
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
        capture_slot: capture_slot.clone(),
        tint: resources.tint,
        transform,
        view: view.clone(),
        clip_from_local,
    })
}
