//! Resolution and validation of one ordinary Forward draw.

use fluxel_material::{MaterialInstance, SurfaceAlphaMode, SurfaceSideOrientation};
use fluxel_shader::ShaderComposer;

use crate::{MeshVisibility, RenderTransform, RenderView};

use super::{
    ForwardAssets, ForwardDrawInput, ForwardDrawSource, ForwardPreparationError, ForwardResources,
    effective_alpha_mode, finite_transform, material_request, multiply_columns, normal_from_local,
    resolve_asset, resolve_asset_textures, resolve_resource_aliases, resolve_uniform_values,
    validate_cpu_geometry, validate_material_schema,
};

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_draw(
    view: &RenderView,
    source: ForwardDrawSource,
    order: i64,
    resources: &ForwardResources,
    instance: Option<&MaterialInstance>,
    mesh_side_orientation: SurfaceSideOrientation,
    mesh_visibility: MeshVisibility,
    transform: RenderTransform,
    assets: &ForwardAssets,
    composer: &mut ShaderComposer,
    environment: bool,
    specular: bool,
) -> Result<ForwardDrawInput, ForwardPreparationError> {
    if !finite_transform(transform) {
        return Err(ForwardPreparationError::InvalidTransform { source });
    }
    if view.emissive_enabled
        && instance.is_none()
        && !resources.tint.iter().all(|value| value.is_finite())
    {
        return Err(ForwardPreparationError::InvalidTint { source });
    }
    let geometry = resolve_asset(&assets.geometry, resources.geometry, "geometry", source)?;
    let material = resolve_asset(&assets.materials, resources.material, "material", source)?;
    let automatic_alpha_mode = super::alpha::with_visibility(
        super::alpha::resolve(material.value(), instance, source)?,
        mesh_visibility,
    );
    let instance_transforms =
        super::super::forward_instances::instance_transforms(geometry.value(), source)?;
    let mesh_layout = super::super::mesh_vertex_layout(&geometry.value().layout);
    let mut request = material_request(
        material.value(),
        &mesh_layout,
        view,
        source,
        environment,
        specular,
        None,
    )?;
    if instance_transforms.is_some() {
        if matches!(
            automatic_alpha_mode.unwrap_or_else(|| effective_alpha_mode(
                material.value().surface_policy().alpha_mode,
                mesh_visibility,
            )),
            SurfaceAlphaMode::Blend | SurfaceAlphaMode::MaskBlend,
        ) || material.value().transmission_extensions().is_some()
        {
            return Err(ForwardPreparationError::InstanceMaterialUnsupported { source });
        }
        request = super::super::with_instance_transform(request, source)?;
    }
    let prepared = crate::preparation::prepare_material_for_alpha(
        &material,
        &geometry.value().layout,
        composer,
        request,
        automatic_alpha_mode,
    )
    .map_err(ForwardPreparationError::Material)?;
    super::forward_transmission::validate_transform(material.value(), transform, source)?;
    let schema = validate_material_schema(&prepared, source)?;
    let resources =
        resolve_resource_aliases(resources, instance, prepared.material(), &schema, source)?;
    let uniform_values = resolve_uniform_values(instance, &schema, resources.tint, source)?;
    let textures = resolve_asset_textures(&resources, &schema, assets, source)?;
    let samplers = super::super::forward_sampler::resolve_sampler_bindings(
        &resources.samplers,
        &schema,
        source,
    )?;
    let prepared = prepared.with_binding_schema(schema);
    validate_cpu_geometry(
        geometry.value(),
        prepared.geometry(),
        instance_transforms.map(|instances| instances.count),
        source,
    )?;
    let normal_from_local = normal_from_local(geometry.value(), &prepared, transform, source)?;
    let clip_from_local = multiply_columns(view.transform, transform);
    if !clip_from_local
        .iter()
        .flatten()
        .all(|value| value.is_finite())
    {
        return Err(ForwardPreparationError::InvalidTransform { source });
    }
    Ok(ForwardDrawInput {
        source,
        order,
        geometry,
        material,
        textures,
        prepared,
        samplers,
        tint: resources.tint,
        uniform_values,
        mesh_side_orientation,
        mesh_visibility,
        transform,
        view: view.clone(),
        clip_from_local,
        normal_from_local,
        instance_transforms,
        particle: None,
        scene_refraction: None,
    })
}
