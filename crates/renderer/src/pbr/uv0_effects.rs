//! Effect-enabled PBR requests for UV0 surfaces without tangent-space normals.

use fluxel_shader::{
    BindingDeclaration, BindingKind, ShaderCompositionRequest, ShaderStage, SourceUnit,
};

use crate::GeometryLayout;

pub(crate) fn forward_request(layout: &GeometryLayout) -> Option<ShaderCompositionRequest> {
    let mut request = super::geometry_uv0_request(layout, false)?;
    request
        .source_units
        .retain(|unit| unit.id.0 != "pbr-forward-uv0");
    request.source_units.push(SourceUnit::new(
        "pbr-forward-uv0-effect",
        super::templates::FORWARD_UV0_EFFECT,
    ));
    request.static_options.insert(
        "renderer.surface_effects".into(),
        "emissive-ao-rgba16float-v1".into(),
    );
    super::install_emissive_evaluator(&mut request);
    Some(request)
}

pub(crate) fn forward_irradiance_request(
    layout: &GeometryLayout,
) -> Option<ShaderCompositionRequest> {
    let mut request = forward_request(layout)?;
    request
        .source_units
        .retain(|unit| unit.id.0 != "pbr-forward-uv0-effect");
    request.source_units.push(SourceUnit::new(
        "pbr-diffuse-irradiance",
        super::templates::DIFFUSE_IRRADIANCE,
    ));
    request.source_units.push(SourceUnit::new(
        "pbr-forward-uv0-effect-irradiance",
        super::templates::FORWARD_UV0_EFFECT_IRRADIANCE,
    ));
    Some(
        request
            .with_binding(
                BindingDeclaration::new(2, 0, BindingKind::Texture)
                    .with_visibility([ShaderStage::Fragment]),
            )
            .with_binding(
                BindingDeclaration::new(2, 1, BindingKind::Sampler)
                    .with_visibility([ShaderStage::Fragment]),
            )
            .with_static_option("renderer.environment", "latlong-diffuse-irradiance-v1"),
    )
}

pub(crate) fn gbuffer_request(layout: &GeometryLayout) -> Option<ShaderCompositionRequest> {
    let mut request = super::geometry_uv0_request(layout, true)?;
    request
        .source_units
        .retain(|unit| unit.id.0 != "pbr-gbuffer-uv0");
    request.source_units.push(SourceUnit::new(
        "pbr-gbuffer-uv0-effect",
        super::templates::GBUFFER_UV0_EFFECT,
    ));
    request.static_options.insert(
        "renderer.surface_effects".into(),
        "emissive-ao-rgba16float-v1".into(),
    );
    super::install_emissive_evaluator(&mut request);
    Some(request)
}
