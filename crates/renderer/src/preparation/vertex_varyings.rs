//! Renderer specialization for Material UV sets and typed vertex varyings.

use fluxel_material::MaterialType;
use fluxel_shader::{ScalarType, ShaderCompositionRequest, VertexInputDeclaration, VertexType};

use super::MaterialPreparationError;

const FIELD_MARKERS: [(&str, u32); 2] = [
    ("/* fluxel:vertex-varying-fields-5 */", 5),
    ("/* fluxel:vertex-varying-fields-8 */", 8),
];
const ASSIGNMENT_TARGETS: [&str; 2] = ["output", "o"];
const ARGUMENT_MARKER: &str = "/* fluxel:vertex-varying-arguments */";
const UV1_INPUT_MARKER: &str = "/* fluxel:uv1-input */";
// Built-in mesh semantics occupy locations 0..4; instance columns use 4..8.
const UV1_INPUT_LOCATION: u32 = 8;

pub(super) fn specialize(
    request: &mut ShaderCompositionRequest,
    varyings: &[(u32, MaterialType)],
    uv1_argument: bool,
    uv1_input: bool,
) -> Result<(), MaterialPreparationError> {
    if varyings.is_empty() && !uv1_argument {
        return Ok(());
    }
    let context = request.static_options.get("renderer.material_context");
    if !matches!(context.map(String::as_str), Some("uv0" | "uv0-normal")) {
        return Err(MaterialPreparationError::InvalidVertexVaryingRoute {
            context: context.cloned(),
        });
    }

    let mut fields_replaced = false;
    let mut assignments_replaced = false;
    let mut arguments_replaced = false;
    let mut uv1_input_replaced = false;
    let mut uv1_argument_replaced = false;
    for unit in &mut request.source_units {
        let source = &mut unit.source;
        if uv1_argument {
            if uv1_input && source.contains(UV1_INPUT_MARKER) {
                replace_marker(
                    source,
                    UV1_INPUT_MARKER,
                    &format!(", @location({UV1_INPUT_LOCATION}) fluxel_uv1: vec2<f32>"),
                    "uv1 input",
                    &mut uv1_input_replaced,
                )?;
            }
            *source = source
                .replace(
                    "/* fluxel:uv1-vertex-param */",
                    if uv1_input {
                        ", fluxel_uv1: vec2<f32>"
                    } else {
                        ""
                    },
                )
                .replace(
                    "/* fluxel:uv1-vertex-argument */",
                    if uv1_input { ", fluxel_uv1" } else { "" },
                );
            const UV1_ARG: &str = "/* fluxel:uv1-argument */";
            uv1_argument_replaced |= source.contains(UV1_ARG);
            *source = source
                .replace(
                    UV1_ARG,
                    if uv1_input {
                        "input.fluxel_uv1, "
                    } else {
                        "vec2<f32>(0.0), "
                    },
                )
                .replace("/* fluxel:uv1-local-param */", "fluxel_uv1: vec2<f32>, ")
                .replace("/* fluxel:uv1-local-argument */", "fluxel_uv1, ");
        }
        for (marker, location) in FIELD_MARKERS {
            if source.contains(marker) {
                replace_marker(
                    source,
                    marker,
                    &fields(varyings, location, uv1_input)?,
                    "fields",
                    &mut fields_replaced,
                )?;
            }
        }
        for target in ASSIGNMENT_TARGETS {
            let marker = format!(
                "/* fluxel:vertex-varying-write-{target} */{target}.vertex_uv0 = evaluate_pbr_vertex_uv0(uv0);"
            );
            if source.contains(&marker) {
                replace_marker(
                    source,
                    &marker,
                    &assignments(varyings, target, uv1_input),
                    "assignments",
                    &mut assignments_replaced,
                )?;
            }
        }
        if source.contains(ARGUMENT_MARKER) {
            replace_marker(
                source,
                ARGUMENT_MARKER,
                &arguments(varyings),
                "arguments",
                &mut arguments_replaced,
            )?;
        }
    }
    for (replaced, marker) in [
        (fields_replaced, "fields"),
        (assignments_replaced, "assignments"),
        (arguments_replaced, "arguments"),
        (!uv1_input || uv1_input_replaced, "uv1 input"),
        (!uv1_argument || uv1_argument_replaced, "uv1 argument"),
    ] {
        if !replaced {
            return Err(MaterialPreparationError::MissingVertexVaryingMarker { marker });
        }
    }
    if uv1_input {
        request.vertex_inputs.push(VertexInputDeclaration::new(
            "UV1",
            UV1_INPUT_LOCATION,
            VertexType::Vec2(ScalarType::F32),
        ));
    }
    Ok(())
}

fn replace_marker(
    source: &mut String,
    marker: &str,
    replacement: &str,
    role: &'static str,
    replaced: &mut bool,
) -> Result<(), MaterialPreparationError> {
    if *replaced || source.matches(marker).count() != 1 {
        return Err(MaterialPreparationError::MissingVertexVaryingMarker { marker: role });
    }
    *replaced = true;
    *source = source.replacen(marker, replacement, 1);
    Ok(())
}

fn fields(
    varyings: &[(u32, MaterialType)],
    mut location: u32,
    uv1_input: bool,
) -> Result<String, MaterialPreparationError> {
    let mut output = String::new();
    if uv1_input {
        output.push_str(&format!("@location({location}) fluxel_uv1: vec2<f32>,\n"));
        location += 1;
    }
    for (id, ty) in varyings {
        let count = column_count(*ty);
        for column in 0..count {
            let name = field_name(*id, *ty, column);
            let interpolation = match ty {
                MaterialType::I32 | MaterialType::U32 => " @interpolate(flat)",
                _ => "",
            };
            let field_type = if *ty == MaterialType::Mat4x4 {
                "vec4<f32>"
            } else {
                ty.wgsl()
            };
            output.push_str(&format!(
                "@location({location}){interpolation} {name}: {field_type},\n"
            ));
            location = location
                .checked_add(1)
                .ok_or(MaterialPreparationError::VertexVaryingLocationOverflow)?;
        }
    }
    Ok(output)
}

fn assignments(varyings: &[(u32, MaterialType)], target: &str, uv1_input: bool) -> String {
    let mut output = String::new();
    if uv1_input {
        output.push_str(&format!("{target}.fluxel_uv1 = fluxel_uv1;\n"));
    }
    if varyings.is_empty() {
        output.push_str(&format!(
            "{target}.vertex_uv0 = evaluate_pbr_vertex_uv0(uv0);\n"
        ));
        return output;
    }
    output.push_str(&format!(
        "let fluxel_vertex_varyings = evaluate_pbr_vertex_varyings(uv0);\n{target}.vertex_uv0 = fluxel_vertex_varyings.uv0;\n"
    ));
    for (id, ty) in varyings {
        for column in 0..column_count(*ty) {
            let field = field_name(*id, *ty, column);
            let source = if *ty == MaterialType::Mat4x4 {
                format!("fluxel_vertex_varyings.varying_{id}[{column}]")
            } else {
                format!("fluxel_vertex_varyings.varying_{id}")
            };
            output.push_str(&format!("{target}.{field} = {source};\n"));
        }
    }
    output
}

fn arguments(varyings: &[(u32, MaterialType)]) -> String {
    let mut output = String::new();
    for (id, ty) in varyings {
        if *ty == MaterialType::Mat4x4 {
            let columns = (0..4)
                .map(|column| format!("input.{}", field_name(*id, *ty, column)))
                .collect::<Vec<_>>()
                .join(", ");
            output.push_str(&format!(", mat4x4<f32>({columns})"));
        } else {
            output.push_str(&format!(", input.{}", field_name(*id, *ty, 0)));
        }
    }
    output
}

const fn column_count(ty: MaterialType) -> u32 {
    if matches!(ty, MaterialType::Mat4x4) {
        4
    } else {
        1
    }
}

fn field_name(id: u32, ty: MaterialType, column: u32) -> String {
    if ty == MaterialType::Mat4x4 {
        format!("varying_{id}_column_{column}")
    } else {
        format!("varying_{id}")
    }
}
