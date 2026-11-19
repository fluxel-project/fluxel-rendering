//! Renderer specialization for typed Material vertex-to-fragment varyings.

use fluxel_material::MaterialType;
use fluxel_shader::ShaderCompositionRequest;

use super::MaterialPreparationError;

const FIELD_MARKERS: [(&str, u32); 2] = [
    ("/* fluxel:vertex-varying-fields-5 */", 5),
    ("/* fluxel:vertex-varying-fields-8 */", 8),
];
const ASSIGNMENT_TARGETS: [&str; 2] = ["output", "o"];
const ARGUMENT_MARKER: &str = "/* fluxel:vertex-varying-arguments */";

pub(super) fn specialize(
    request: &mut ShaderCompositionRequest,
    varyings: &[(u32, MaterialType)],
) -> Result<(), MaterialPreparationError> {
    if varyings.is_empty() {
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
    for unit in &mut request.source_units {
        let source = &mut unit.source;
        for (marker, location) in FIELD_MARKERS {
            if source.contains(marker) {
                replace_marker(
                    source,
                    marker,
                    &fields(varyings, location)?,
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
                    &assignments(varyings, target),
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
    ] {
        if !replaced {
            return Err(MaterialPreparationError::MissingVertexVaryingMarker { marker });
        }
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
) -> Result<String, MaterialPreparationError> {
    let mut output = String::new();
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

fn assignments(varyings: &[(u32, MaterialType)], target: &str) -> String {
    let mut output = format!(
        "let fluxel_vertex_varyings = evaluate_pbr_vertex_varyings(uv0);\n{target}.vertex_uv0 = fluxel_vertex_varyings.uv0;\n"
    );
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
