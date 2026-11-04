//! Normalization and authoring validation for material definitions.

use super::*;

#[derive(Clone, Debug)]
pub(super) struct NormalizedMaterialIr {
    fragments: Vec<MaterialFragment>,
    root_fragment: String,
    pub(super) parameters: Vec<MaterialParameter>,
    pub(super) resources: Vec<MaterialResource>,
    static_options: BTreeMap<String, StaticValue>,
}

impl NormalizedMaterialIr {
    pub(super) fn from_definition(definition: &MaterialDefinition) -> Result<Self, MaterialError> {
        let mut fragments = definition.fragments.clone();
        fragments.sort_by(|left, right| left.name.cmp(&right.name));
        let mut names = BTreeSet::new();
        for fragment in &fragments {
            if !names.insert(fragment.name.as_str()) {
                return Err(MaterialError::DuplicateFragment {
                    name: fragment.name.clone(),
                });
            }
        }
        let fragment_map = fragments
            .iter()
            .map(|fragment| (fragment.name.as_str(), fragment))
            .collect::<BTreeMap<_, _>>();
        let root = fragment_map
            .get(definition.root_fragment.as_str())
            .ok_or_else(|| MaterialError::MissingFragment {
                name: definition.root_fragment.clone(),
            })?;
        if definition.output == SurfaceOutput::Color && root.output != MaterialType::Vec4 {
            return Err(MaterialError::UnsupportedOutput {
                output: definition.output,
            });
        }
        if !root.inputs.is_empty() {
            return Err(MaterialError::RootInputsUnsupported {
                fragment: root.name.clone(),
            });
        }
        let mut parameter_names = BTreeSet::new();
        for parameter in &definition.parameters {
            if !parameter_names.insert(parameter.name.as_str()) {
                return Err(MaterialError::DuplicateParameter {
                    name: parameter.name.clone(),
                });
            }
        }
        let mut resource_names = BTreeSet::new();
        let mut locations = BTreeSet::new();
        for resource in &definition.resources {
            if !resource_names.insert(resource.name.as_str()) {
                return Err(MaterialError::DuplicateResource {
                    name: resource.name.clone(),
                });
            }
            if !locations.insert(resource.location) {
                return Err(MaterialError::BindingConflict {
                    location: resource.location,
                });
            }
        }
        let resources = definition
            .resources
            .iter()
            .map(|resource| (resource.name.as_str(), resource))
            .collect::<BTreeMap<_, _>>();
        let mut parameter_resources = BTreeSet::new();
        for parameter in &definition.parameters {
            let conflicts_with_resource = resources
                .get(parameter.name.as_str())
                .is_some_and(|resource| resource.name != parameter.resource);
            if conflicts_with_resource
                || fragment_map.contains_key(parameter.name.as_str())
                || definition
                    .static_options
                    .contains_key(parameter.name.as_str())
                || parameter.name == "evaluate_material"
            {
                return Err(MaterialError::ParameterNameConflict {
                    parameter: parameter.name.clone(),
                });
            }
            let resource = resources.get(parameter.resource.as_str()).ok_or_else(|| {
                MaterialError::ParameterResourceMissing {
                    parameter: parameter.name.clone(),
                    resource: parameter.resource.clone(),
                }
            })?;
            if !parameter_resources.insert(resource.name.as_str()) {
                return Err(MaterialError::ParameterResourceConflict {
                    resource: resource.name.clone(),
                });
            }
            if parameter.ty != MaterialType::Vec4
                || resource.kind != MaterialResourceKind::UniformVec4
            {
                return Err(MaterialError::ParameterResourceTypeMismatch {
                    parameter: parameter.name.clone(),
                    parameter_type: parameter.ty,
                    resource_kind: resource.kind,
                });
            }
        }
        validate_top_level_names(
            &fragments,
            &definition.resources,
            &definition.parameters,
            &definition.static_options,
        )?;
        for fragment in &fragments {
            for call in &fragment.calls {
                let callee = fragment_map.get(call.fragment.as_str()).ok_or_else(|| {
                    MaterialError::InvalidCall {
                        caller: fragment.name.clone(),
                        callee: call.fragment.clone(),
                    }
                })?;
                if callee
                    .inputs
                    .iter()
                    .map(|input| input.ty)
                    .ne(call.arguments.iter().copied())
                {
                    return Err(MaterialError::CallTypeMismatch {
                        caller: fragment.name.clone(),
                        callee: call.fragment.clone(),
                    });
                }
            }
        }
        let mut visiting = BTreeSet::new();
        let mut visited = BTreeSet::new();
        for fragment in &fragments {
            visit(&fragment.name, &fragment_map, &mut visiting, &mut visited)?;
        }
        Ok(Self {
            fragments,
            root_fragment: definition.root_fragment.clone(),
            parameters: sorted_by_name(definition.parameters.clone(), |value| &value.name),
            resources: sorted_by_name(definition.resources.clone(), |value| &value.name),
            static_options: definition.static_options.clone(),
        })
    }

    pub(super) fn variant_key(&self) -> MaterialVariantKey {
        let mut canonical = String::new();
        canonical_piece(&mut canonical, "root", &self.root_fragment);
        for fragment in &self.fragments {
            canonical_piece(&mut canonical, "fragment", &fragment.name);
            canonical_piece(&mut canonical, "output", fragment.output.wgsl());
            canonical_piece(&mut canonical, "wgsl", &fragment.wgsl);
            for input in &fragment.inputs {
                canonical_piece(&mut canonical, "input-name", &input.name);
                canonical_piece(&mut canonical, "input-type", input.ty.wgsl());
            }
            for call in &fragment.calls {
                canonical_piece(&mut canonical, "call", &call.fragment);
                for argument in &call.arguments {
                    canonical_piece(&mut canonical, "argument", argument.wgsl());
                }
            }
        }
        for parameter in &self.parameters {
            canonical_piece(&mut canonical, "parameter", &parameter.name);
            canonical_piece(&mut canonical, "parameter-type", parameter.ty.wgsl());
            canonical_piece(&mut canonical, "parameter-resource", &parameter.resource);
        }
        for resource in &self.resources {
            canonical_piece(&mut canonical, "resource", &resource.name);
            canonical_piece(&mut canonical, "resource-kind", resource.kind.wgsl());
            canonical_piece(
                &mut canonical,
                "resource-group",
                &resource.location.group.to_string(),
            );
            canonical_piece(
                &mut canonical,
                "resource-binding",
                &resource.location.binding.to_string(),
            );
        }
        for (name, value) in &self.static_options {
            canonical_piece(&mut canonical, "static-name", name);
            canonical_piece(&mut canonical, "static-value", &value.canonical());
        }
        MaterialVariantKey {
            canonical,
            material: None,
        }
    }

    pub(super) fn contribution(&self) -> MaterialShaderContribution {
        let mut source = String::from("// Generated by fluxel-material.\n");
        for (name, value) in &self.static_options {
            match value {
                StaticValue::Bool(value) => {
                    source.push_str(&format!("const {name}: bool = {value};\n"))
                }
                StaticValue::U32(value) => {
                    source.push_str(&format!("const {name}: u32 = {value}u;\n"))
                }
            }
        }
        for resource in &self.resources {
            let shader_name = self
                .parameters
                .iter()
                .find(|parameter| parameter.resource == resource.name)
                .map_or(resource.name.as_str(), |parameter| parameter.name.as_str());
            let prefix = match resource.kind {
                MaterialResourceKind::UniformVec4 => "var<uniform>",
                MaterialResourceKind::Texture2dF32 => "var",
            };
            source.push_str(&format!(
                "@group({}) @binding({}) {prefix} {}: {};\n",
                resource.location.group,
                resource.location.binding,
                shader_name,
                resource.kind.wgsl()
            ));
        }
        for fragment in &self.fragments {
            source.push_str(&fragment.wgsl);
            source.push('\n');
        }
        source.push_str(&format!(
            "fn evaluate_material() -> vec4<f32> {{\n    return {}();\n}}\n",
            self.root_fragment
        ));
        let variant_key = self.variant_key();
        MaterialShaderContribution {
            source_id: source_unit_id(variant_key.as_str()),
            source,
            bindings: self
                .resources
                .iter()
                .map(|resource| MaterialBindingDeclaration {
                    name: resource.name.clone(),
                    location: resource.location,
                    kind: resource.kind,
                })
                .collect(),
            parameter_bindings: self
                .parameters
                .iter()
                .map(|parameter| {
                    let resource = self
                        .resources
                        .iter()
                        .find(|resource| resource.name == parameter.resource)
                        .expect("validated parameter resource");
                    MaterialParameterBinding {
                        parameter: parameter.name.clone(),
                        resource: resource.name.clone(),
                        location: resource.location,
                        ty: parameter.ty,
                    }
                })
                .collect(),
            callables: self
                .fragments
                .iter()
                .map(|fragment| CallableDeclaration {
                    name: fragment.name.clone(),
                    arguments: fragment
                        .inputs
                        .iter()
                        .map(|input| input.ty.shader_type())
                        .collect(),
                    result: Some(fragment.output.shader_type()),
                })
                .collect(),
            static_options: self.static_options.clone(),
        }
    }
}

fn validate_top_level_names(
    fragments: &[MaterialFragment],
    resources: &[MaterialResource],
    parameters: &[MaterialParameter],
    static_options: &BTreeMap<String, StaticValue>,
) -> Result<(), MaterialError> {
    let mut names = BTreeSet::from([String::from("evaluate_material")]);
    let mut insert = |name: &str| {
        if names.insert(name.to_owned()) {
            Ok(())
        } else {
            Err(MaterialError::TopLevelNameConflict {
                name: name.to_owned(),
            })
        }
    };
    for name in static_options.keys() {
        insert(name)?;
    }
    for resource in resources {
        let shader_name = parameters
            .iter()
            .find(|parameter| parameter.resource == resource.name)
            .map_or(resource.name.as_str(), |parameter| parameter.name.as_str());
        insert(shader_name)?;
    }
    for fragment in fragments {
        for name in wgsl_top_level_names(&fragment.wgsl)? {
            insert(&name)?;
        }
    }
    Ok(())
}

fn source_unit_id(canonical: &str) -> String {
    format!("material:{:x}", Sha256::digest(canonical.as_bytes()))
}

fn visit(
    name: &str,
    fragments: &BTreeMap<&str, &MaterialFragment>,
    visiting: &mut BTreeSet<String>,
    visited: &mut BTreeSet<String>,
) -> Result<(), MaterialError> {
    if visited.contains(name) {
        return Ok(());
    }
    if !visiting.insert(name.to_owned()) {
        return Err(MaterialError::CallCycle {
            fragment: name.to_owned(),
        });
    }
    let fragment = fragments.get(name).expect("validated root and calls");
    for call in &fragment.calls {
        visit(&call.fragment, fragments, visiting, visited)?;
    }
    visiting.remove(name);
    visited.insert(name.to_owned());
    Ok(())
}

fn sorted_by_name<T>(mut values: Vec<T>, name: impl Fn(&T) -> &String) -> Vec<T> {
    values.sort_by(|left, right| name(left).cmp(name(right)));
    values
}

fn canonical_piece(output: &mut String, label: &str, value: &str) {
    output.push_str(label);
    output.push(':');
    output.push_str(&value.len().to_string());
    output.push(':');
    output.push_str(value);
    output.push(';');
}
