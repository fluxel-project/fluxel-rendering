//! Normalization and authoring validation for material definitions.

use std::collections::{BTreeMap, BTreeSet};

use super::canonical::{
    calls as canonical_calls, inputs as canonical_inputs, piece as canonical_piece,
    resources as canonical_resources, sorted_by_name, source_id as source_unit_id,
};
use super::names::{
    shader_resource_name, validate_generated_node_references, validate_top_level_names,
};
pub(super) use super::validation_code_nodes::FunctionSignature;
use super::validation_code_nodes::{validate_code_node_sources, visit};
use super::*;
use fluxel_shader::CallableDeclaration;
#[derive(Clone, Debug)]
pub(super) struct NormalizedMaterialIr {
    fragments: Vec<MaterialFragment>,
    code_nodes: Vec<MaterialCodeNode>,
    root_fragment: String,
    pub(super) parameters: Vec<MaterialParameter>,
    pub(super) resources: Vec<MaterialResource>,
    pub(super) texture_encoding_requirements: Vec<MaterialTextureEncodingRequirement>,
    texture_uv: BTreeMap<String, MaterialTextureUv>,
    static_options: BTreeMap<String, StaticValue>,
    pub(super) output: SurfaceOutput,
    surface_extensions: SurfaceExtensions,
    emissive_configuration: Option<EmissiveConfiguration>,
    pbr_surface_factors: Option<PbrSurfaceFactors>,
    pub(super) ambient_occlusion_configuration: Option<AmbientOcclusionConfiguration>,
    pub(super) diffuse_roughness_configuration: Option<DiffuseRoughnessConfiguration>,
    pub(super) normal_map_settings: Option<NormalMapSettings>,
    pub(super) surface_policy: SurfacePolicy,
    pub(super) transmission_extensions: Option<TransmissionExtensions>,
    transmission_texture_configuration: Option<TransmissionTextureConfiguration>,
    refraction_environment: Option<RefractionEnvironment>,
    node_graph: Option<super::node_graph::NormalizedNodeGraph>,
    vertex_uv0: Option<super::vertex_uv0::NormalizedVertexUv0>,
}
impl NormalizedMaterialIr {
    pub(super) fn from_definition(definition: &MaterialDefinition) -> Result<Self, MaterialError> {
        if !definition.surface_policy().alpha_cutoff.is_finite() {
            return Err(MaterialError::SurfaceAlphaCutoffNonFinite);
        }
        if definition.normal_map_settings().is_some()
            && !super::surface::has_tangent_space_normal(definition.output())
        {
            return Err(MaterialError::NormalMapSettingsUnsupported {
                output: definition.output().clone(),
            });
        }
        let mut fragments = definition.fragments().to_vec();
        fragments.sort_by(|left, right| left.name.cmp(&right.name));
        let mut code_nodes = definition.code_nodes().to_vec();
        code_nodes.sort_by(|left, right| left.name.cmp(&right.name));
        validate_code_node_sources(&code_nodes)?;
        let mut functions = function_signatures(&fragments, &code_nodes)?;
        let vertex_uv0 = super::vertex_uv0::validate(definition, &functions)?;
        let node_graph = definition
            .node_graph()
            .map(|graph| {
                super::node_graph::normalize_surface(
                    graph,
                    &functions,
                    definition.output(),
                    vertex_uv0.is_some(),
                    super::vertex_uv0::varying_types(vertex_uv0.as_ref()),
                )
            })
            .transpose()?;
        if node_graph.is_none()
            && matches!(
                definition.output(),
                SurfaceOutput::MetallicRoughnessUv0Graph { .. }
            )
        {
            return Err(MaterialError::PbrGraphRequired);
        }
        let graphs = node_graph
            .iter()
            .chain(vertex_uv0.iter().filter_map(|uv| uv.graph()));
        for graph in graphs.clone() {
            validate_generated_node_references(&fragments, &code_nodes, graph)?;
        }
        let root_fragment = node_graph.as_ref().map_or_else(
            || definition.root_fragment().to_owned(),
            |graph| graph.root.clone(),
        );
        if let Some(graph) = &node_graph {
            functions.insert(
                graph.root.clone(),
                FunctionSignature {
                    inputs: graph.inputs.clone(),
                    output: graph.output,
                    calls: Vec::new(),
                },
            );
        }
        let root = functions.get(root_fragment.as_str()).ok_or_else(|| {
            MaterialError::MissingFragment {
                name: root_fragment.clone(),
            }
        })?;
        validate_parameters(
            definition.parameters(),
            &functions,
            definition.static_options(),
        )?;
        let resources = merge_resources(definition.resources(), &code_nodes)?;
        validate_parameter_resources(definition.parameters(), &resources)?;
        super::surface::validate_definition(
            definition,
            root,
            &functions,
            &root_fragment,
            node_graph
                .as_ref()
                .map_or(&[][..], |graph| graph.pbr_inputs.as_slice()),
            definition.parameters(),
            &resources,
        )?;
        let texture_encoding_requirements =
            super::texture_encoding::effective_requirements(definition)?;
        super::texture_encoding::validate(&texture_encoding_requirements, &resources)?;
        let texture_uv = super::texture_uv::validate(
            definition.texture_uv(),
            definition.pbr_surface_factors(),
            definition.emissive_configuration(),
            definition.diffuse_roughness_configuration(),
            definition.ambient_occlusion_configuration(),
            definition.transmission_texture_configuration(),
        )?;
        validate_top_level_names(
            &fragments,
            &code_nodes,
            &resources,
            definition.parameters(),
            definition.static_options(),
            definition.output(),
            definition.diffuse_roughness_configuration().is_some(),
            definition.ambient_occlusion_configuration().is_some(),
            graphs,
        )?;
        validate_calls(&functions)?;
        Ok(Self {
            fragments,
            code_nodes,
            root_fragment,
            parameters: sorted_by_name(definition.parameters().to_vec(), |value| &value.name),
            resources,
            texture_encoding_requirements,
            texture_uv,
            static_options: definition.static_options().clone(),
            output: definition.output().clone(),
            surface_extensions: definition.surface_extensions().clone(),
            emissive_configuration: definition.emissive_configuration().cloned(),
            pbr_surface_factors: definition.pbr_surface_factors().cloned(),
            ambient_occlusion_configuration: definition.ambient_occlusion_configuration().cloned(),
            diffuse_roughness_configuration: definition.diffuse_roughness_configuration().cloned(),
            normal_map_settings: definition.normal_map_settings(),
            surface_policy: definition.surface_policy(),
            transmission_extensions: definition.transmission_extensions().cloned(),
            transmission_texture_configuration: definition
                .transmission_texture_configuration()
                .cloned(),
            refraction_environment: definition.refraction_environment().cloned(),
            node_graph,
            vertex_uv0,
        })
    }

    pub(super) fn variant_key(&self) -> MaterialVariantKey {
        let mut canonical = String::new();
        canonical_piece(&mut canonical, "root", &self.root_fragment);
        super::surface::canonical(&self.output, &mut canonical);
        super::vertex_uv0::canonical(&self.output, self.vertex_uv0.as_ref(), &mut canonical);
        super::surface::canonical_extensions(
            &self.surface_extensions,
            self.emissive_configuration.as_ref(),
            self.pbr_surface_factors.as_ref(),
            &mut canonical,
        );
        super::diffuse_roughness::canonical(
            self.diffuse_roughness_configuration.as_ref(),
            &mut canonical,
        );
        super::ambient_occlusion::canonical(
            self.ambient_occlusion_configuration.as_ref(),
            &mut canonical,
        );
        super::surface::canonical_normal_map_settings(self.normal_map_settings, &mut canonical);
        super::surface::canonical_policy(self.surface_policy, &mut canonical);
        super::transmission::canonical(self.transmission_extensions.as_ref(), &mut canonical);
        super::transmission::canonical_textures(
            self.transmission_texture_configuration.as_ref(),
            &mut canonical,
        );
        super::transmission::canonical_environment(
            self.refraction_environment.as_ref(),
            &mut canonical,
        );
        for fragment in &self.fragments {
            canonical_piece(&mut canonical, "fragment", &fragment.name);
            canonical_piece(&mut canonical, "output", fragment.output.wgsl());
            canonical_piece(&mut canonical, "wgsl", &fragment.wgsl);
            canonical_inputs(&mut canonical, &fragment.inputs);
            canonical_calls(&mut canonical, &fragment.calls);
        }
        for node in &self.code_nodes {
            canonical_piece(&mut canonical, "code-node", &node.name);
            canonical_piece(&mut canonical, "source-id", &node.source.id.0);
            canonical_piece(&mut canonical, "source", &node.source.source);
            canonical_piece(&mut canonical, "output", node.output.wgsl());
            canonical_inputs(&mut canonical, &node.inputs);
            canonical_calls(&mut canonical, &node.calls);
            let mut resources = node.resources.clone();
            resources.sort_by(|left, right| left.name.cmp(&right.name));
            canonical_resources(&mut canonical, &resources);
        }
        if let Some(graph) = &self.node_graph {
            canonical_piece(&mut canonical, "node-graph", &graph.source);
        }
        for parameter in &self.parameters {
            canonical_piece(&mut canonical, "parameter", &parameter.name);
            canonical_piece(&mut canonical, "parameter-type", parameter.ty.wgsl());
            canonical_piece(&mut canonical, "parameter-resource", &parameter.resource);
        }
        canonical_resources(&mut canonical, &self.resources);
        super::texture_encoding::canonical(&self.texture_encoding_requirements, &mut canonical);
        super::texture_uv::canonical(&self.texture_uv, &mut canonical);
        for (name, value) in &self.static_options {
            canonical_piece(&mut canonical, "static-name", name);
            canonical_piece(&mut canonical, "static-value", &value.canonical());
        }
        MaterialVariantKey::from_canonical(canonical)
    }

    pub(super) fn contribution(&self) -> MaterialShaderContribution {
        let mut generated = String::from("// Generated by fluxel-material.\n");
        for (name, value) in &self.static_options {
            match value {
                StaticValue::Bool(value) => {
                    generated.push_str(&format!("const {name}: bool = {value};\n"))
                }
                StaticValue::U32(value) => {
                    generated.push_str(&format!("const {name}: u32 = {value}u;\n"))
                }
            }
        }
        for resource in &self.resources {
            let shader_name = shader_resource_name(resource, &self.parameters);
            let prefix = if resource.kind.uniform_material_type().is_some() {
                "var<uniform>"
            } else {
                "var"
            };
            generated.push_str(&format!(
                "@group({}) @binding({}) {prefix} {shader_name}: {};\n",
                resource.location.group,
                resource.location.binding,
                resource.kind.wgsl()
            ));
        }
        for fragment in &self.fragments {
            generated.push_str(&fragment.wgsl);
            generated.push('\n');
        }
        if let Some(graph) = &self.node_graph {
            generated.push_str(&graph.source);
        }
        let vertex_varyings = super::vertex_uv0::varying_types(self.vertex_uv0.as_ref());
        let texture_uv_requirements = super::texture_uv::requirements(
            &self.texture_uv,
            self.pbr_surface_factors.as_ref(),
            self.diffuse_roughness_configuration.as_ref(),
            self.ambient_occlusion_configuration.as_ref(),
            self.transmission_texture_configuration.as_ref(),
        );
        super::surface::append_evaluators(
            &mut generated,
            &self.root_fragment,
            self.node_graph
                .as_ref()
                .map_or(&[][..], |graph| graph.pbr_inputs.as_slice()),
            &self.output,
            &self.surface_extensions,
            self.emissive_configuration.as_ref(),
            self.pbr_surface_factors.as_ref(),
            self.normal_map_settings,
            self.surface_policy,
            vertex_varyings,
            &self.texture_uv,
            self.ambient_occlusion_configuration.is_some(),
        );
        super::diffuse_roughness::append_evaluator(
            &mut generated,
            self.diffuse_roughness_configuration.as_ref(),
            &self.output,
            &self.texture_uv,
        );
        super::ambient_occlusion::append_evaluator(
            &mut generated,
            self.ambient_occlusion_configuration.as_ref(),
            &self.surface_extensions,
            self.pbr_surface_factors.as_ref(),
            &self.texture_uv,
        );
        super::vertex_uv0::append_evaluator(&self.output, self.vertex_uv0.as_ref(), &mut generated);
        super::transmission::append_evaluators(
            &mut generated,
            self.transmission_extensions.as_ref(),
            self.transmission_texture_configuration.as_ref(),
            &self.texture_uv,
        );
        let variant_key = self.variant_key();
        MaterialShaderContribution::new(
            source_unit_id(variant_key.as_str()),
            generated,
            self.code_nodes
                .iter()
                .map(|node| node.source.clone())
                .collect(),
            self.resources
                .iter()
                .map(|resource| MaterialBindingDeclaration {
                    name: resource.name.clone(),
                    location: resource.location,
                    kind: resource.kind,
                })
                .collect(),
            self.parameters
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
            callable_declarations(
                &self.fragments,
                &self.code_nodes,
                self.node_graph
                    .iter()
                    .chain(self.vertex_uv0.iter().filter_map(|uv| uv.graph())),
            ),
            self.static_options.clone(),
        )
        .with_vertex_varyings(vertex_varyings.to_vec())
        .with_texture_uv_requirements(texture_uv_requirements.any, texture_uv_requirements.surface)
    }
}

fn function_signatures(
    fragments: &[MaterialFragment],
    nodes: &[MaterialCodeNode],
) -> Result<BTreeMap<String, FunctionSignature>, MaterialError> {
    let mut functions = BTreeMap::new();
    for fragment in fragments {
        insert_signature(
            &mut functions,
            &fragment.name,
            &fragment.inputs,
            fragment.output,
            &fragment.calls,
        )?;
    }
    for node in nodes {
        insert_signature(
            &mut functions,
            &node.name,
            &node.inputs,
            node.output,
            &node.calls,
        )?;
    }
    Ok(functions)
}

fn insert_signature(
    functions: &mut BTreeMap<String, FunctionSignature>,
    name: &str,
    inputs: &[FragmentInput],
    output: MaterialType,
    calls: &[FragmentCall],
) -> Result<(), MaterialError> {
    let mut input_names = BTreeSet::new();
    for input in inputs {
        if !input_names.insert(input.name.as_str()) {
            return Err(MaterialError::DuplicateCallableInput {
                callable: name.into(),
                input: input.name.clone(),
            });
        }
    }
    if functions
        .insert(
            name.into(),
            FunctionSignature {
                inputs: inputs.to_vec(),
                output,
                calls: calls.to_vec(),
            },
        )
        .is_some()
    {
        return Err(MaterialError::DuplicateFragment { name: name.into() });
    }
    Ok(())
}

fn validate_parameters(
    parameters: &[MaterialParameter],
    functions: &BTreeMap<String, FunctionSignature>,
    static_options: &BTreeMap<String, StaticValue>,
) -> Result<(), MaterialError> {
    let mut names = BTreeSet::new();
    for parameter in parameters {
        if !names.insert(parameter.name.as_str()) {
            return Err(MaterialError::DuplicateParameter {
                name: parameter.name.clone(),
            });
        }
        if functions.contains_key(&parameter.name)
            || static_options.contains_key(&parameter.name)
            || parameter.name == "evaluate_material"
        {
            return Err(MaterialError::ParameterNameConflict {
                parameter: parameter.name.clone(),
            });
        }
    }
    Ok(())
}

fn merge_resources(
    definition_resources: &[MaterialResource],
    nodes: &[MaterialCodeNode],
) -> Result<Vec<MaterialResource>, MaterialError> {
    let mut resources = BTreeMap::<String, MaterialResource>::new();
    let mut locations = BTreeMap::<BindingLocation, String>::new();
    for resource in definition_resources {
        if resources.contains_key(&resource.name) {
            return Err(MaterialError::DuplicateResource {
                name: resource.name.clone(),
            });
        }
        insert_resource(&mut resources, &mut locations, resource, true)?;
    }
    for node in nodes {
        for resource in &node.resources {
            insert_resource(&mut resources, &mut locations, resource, false)?;
        }
    }
    Ok(resources.into_values().collect())
}

fn insert_resource(
    resources: &mut BTreeMap<String, MaterialResource>,
    locations: &mut BTreeMap<BindingLocation, String>,
    candidate: &MaterialResource,
    definition_resource: bool,
) -> Result<(), MaterialError> {
    if let Some(existing) = resources.get(&candidate.name) {
        if existing == candidate && !definition_resource {
            return Ok(());
        }
        return Err(if definition_resource {
            MaterialError::DuplicateResource {
                name: candidate.name.clone(),
            }
        } else {
            MaterialError::ConflictingResourceDeclaration {
                name: candidate.name.clone(),
            }
        });
    }
    if locations
        .insert(candidate.location, candidate.name.clone())
        .is_some()
    {
        return Err(MaterialError::BindingConflict {
            location: candidate.location,
        });
    }
    resources.insert(candidate.name.clone(), candidate.clone());
    Ok(())
}

fn validate_parameter_resources(
    parameters: &[MaterialParameter],
    resources: &[MaterialResource],
) -> Result<(), MaterialError> {
    let resources = resources
        .iter()
        .map(|resource| (resource.name.as_str(), resource))
        .collect::<BTreeMap<_, _>>();
    let mut parameter_resources = BTreeSet::new();
    for parameter in parameters {
        let conflicts_with_resource = resources
            .get(parameter.name.as_str())
            .is_some_and(|resource| resource.name != parameter.resource);
        if conflicts_with_resource {
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
        if resource.kind.uniform_material_type() != Some(parameter.ty) {
            return Err(MaterialError::ParameterResourceTypeMismatch {
                parameter: parameter.name.clone(),
                parameter_type: parameter.ty,
                resource_kind: resource.kind,
            });
        }
    }
    Ok(())
}

fn validate_calls(functions: &BTreeMap<String, FunctionSignature>) -> Result<(), MaterialError> {
    for (caller, function) in functions {
        for call in &function.calls {
            let callee =
                functions
                    .get(&call.fragment)
                    .ok_or_else(|| MaterialError::InvalidCall {
                        caller: caller.clone(),
                        callee: call.fragment.clone(),
                    })?;
            if callee
                .inputs
                .iter()
                .map(|input| input.ty)
                .ne(call.arguments.iter().copied())
            {
                return Err(MaterialError::CallTypeMismatch {
                    caller: caller.clone(),
                    callee: call.fragment.clone(),
                });
            }
        }
    }
    let mut visiting = BTreeSet::new();
    let mut visited = BTreeSet::new();
    for name in functions.keys() {
        visit(name, functions, &mut visiting, &mut visited)?;
    }
    Ok(())
}

fn callable_declarations<'a>(
    fragments: &[MaterialFragment],
    nodes: &[MaterialCodeNode],
    graphs: impl IntoIterator<Item = &'a super::node_graph::NormalizedNodeGraph>,
) -> Vec<CallableDeclaration> {
    let mut callables = Vec::with_capacity(fragments.len() + nodes.len());
    for (name, inputs, output) in fragments
        .iter()
        .map(|fragment| (&fragment.name, &fragment.inputs, fragment.output))
        .chain(
            nodes
                .iter()
                .map(|node| (&node.name, &node.inputs, node.output)),
        )
    {
        callables.push(CallableDeclaration {
            name: name.clone(),
            arguments: inputs.iter().map(|input| input.ty.shader_type()).collect(),
            result: Some(output.shader_type()),
        });
    }
    for graph in graphs {
        callables.extend(graph.callables.iter().cloned());
    }
    callables
}
