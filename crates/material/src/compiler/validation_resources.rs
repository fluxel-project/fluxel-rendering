//! Validation of material parameters and resource binding identities.

use std::collections::{BTreeMap, BTreeSet};

use super::validation_code_nodes::FunctionSignature;
use super::{
    BindingLocation, MaterialCodeNode, MaterialError, MaterialParameter, MaterialResource,
    StaticValue,
};

pub(super) fn validate_parameters(
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

pub(super) fn merge_resources(
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

pub(super) fn validate_parameter_resources(
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
