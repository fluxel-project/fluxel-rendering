//! Canonical material-variant serialization helpers.

use sha2::{Digest, Sha256};

use super::{FragmentCall, FragmentInput, MaterialResource};

/// Sorts values by their stable material name before canonical encoding.
pub(super) fn sorted_by_name<T>(mut values: Vec<T>, name: impl Fn(&T) -> &String) -> Vec<T> {
    values.sort_by(|left, right| name(left).cmp(name(right)));
    values
}

/// Appends one length-delimited field to a canonical encoding.
pub(super) fn piece(output: &mut String, label: &str, value: &str) {
    output.push_str(label);
    output.push(':');
    output.push_str(&value.len().to_string());
    output.push(':');
    output.push_str(value);
    output.push(';');
}

/// Appends callable input declarations to a canonical encoding.
pub(super) fn inputs(output: &mut String, inputs: &[FragmentInput]) {
    for input in inputs {
        piece(output, "input-name", &input.name);
        piece(output, "input-type", input.ty.wgsl());
    }
}

/// Appends callable edges to a canonical encoding.
pub(super) fn calls(output: &mut String, calls: &[FragmentCall]) {
    for call in calls {
        piece(output, "call", &call.fragment);
        for argument in &call.arguments {
            piece(output, "argument", argument.wgsl());
        }
    }
}

/// Appends logical resources to a canonical encoding.
pub(super) fn resources(output: &mut String, resources: &[MaterialResource]) {
    for resource in resources {
        piece(output, "resource", &resource.name);
        piece(output, "resource-kind", resource.kind.wgsl());
        piece(
            output,
            "resource-group",
            &resource.location.group.to_string(),
        );
        piece(
            output,
            "resource-binding",
            &resource.location.binding.to_string(),
        );
    }
}

/// Derives the compact source-unit identity from complete canonical facts.
pub(super) fn source_id(canonical: &str) -> String {
    format!("material:{:x}", Sha256::digest(canonical.as_bytes()))
}
