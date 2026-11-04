//! WGSL composition, validation, and reflection implementation.

use super::*;

fn stage(x: naga::ShaderStage) -> Option<ShaderStage> {
    match x {
        naga::ShaderStage::Vertex => Some(ShaderStage::Vertex),
        naga::ShaderStage::Fragment => Some(ShaderStage::Fragment),
        naga::ShaderStage::Compute => Some(ShaderStage::Compute),
        _ => None,
    }
}
pub(super) fn build(
    r: &ShaderCompositionRequest,
    key: ShaderArtifactKey,
) -> Result<ShaderArtifact, ShaderError> {
    if r.source_units.is_empty() {
        return Err(ShaderError::EmptySource);
    }
    if r.entry_points.is_empty() {
        return Err(ShaderError::EmptyEntries);
    }
    let mut entry_ids = std::collections::HashSet::new();
    for entry in &r.entry_points {
        if !entry_ids.insert(entry) {
            return Err(ShaderError::DuplicateEntry(entry.clone()));
        }
    }
    let mut ids = BTreeSet::new();
    let mut src = String::new();
    let mut source_ranges = Vec::new();
    for x in &r.source_units {
        if !ids.insert(x.id.clone()) {
            return Err(ShaderError::DuplicateSource(x.id.0.clone()));
        }
        let start = src.len();
        src += &x.source;
        source_ranges.push((start..src.len(), x));
        src.push('\n')
    }
    let locate = |offset: Option<usize>| -> (Option<SourceUnitId>, Option<u32>, Option<u32>) {
        let Some(offset) = offset else {
            return (None, None, None);
        };
        let Some((range, unit)) = source_ranges
            .iter()
            .find(|(range, _)| range.start <= offset && offset <= range.end)
        else {
            return (None, None, None);
        };
        let local = offset - range.start;
        let Some(before) = unit.source.get(..local) else {
            return (None, None, None);
        };
        let line = before.bytes().filter(|byte| *byte == b'\n').count() + 1;
        let column = before.rsplit('\n').next().unwrap_or("").chars().count() + 1;
        (
            Some(unit.id.clone()),
            u32::try_from(line).ok(),
            u32::try_from(column).ok(),
        )
    };
    let m = naga::front::wgsl::parse_str(&src).map_err(|e| {
        let (source, line, column) = locate(
            e.labels()
                .next()
                .and_then(|(span, _)| span.to_range())
                .map(|range| range.start),
        );
        ShaderError::Parse {
            source,
            line,
            column,
            message: e.to_string(),
        }
    })?;
    let i = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&m)
    .map_err(|e| {
        let (source, line, column) = locate(
            e.spans()
                .next()
                .and_then(|(span, _)| span.to_range())
                .map(|range| range.start),
        );
        ShaderError::Validate {
            source,
            line,
            column,
            message: e.to_string(),
        }
    })?;
    let mut requirements = r.requirements.clone();
    requirements
        .naga_capabilities
        .extend(required_naga_capabilities(&m).map_err(|e| {
            let (source, line, column) = locate(
                e.spans()
                    .next()
                    .and_then(|(span, _)| span.to_range())
                    .map(|range| range.start),
            );
            ShaderError::Validate {
                source,
                line,
                column,
                message: e.to_string(),
            }
        })?);
    for (capability, feature) in [
        (NagaCapability::PrimitiveIndex, "primitive-index"),
        (NagaCapability::EarlyDepthTest, "shader-early-depth-test"),
        (NagaCapability::RayQuery, "ray-query"),
    ] {
        if requirements.naga_capabilities.contains(&capability) {
            requirements.names.insert(feature.into());
        }
    }
    let mut callable_names = BTreeSet::new();
    for callable in &r.callables {
        if !callable_names.insert(callable.name.as_str()) {
            return Err(ShaderError::CallableMismatch {
                name: callable.name.clone(),
            });
        }
        let Some((_, function)) = m
            .functions
            .iter()
            .find(|(_, f)| f.name.as_deref() == Some(&callable.name))
        else {
            return Err(ShaderError::CallableMismatch {
                name: callable.name.clone(),
            });
        };
        let actual_arguments = function
            .arguments
            .iter()
            .map(|arg| vertex_type(&m.types[arg.ty].inner))
            .collect::<Result<Vec<_>, _>>();
        let actual_result = function
            .result
            .as_ref()
            .map(|result| vertex_type(&m.types[result.ty].inner))
            .transpose();
        if actual_arguments.as_ref().ok() != Some(&callable.arguments)
            || actual_result.as_ref().ok() != Some(&callable.result)
        {
            return Err(ShaderError::CallableMismatch {
                name: callable.name.clone(),
            });
        }
    }
    let mut sel = Vec::new();
    for e in &r.entry_points {
        let Some(x) = m
            .entry_points
            .iter()
            .enumerate()
            .find(|(_, x)| stage(x.stage) == Some(e.stage) && x.name == e.name)
        else {
            return Err(ShaderError::MissingEntry(e.clone()));
        };
        sel.push(x)
    }
    if sel
        .iter()
        .filter(|(_, entry)| entry.stage == naga::ShaderStage::Vertex)
        .nth(1)
        .is_some()
    {
        return Err(ShaderError::MultipleVertexEntries);
    }
    let mut bs = Vec::new();
    for (h, g) in m.global_variables.iter() {
        let Some(p) = g.binding else { continue };
        let (mut v, mut rd, mut wr, mut q, mut a) = (BTreeSet::new(), false, false, false, false);
        let mut entry_access = BTreeMap::new();
        for (n, e) in &sel {
            let u = i.get_entry_point(*n)[h];
            if !u.is_empty() {
                let entry_stage = stage(e.stage).expect("selected stages were validated");
                v.insert(entry_stage);
                rd |= u.contains(naga::valid::GlobalUse::READ);
                wr |= u.contains(naga::valid::GlobalUse::WRITE);
                q |= u.contains(naga::valid::GlobalUse::QUERY);
                a |= u.contains(naga::valid::GlobalUse::ATOMIC);
                entry_access.insert(
                    EntryPoint::new(&e.name, entry_stage),
                    BindingAccess {
                        read: u.contains(naga::valid::GlobalUse::READ),
                        write: u.contains(naga::valid::GlobalUse::WRITE),
                        query: u.contains(naga::valid::GlobalUse::QUERY),
                        atomic: u.contains(naga::valid::GlobalUse::ATOMIC),
                    },
                );
            }
        }
        if !v.is_empty() {
            let kind = kind(&m, g)?;
            bs.push(ReflectedBinding {
                group: p.group,
                binding: p.binding,
                kind,
                shape: binding_shape(&m, g),
                visibility: v,
                entry_access,
                read: rd,
                write: wr,
                query: q,
                atomic: a,
            })
        }
    }
    let mut declared_bindings = BTreeSet::new();
    for x in &r.bindings {
        if !declared_bindings.insert((x.group, x.binding)) {
            return Err(ShaderError::BindingMismatch {
                group: x.group,
                binding: x.binding,
            });
        }
        if !bs.iter().any(|b| {
            b.group == x.group
                && b.binding == x.binding
                && b.kind == x.kind
                && x.shape.is_none_or(|shape| b.shape == Some(shape))
                && x.visibility
                    .as_ref()
                    .is_none_or(|visibility| &b.visibility == visibility)
                && x.access.is_none_or(|access| {
                    access
                        == BindingAccess {
                            read: b.read,
                            write: b.write,
                            query: b.query,
                            atomic: b.atomic,
                        }
                })
        }) {
            return Err(ShaderError::BindingMismatch {
                group: x.group,
                binding: x.binding,
            });
        }
    }
    for binding in &bs {
        if !declared_bindings.contains(&(binding.group, binding.binding)) {
            return Err(ShaderError::BindingMismatch {
                group: binding.group,
                binding: binding.binding,
            });
        }
    }
    let mut vi = Vec::new();
    for (_, e) in sel
        .iter()
        .filter(|(_, e)| e.stage == naga::ShaderStage::Vertex)
    {
        for x in &e.function.arguments {
            locs(&m, x.ty, x.binding.as_ref(), &mut vi)?
        }
    }
    let mut locations = BTreeSet::new();
    let mut semantics = BTreeSet::new();
    for x in &r.vertex_inputs {
        if !locations.insert(x.location) || !semantics.insert(x.semantic.as_str()) {
            return Err(ShaderError::VertexMismatch {
                semantic: x.semantic.clone(),
                location: x.location,
            });
        }
        if !vi.iter().any(|v| v.location == x.location && v.ty == x.ty) {
            return Err(ShaderError::VertexMismatch {
                semantic: x.semantic.clone(),
                location: x.location,
            });
        }
    }
    if !r.vertex_inputs.is_empty() {
        for reflected in &vi {
            if !locations.contains(&reflected.location) {
                return Err(ShaderError::VertexMismatch {
                    semantic: String::new(),
                    location: reflected.location,
                });
            }
        }
    }
    Ok(ShaderArtifact {
        source: src,
        source_map: source_ranges
            .iter()
            .map(|(bytes, unit)| SourceSpan {
                source: unit.id.clone(),
                bytes: bytes.clone(),
            })
            .collect(),
        interface: ShaderInterface {
            entries: r.entry_points.clone(),
            bindings: bs,
            vertex_inputs: vi,
        },
        requirements,
        key,
    })
}
fn binding_shape(module: &naga::Module, global: &naga::GlobalVariable) -> Option<BindingShape> {
    match (global.space, &module.types[global.ty].inner) {
        (naga::AddressSpace::Uniform, inner) => vertex_type(inner).ok().map(BindingShape::Uniform),
        (
            naga::AddressSpace::Handle,
            naga::TypeInner::Image {
                dim: naga::ImageDimension::D2,
                arrayed: false,
                class:
                    naga::ImageClass::Sampled {
                        kind: naga::ScalarKind::Float,
                        multi: false,
                    },
            },
        ) => Some(BindingShape::Texture2dF32),
        _ => None,
    }
}
fn locs(
    m: &naga::Module,
    t: naga::Handle<naga::Type>,
    b: Option<&naga::Binding>,
    o: &mut Vec<ReflectedVertexInput>,
) -> Result<(), ShaderError> {
    if let Some(naga::Binding::Location { location, .. }) = b {
        o.push(ReflectedVertexInput {
            location: *location,
            ty: vertex_type(&m.types[t].inner)?,
        })
    } else if let naga::TypeInner::Struct { members, .. } = &m.types[t].inner {
        for x in members {
            locs(m, x.ty, x.binding.as_ref(), o)?
        }
    }
    Ok(())
}
fn vertex_type(inner: &naga::TypeInner) -> Result<VertexType, ShaderError> {
    fn scalar(scalar: naga::Scalar) -> Result<ScalarType, ShaderError> {
        match (scalar.kind, scalar.width) {
            (naga::ScalarKind::Float, 4) => Ok(ScalarType::F32),
            (naga::ScalarKind::Sint, 4) => Ok(ScalarType::I32),
            (naga::ScalarKind::Uint, 4) => Ok(ScalarType::U32),
            (naga::ScalarKind::Float, 2) => Ok(ScalarType::F16),
            _ => Err(ShaderError::UnsupportedVertexType),
        }
    }
    match inner {
        naga::TypeInner::Scalar(value) => Ok(VertexType::Scalar(scalar(*value)?)),
        naga::TypeInner::Vector {
            size,
            scalar: value,
        } => {
            let value = scalar(*value)?;
            Ok(match size {
                naga::VectorSize::Bi => VertexType::Vec2(value),
                naga::VectorSize::Tri => VertexType::Vec3(value),
                naga::VectorSize::Quad => VertexType::Vec4(value),
            })
        }
        _ => Err(ShaderError::UnsupportedVertexType),
    }
}
fn kind(m: &naga::Module, g: &naga::GlobalVariable) -> Result<BindingKind, ShaderError> {
    match g.space {
        naga::AddressSpace::Uniform => Ok(BindingKind::Uniform),
        naga::AddressSpace::Storage { access } if access.contains(naga::StorageAccess::STORE) => {
            Ok(BindingKind::StorageReadWrite)
        }
        naga::AddressSpace::Storage { .. } => Ok(BindingKind::StorageRead),
        naga::AddressSpace::Handle => match m.types[g.ty].inner {
            naga::TypeInner::Sampler { .. } => Ok(BindingKind::Sampler),
            naga::TypeInner::Image {
                class: naga::ImageClass::Storage { .. },
                ..
            } => Ok(BindingKind::StorageTexture),
            naga::TypeInner::Image { .. } => Ok(BindingKind::Texture),
            _ => Err(ShaderError::UnsupportedBinding),
        },
        _ => Err(ShaderError::UnsupportedBinding),
    }
}
