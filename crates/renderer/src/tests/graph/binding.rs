use super::*;
use fluxel_rhi::api::{
    AccessMask, BufferDesc, StageMask, TextureAspects, TextureDimension, TextureFormat,
    TextureUsage,
};

use crate::graph::{compiler::GraphCompiler, definition::GraphBuilder};

fn descriptor() -> BufferDesc {
    BufferDesc {
        label: None,
        size: 16,
        usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
        memory: MemoryClass::Device,
    }
}

fn use_state() -> BufferUses {
    BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    }
}

fn texture_desc() -> TextureDesc {
    TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [8, 8, 1],
        array_layers: 1,
        format: TextureFormat::Rgba8Unorm,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
        view_formats: Vec::new(),
    }
}

fn texture_manifest() -> TextureBindingManifest {
    let mut graph = GraphBuilder::default();
    let texture = graph.create_texture(texture_desc()).unwrap();
    graph
        .add_pass("draw", |pass| {
            let _ = pass.write_texture_with_use(
                texture,
                TextureRange::new(TextureAspects::COLOR, 0..1, 0..1),
                TextureUses {
                    stages: StageMask::COLOR_OUTPUT,
                    accesses: AccessMask::COLOR_WRITE,
                },
                ImageLayout::ColorAttachment,
                TextureUsage::RENDER_ATTACHMENT,
            )?;
            pass.side_effect();
            Ok(())
        })
        .unwrap();
    GraphCompiler
        .compile(&graph)
        .unwrap()
        .texture_bindings()
        .clone()
}

fn four_texture_manifest() -> TextureBindingManifest {
    let mut graph = GraphBuilder::default();
    let first = graph.create_texture(texture_desc()).unwrap();
    let left = graph.create_texture(texture_desc()).unwrap();
    let right = graph.create_texture(texture_desc()).unwrap();
    let last = graph.create_texture(texture_desc()).unwrap();
    let write = TextureUses {
        stages: StageMask::COLOR_OUTPUT,
        accesses: AccessMask::COLOR_WRITE,
    };
    let range = TextureRange::new(TextureAspects::COLOR, 0..1, 0..1);

    graph
        .add_pass("first", |pass| {
            let _ = pass.write_texture_with_use(
                first,
                range.clone(),
                write,
                ImageLayout::ColorAttachment,
                TextureUsage::RENDER_ATTACHMENT,
            )?;
            pass.side_effect();
            Ok(())
        })
        .unwrap();
    graph
        .add_pass("middle", |pass| {
            let _ = pass.write_texture_with_use(
                left,
                range.clone(),
                write,
                ImageLayout::ColorAttachment,
                TextureUsage::RENDER_ATTACHMENT,
            )?;
            let _ = pass.write_texture_with_use(
                right,
                range.clone(),
                write,
                ImageLayout::ColorAttachment,
                TextureUsage::RENDER_ATTACHMENT,
            )?;
            pass.side_effect();
            Ok(())
        })
        .unwrap();
    graph
        .add_pass("last", |pass| {
            let _ = pass.write_texture_with_use(
                last,
                range,
                write,
                ImageLayout::ColorAttachment,
                TextureUsage::RENDER_ATTACHMENT,
            )?;
            pass.side_effect();
            Ok(())
        })
        .unwrap();
    GraphCompiler
        .compile(&graph)
        .unwrap()
        .texture_bindings()
        .clone()
}

fn texture_bindings(manifest: &TextureBindingManifest, device: DeviceId) -> Vec<TextureBinding> {
    manifest
        .requirements
        .iter()
        .map(|requirement| TextureBinding {
            initial: requirement
                .initial
                .expect("fixture texture state is declared")
                .0,
            layout: requirement
                .initial
                .expect("fixture texture state is declared")
                .1,
            slot: requirement.slot,
            instance: ResourceInstanceId::fresh().unwrap(),
            device,
            desc: requirement.desc.clone(),
        })
        .collect()
}

fn overlapping_manifest() -> BufferBindingManifest {
    let mut graph = GraphBuilder::default();
    let source = graph.import_buffer_desc(descriptor(), use_state()).unwrap();
    let target = graph.create_buffer_desc(descriptor()).unwrap();
    let (_, target) = graph
        .add_pass("copy", |pass| {
            pass.read_with_use(source, 0..16, use_state(), BufferUsage::COPY_SRC)?;
            pass.write_with_use(
                target,
                0..16,
                BufferUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_WRITE,
                },
                BufferUsage::COPY_DST,
            )
        })
        .unwrap();
    graph
        .add_pass("observe", |pass| {
            pass.read_with_use(target, 0..16, use_state(), BufferUsage::COPY_SRC)?;
            pass.side_effect();
            Ok(())
        })
        .unwrap();
    GraphCompiler
        .compile(&graph)
        .unwrap()
        .buffer_bindings()
        .clone()
}

fn bindings(manifest: &BufferBindingManifest, device: DeviceId) -> Vec<BufferBinding> {
    manifest
        .requirements()
        .iter()
        .map(|requirement| BufferBinding {
            slot: requirement.slot,
            instance: ResourceInstanceId::fresh().unwrap(),
            device,
            size: requirement.size,
            usage: requirement.usage,
            memory: requirement.memory,
            initial: requirement
                .initial
                .expect("fixture buffer state is declared"),
        })
        .collect()
}

#[test]
fn accepts_complete_compatible_frame_bindings() {
    let manifest = overlapping_manifest();
    let device = DeviceId::allocate().unwrap();
    manifest
        .validate(device, &bindings(&manifest, device))
        .unwrap();
}

#[test]
fn refuses_live_resources_without_explicit_rhi_use_contracts() {
    let mut graph = GraphBuilder::default();
    let buffer = graph.create_buffer_desc(descriptor()).unwrap();
    graph
        .add_pass("legacy-use", |pass| {
            let _ = pass.write(buffer, 0..16)?;
            pass.side_effect();
            Ok(())
        })
        .unwrap();
    let manifest = GraphCompiler
        .compile(&graph)
        .unwrap()
        .buffer_bindings()
        .clone();
    let device = DeviceId::allocate().unwrap();
    assert_eq!(
        manifest.validate(device, &bindings(&manifest, device)),
        Err(BindingError::UndeclaredUse)
    );
}

#[test]
fn rejects_missing_extra_duplicate_foreign_and_mismatched_bindings() {
    let manifest = overlapping_manifest();
    let device = DeviceId::allocate().unwrap();
    let valid = bindings(&manifest, device);

    assert_eq!(
        manifest.validate(device, &valid[..1]),
        Err(BindingError::MissingResource)
    );
    let mut extra = valid.clone();
    extra.push(valid[0].clone());
    assert_eq!(
        manifest.validate(device, &extra),
        Err(BindingError::DuplicateSlot)
    );
    let mut undeclared = valid.clone();
    undeclared[0].slot.resource += 100;
    assert_eq!(
        manifest.validate(device, &undeclared),
        Err(BindingError::UndeclaredResource)
    );
    let mut foreign = valid.clone();
    foreign[0].device = DeviceId::allocate().unwrap();
    assert_eq!(
        manifest.validate(device, &foreign),
        Err(BindingError::WrongDevice)
    );
    let mut short = valid.clone();
    short[0].size -= 1;
    assert_eq!(
        manifest.validate(device, &short),
        Err(BindingError::IncompatibleDescriptor)
    );
    let mut wrong_usage = valid.clone();
    wrong_usage[0].usage = BufferUsage(0);
    assert_eq!(
        manifest.validate(device, &wrong_usage),
        Err(BindingError::IncompatibleDescriptor)
    );
    let mut wrong_initial = valid;
    wrong_initial[0].initial = BufferUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    };
    assert_eq!(
        manifest.validate(device, &wrong_initial),
        Err(BindingError::IncompatibleIncomingState)
    );
}

#[test]
fn rejects_new_alias_between_live_graph_resources() {
    let manifest = overlapping_manifest();
    let device = DeviceId::allocate().unwrap();
    let mut bindings = bindings(&manifest, device);
    bindings[1].instance = bindings[0].instance;
    assert_eq!(
        manifest.validate(device, &bindings),
        Err(BindingError::AliasedLiveResources)
    );
}

#[test]
fn rejects_new_alias_even_when_lifetimes_do_not_overlap_without_alias_plan() {
    let mut graph = GraphBuilder::default();
    let first = graph.create_buffer_desc(descriptor()).unwrap();
    let second = graph.create_buffer_desc(descriptor()).unwrap();
    graph
        .add_pass("first-use", |pass| {
            let _ = pass.write_with_use(
                first,
                0..16,
                BufferUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_WRITE,
                },
                BufferUsage::COPY_DST,
            )?;
            pass.side_effect();
            Ok(())
        })
        .unwrap();
    graph
        .add_pass("second-use", |pass| {
            let _ = pass.write_with_use(
                second,
                0..16,
                BufferUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_WRITE,
                },
                BufferUsage::COPY_DST,
            )?;
            pass.side_effect();
            Ok(())
        })
        .unwrap();
    let manifest = GraphCompiler
        .compile(&graph)
        .unwrap()
        .buffer_bindings()
        .clone();
    let device = DeviceId::allocate().unwrap();
    let mut bindings = bindings(&manifest, device);
    bindings[1].instance = bindings[0].instance;
    assert_eq!(
        manifest.validate(device, &bindings),
        Err(BindingError::AliasedResourcesRequireRecompile)
    );
}

#[test]
fn texture_manifest_checks_description_ranges_device_and_incoming_layout() {
    let manifest = texture_manifest();
    let device = DeviceId::allocate().unwrap();
    let valid = texture_bindings(&manifest, device);
    manifest.validate(device, &valid).unwrap();

    let mut foreign = valid.clone();
    foreign[0].device = DeviceId::allocate().unwrap();
    assert_eq!(
        manifest.validate(device, &foreign),
        Err(BindingError::WrongDevice)
    );

    let mut wrong_extent = valid.clone();
    wrong_extent[0].desc.extent[0] += 1;
    assert_eq!(
        manifest.validate(device, &wrong_extent),
        Err(BindingError::IncompatibleDescriptor)
    );

    let mut wrong_layout = valid;
    wrong_layout[0].layout = ImageLayout::General;
    assert_eq!(
        manifest.validate(device, &wrong_layout),
        Err(BindingError::IncompatibleIncomingState)
    );
}

#[test]
fn texture_manifest_accepts_shuffled_bindings_and_retains_input_guards() {
    let manifest = four_texture_manifest();
    let device = DeviceId::allocate().unwrap();
    let valid = texture_bindings(&manifest, device);

    let mut shuffled = valid.clone();
    shuffled.rotate_left(1);
    manifest.validate(device, &shuffled).unwrap();

    assert_eq!(
        manifest.validate(device, &valid[..3]),
        Err(BindingError::MissingResource)
    );
    let mut duplicate = valid.clone();
    duplicate.push(valid[0].clone());
    assert_eq!(
        manifest.validate(device, &duplicate),
        Err(BindingError::DuplicateSlot)
    );
    let mut undeclared = valid.clone();
    undeclared[0].slot.resource += 100;
    assert_eq!(
        manifest.validate(device, &undeclared),
        Err(BindingError::UndeclaredResource)
    );
    let mut wrong_device = valid.clone();
    wrong_device[0].device = DeviceId::allocate().unwrap();
    assert_eq!(
        manifest.validate(device, &wrong_device),
        Err(BindingError::WrongDevice)
    );
    let mut wrong_incoming = valid;
    wrong_incoming[0].layout = ImageLayout::General;
    assert_eq!(
        manifest.validate(device, &wrong_incoming),
        Err(BindingError::IncompatibleIncomingState)
    );
}

#[test]
fn texture_alias_priority_uses_the_first_requirement_pair() {
    let manifest = four_texture_manifest();
    let requirements = manifest.requirements();
    assert_eq!(requirements.len(), 4);
    assert!(requirements[0].last_pass < requirements[3].first_pass);
    assert!(requirements[1].first_pass <= requirements[2].last_pass);
    assert!(requirements[2].first_pass <= requirements[1].last_pass);

    let device = DeviceId::allocate().unwrap();
    let mut bindings = texture_bindings(&manifest, device);
    let first_instance = ResourceInstanceId::fresh().unwrap();
    let second_instance = ResourceInstanceId::fresh().unwrap();
    bindings[0].instance = first_instance;
    bindings[3].instance = first_instance;
    bindings[1].instance = second_instance;
    bindings[2].instance = second_instance;

    assert_eq!(
        manifest.validate(device, &bindings),
        Err(BindingError::AliasedResourcesRequireRecompile)
    );
}

#[test]
fn texture_aliases_retain_live_and_recompile_refusals() {
    let manifest = four_texture_manifest();
    let device = DeviceId::allocate().unwrap();

    let mut overlapping = texture_bindings(&manifest, device);
    overlapping[2].instance = overlapping[1].instance;
    assert_eq!(
        manifest.validate(device, &overlapping),
        Err(BindingError::AliasedLiveResources)
    );

    let mut disjoint = texture_bindings(&manifest, device);
    disjoint[3].instance = disjoint[0].instance;
    assert_eq!(
        manifest.validate(device, &disjoint),
        Err(BindingError::AliasedResourcesRequireRecompile)
    );
}

#[test]
fn imported_texture_requires_and_validates_declared_incoming_state() {
    let mut graph = GraphBuilder::default();
    let mut desc = texture_desc();
    desc.usage = TextureUsage::TEXTURE_BINDING;
    let incoming = TextureUses {
        stages: StageMask::FRAGMENT,
        accesses: AccessMask::SHADER_READ,
    };
    let texture = graph
        .import_texture_with_state(desc, incoming, ImageLayout::ShaderReadOnly)
        .unwrap();
    graph
        .add_pass("sample", |pass| {
            pass.read_texture_with_use(
                texture,
                TextureRange::new(TextureAspects::COLOR, 0..1, 0..1),
                incoming,
                ImageLayout::ShaderReadOnly,
                TextureUsage::TEXTURE_BINDING,
            )?;
            pass.side_effect();
            Ok(())
        })
        .unwrap();
    let manifest = GraphCompiler
        .compile(&graph)
        .unwrap()
        .texture_bindings()
        .clone();
    let device = DeviceId::allocate().unwrap();
    let valid = texture_bindings(&manifest, device);
    manifest.validate(device, &valid).unwrap();

    let mut wrong_state = valid;
    wrong_state[0].layout = ImageLayout::Common;
    assert_eq!(
        manifest.validate(device, &wrong_state),
        Err(BindingError::IncompatibleIncomingState)
    );
}
