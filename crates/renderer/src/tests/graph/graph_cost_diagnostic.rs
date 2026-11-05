//! Ignored CPU/API diagnostic for measured S8 Graph validation cost.
//! No device, encoder, or GPU work is created; this is not a release gate.

use std::{hint::black_box, time::Instant};

use fluxel_rhi::api::{
    AccessMask, BufferDesc, BufferUsage, BufferUses, DeviceId, ImageLayout, MemoryClass, StageMask,
    TextureAspects, TextureDesc, TextureDimension, TextureFormat, TextureUsage, TextureUses,
};

use super::{
    access::AccessPlan,
    binding::{
        BindingError, BufferSlot, ResourceInstanceId, TextureBinding, TextureBindingManifest,
    },
    compiler::GraphCompiler,
    definition::{AccessKind, BufferAccessContract, GraphBuilder, TextureRange},
};

const MANIFEST_SIZES: &[usize] = &[100, 500, 1_000];
const ACCESS_SIZES: &[usize] = &[100, 1_000, 10_000];

fn texture_desc() -> TextureDesc {
    TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [1, 1, 1],
        array_layers: 1,
        format: TextureFormat::Rgba8Unorm,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::TEXTURE_BINDING,
        view_formats: Vec::new(),
    }
}

fn texture_uses() -> TextureUses {
    TextureUses {
        stages: StageMask::FRAGMENT,
        accesses: AccessMask::SHADER_READ,
    }
}

fn buffer_uses() -> BufferUses {
    BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    }
}

fn buffer_contract() -> BufferAccessContract {
    BufferAccessContract {
        uses: buffer_uses(),
        descriptor_usage: BufferUsage::COPY_SRC,
    }
}

/// Produces actual compiler-derived requirements, rather than constructing a
/// manifest directly.  Setup is deliberately outside the measured section.
fn texture_manifest(count: usize) -> TextureBindingManifest {
    let mut graph = GraphBuilder::default();
    let textures = (0..count)
        .map(|_| {
            graph
                .import_texture_with_state(
                    texture_desc(),
                    texture_uses(),
                    ImageLayout::ShaderReadOnly,
                )
                .expect("fixture texture import")
        })
        .collect::<Vec<_>>();
    graph
        .add_pass("cost-manifest", |pass| {
            for texture in &textures {
                pass.read_texture_with_use(
                    *texture,
                    TextureRange::new(TextureAspects::COLOR, 0..1, 0..1),
                    texture_uses(),
                    ImageLayout::ShaderReadOnly,
                    TextureUsage::TEXTURE_BINDING,
                )?;
            }
            pass.side_effect();
            Ok(())
        })
        .expect("fixture pass");
    GraphCompiler
        .compile(&graph)
        .expect("fixture compile")
        .texture_bindings()
        .clone()
}

fn texture_bindings(manifest: &TextureBindingManifest, device: DeviceId) -> Vec<TextureBinding> {
    manifest
        .requirements()
        .iter()
        .map(|requirement| {
            let (initial, layout) = requirement.initial.expect("fixture incoming texture state");
            TextureBinding {
                slot: requirement.slot,
                instance: ResourceInstanceId::fresh().expect("fixture resource identity"),
                device,
                desc: requirement.desc.clone(),
                initial,
                layout,
            }
        })
        .collect()
}

fn access_fixture(draws: usize) -> (AccessPlan, super::definition::GraphPass, BufferSlot) {
    let mut graph = GraphBuilder::default();
    let size = u64::try_from(draws)
        .expect("fixture draw count")
        .checked_mul(16)
        .expect("fixture byte count");
    let buffer = graph
        .import_buffer_desc(
            BufferDesc {
                label: None,
                size,
                usage: BufferUsage::COPY_SRC,
                memory: MemoryClass::Device,
            },
            buffer_uses(),
        )
        .expect("fixture buffer import");
    let (pass, ()) = graph
        .add_pass("cost-access", |pass| {
            for draw in 0..draws {
                let start = u64::try_from(draw).expect("fixture offset") * 16;
                pass.read_with_use(
                    buffer,
                    start..start + 16,
                    buffer_uses(),
                    BufferUsage::COPY_SRC,
                )?;
            }
            pass.side_effect();
            Ok(())
        })
        .expect("fixture pass");
    let plan = AccessPlan::compile(&graph, &[pass]).expect("fixture access plan");
    (plan, pass, BufferSlot::from(buffer))
}

fn repetitions_for(size: usize) -> usize {
    // Keep the smallest case above timer noise while avoiding a large benchmark.
    match size {
        0..=100 => 25,
        101..=1_000 => 3,
        _ => 1,
    }
}

fn print_sample(name: &str, size: usize, calls: usize, elapsed_ns: u128) {
    println!(
        "GRAPH_COST_{name}_N={size} CALLS={calls} ELAPSED_NS={elapsed_ns} NS_PER_CALL={}",
        elapsed_ns / u128::try_from(calls).expect("nonzero call count"),
    );
}

#[test]
#[ignore = "manual CPU/API diagnostic; not release or performance acceptance"]
fn graph_cost_diagnostic() {
    for &count in MANIFEST_SIZES {
        let manifest = texture_manifest(count);
        let device = DeviceId::allocate().expect("fixture device identity");
        let bindings = texture_bindings(&manifest, device);
        assert_eq!(bindings.len(), count);
        manifest
            .validate(device, &bindings)
            .expect("non-aliased bindings must validate");

        let repetitions = repetitions_for(count);
        let started = Instant::now();
        for _ in 0..repetitions {
            black_box(manifest.validate(device, black_box(&bindings)))
                .expect("non-aliased bindings must validate");
        }
        print_sample(
            "MANIFEST_VALIDATE",
            count,
            repetitions,
            started.elapsed().as_nanos(),
        );

        let mut aliased = bindings.clone();
        aliased[1].instance = aliased[0].instance;
        assert_eq!(
            manifest.validate(device, &aliased),
            Err(BindingError::AliasedLiveResources),
            "aliased live resources must retain their existing refusal",
        );
    }
    println!("GRAPH_COST_MANIFEST_ALIAS_NEGATIVE=AliasedLiveResources");

    for &draws in ACCESS_SIZES {
        let (plan, pass, slot) = access_fixture(draws);
        let access = plan.pass(pass).expect("fixture pass access");
        let repetitions = repetitions_for(draws);
        let started = Instant::now();
        for _ in 0..repetitions {
            for draw in 0..draws {
                let start = u64::try_from(draw).expect("fixture offset") * 16;
                black_box(access.require_buffer(
                    slot,
                    start..start + 16,
                    AccessKind::Read,
                    buffer_contract(),
                ))
                .expect("declared range must validate");
            }
        }
        print_sample(
            "PASS_ACCESS_REQUIRE_BUFFER",
            draws,
            draws * repetitions,
            started.elapsed().as_nanos(),
        );
    }
    println!("GRAPH_COST_DIAGNOSTIC_KIND=CPU_API_VALIDATE_ONLY");
    println!("GRAPH_COST_DIAGNOSTIC_NOT_RELEASE_ACCEPTANCE=true");
}
