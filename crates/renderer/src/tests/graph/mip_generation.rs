use fluxel_rhi::api::{
    AccessMask, ImageLayout, StageMask, TextureAspects, TextureDesc, TextureDimension,
    TextureFormat, TextureUsage, TextureUses,
};

use super::*;
use crate::graph::{
    access::{AccessError, AccessPlan},
    binding::TextureSlot,
    definition::{AccessKind, GraphBuilder, GraphError, TextureAccessContract, TextureRange},
    state::StatePlan,
};

fn desc() -> TextureDesc {
    TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [4, 4, 1],
        array_layers: 1,
        format: TextureFormat::Rgba8Unorm,
        mip_levels: 2,
        samples: 1,
        usage: TextureUsage::COPY_DST | TextureUsage::COPY_SRC | TextureUsage::TEXTURE_BINDING,
        view_formats: Vec::new(),
    }
}

fn shader_read() -> TextureAccessContract {
    TextureAccessContract {
        uses: TextureUses {
            stages: StageMask::FRAGMENT,
            accesses: AccessMask::SHADER_READ,
        },
        layout: ImageLayout::ShaderReadOnly,
        descriptor_usage: TextureUsage::TEXTURE_BINDING,
    }
}

fn all_mips() -> TextureRange {
    TextureRange::new(TextureAspects::COLOR, 0..2, 0..1)
}

#[test]
fn cold_two_level_chain_defines_only_after_the_blit_and_plans_separate_states() {
    let mut graph = GraphBuilder::default();
    let texture = graph.create_texture(desc()).unwrap();
    let (upload, uploaded) = graph
        .add_pass("upload-base", |pass| {
            pass.write_texture_with_use(
                texture,
                one_color_mip(0),
                copy_write().uses,
                ImageLayout::CopyDst,
                TextureUsage::COPY_DST,
            )
        })
        .unwrap();

    assert_eq!(
        graph.add_pass("premature-sample", |pass| {
            pass.read_texture_with_use(
                uploaded,
                all_mips(),
                shader_read().uses,
                ImageLayout::ShaderReadOnly,
                TextureUsage::TEXTURE_BINDING,
            )
        }),
        Err(GraphError::UndefinedRead)
    );

    let (generated, generations) = declare_chain(&mut graph, uploaded, 7, 2).unwrap();
    assert_eq!(generations.len(), 1);
    assert_eq!(generations[0].texture, 7);
    assert_eq!(generations[0].source_mip, 0);
    assert_eq!(generations[0].destination_mip, 1);
    let (sample, ()) = graph
        .add_pass("sample-all", |pass| {
            pass.read_texture_with_use(
                generated,
                all_mips(),
                shader_read().uses,
                ImageLayout::ShaderReadOnly,
                TextureUsage::TEXTURE_BINDING,
            )?;
            pass.side_effect();
            Ok(())
        })
        .unwrap();

    let order = [upload, generations[0].pass, sample];
    let states = StatePlan::compile(&graph, &order).unwrap();
    let blit_transitions = &states.pass(generations[0].pass).unwrap().textures;
    assert!(blit_transitions.iter().any(|transition| {
        transition.range == one_color_mip(0)
            && transition.from == copy_write().uses
            && transition.to == copy_read().uses
            && transition.layout == (ImageLayout::CopyDst, ImageLayout::CopySrc)
    }));
    assert!(blit_transitions.iter().any(|transition| {
        transition.range == one_color_mip(1)
            && transition.from.accesses == AccessMask::EMPTY
            && transition.to == copy_write().uses
            && transition.layout == (ImageLayout::Undefined, ImageLayout::CopyDst)
    }));
    let sample_transitions = &states.pass(sample).unwrap().textures;
    assert_eq!(sample_transitions.len(), 2);
    assert!(sample_transitions.iter().all(|transition| {
        transition.to == shader_read().uses && transition.layout.1 == ImageLayout::ShaderReadOnly
    }));

    let access = AccessPlan::compile(&graph, &order).unwrap();
    let pass = access.pass(generations[0].pass).unwrap();
    assert!(
        pass.require_texture(
            TextureSlot::from(generated),
            one_color_mip(0),
            AccessKind::Read,
            copy_read(),
        )
        .is_ok()
    );
    assert!(
        pass.require_texture(
            TextureSlot::from(generated),
            one_color_mip(1),
            AccessKind::Write,
            copy_write(),
        )
        .is_ok()
    );
    assert_eq!(
        pass.require_texture(
            TextureSlot::from(generated),
            all_mips(),
            AccessKind::Read,
            copy_read(),
        ),
        Err(AccessError::TextureRangeOutsideDeclaration)
    );
}

#[test]
fn declaration_rejects_a_mismatched_level_count_before_adding_passes() {
    let mut graph = GraphBuilder::default();
    let texture = graph.create_texture(desc()).unwrap();
    let (_, uploaded) = graph
        .add_pass("upload-base", |pass| {
            pass.write_texture_with_use(
                texture,
                one_color_mip(0),
                copy_write().uses,
                ImageLayout::CopyDst,
                TextureUsage::COPY_DST,
            )
        })
        .unwrap();
    assert!(matches!(
        declare_chain(&mut graph, uploaded, 0, 1),
        Err(GraphError::InvalidRange)
    ));
}
