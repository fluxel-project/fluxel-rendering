use super::*;

#[test]
fn both_records_compile_the_actual_four_image_material_and_reorder_named_sources() {
    let input = input(Case::Alias).unwrap();
    let prepared = ForwardCpuPipeline::new(BoundsVisibility)
        .prepare(
            &input.snapshot,
            input.view,
            RenderBounds::new([-5.; 3], [5.; 3]).unwrap(),
        )
        .unwrap();
    let draws = prepare_forward(
        &prepared,
        &input.assets,
        &mut fluxel_shader::ShaderComposer::new(),
    )
    .unwrap();
    assert_eq!(draws.len(), 2);
    for draw in draws {
        let bindings = draw.prepared().shader().interface().bindings();
        assert_eq!(
            bindings
                .iter()
                .filter(|b| b.group == 0 && b.kind == fluxel_shader::BindingKind::Texture)
                .count(),
            4
        );
        assert!(
            bindings
                .iter()
                .filter(|b| b.group == 0)
                .all(|b| b.read && !b.write)
        );
    }
}

#[test]
fn controls_edit_only_the_declared_image_channel() {
    let baseline = input(Case::Baseline).unwrap();
    for (case, index, channel) in [
        (Case::Base, 0, 0),
        (Case::Normal, 2, 0),
        (Case::Ao, 3, 0),
        (Case::Roughness, 3, 1),
        (Case::Metallic, 3, 2),
        (Case::Emissive, 5, 1),
    ] {
        let changed = input(case).unwrap();
        assert_eq!(changed.geometry_bytes, baseline.geometry_bytes);
        assert_eq!(changed.sources, baseline.sources);
        let mut differences = 0;
        for (i, (old, new)) in baseline
            .assets
            .textures
            .iter()
            .zip(&changed.assets.textures)
            .enumerate()
        {
            assert_eq!(old.value().extent, new.value().extent);
            assert_eq!(old.value().encoding, new.value().encoding);
            for (at, (&a, &b)) in old.value().rgba8.iter().zip(&new.value().rgba8).enumerate() {
                if a != b {
                    assert_eq!((i, at % 4), (index, channel));
                    differences += 1;
                }
            }
        }
        assert_eq!(differences, 4, "exactly one channel in each of four texels");
    }
}

#[test]
fn alias_uses_one_immutable_source_for_two_logical_bindings() {
    let data = input(Case::Alias).unwrap();
    assert_eq!(data.sources[1][0], data.sources[1][3]);
    assert_eq!(data.sources[0][1], data.sources[1][1]);
    assert_eq!(data.assets.textures.len(), 6);
    assert_ne!(data.models[0], data.models[1]);
}

#[test]
fn actual_wrong_role_encoding_is_refused_by_the_consumer_input_contract() {
    let mut data = input(Case::Baseline).unwrap();
    let store = AssetStore::<ColorTextureAsset, ColorTextureData, String>::new();
    let original = data.assets.textures[2].value();
    let replacement = ColorTextureData {
        extent: original.extent,
        rgba8: original.rgba8.clone(),
        encoding: TextureEncoding::Srgb,
        mip_levels: 1,
    };
    data.assets.textures[2] = commit(&store, replacement, 16).unwrap();
    assert!(validate_roles(&data.assets.textures, data.sources).is_err());
}
