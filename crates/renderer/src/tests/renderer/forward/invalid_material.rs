//! A valid shader product outside the consumer's one-vec4 uniform schema.
use super::*;

pub(super) fn assets() -> (ForwardResources, crate::ForwardAssets) {
    let (mut resources, mut assets) = assets_with_texture_payload(vec![0; 4]);
    let definition = MaterialDefinition::new(
        vec![MaterialFragment {
            name: "sampled".into(),
            inputs: Vec::new(),
            output: MaterialType::Vec4,
            calls: Vec::new(),
            wgsl: "fn sampled() -> vec4<f32> { return textureLoad(material_texture, vec2<i32>(0, 0), 0) * tint + extra; }".into(),
        }],
        "sampled",
        vec![
            MaterialParameter { name: "tint".into(), ty: MaterialType::Vec4, resource: "material_tint".into() },
            MaterialParameter { name: "extra".into(), ty: MaterialType::Vec4, resource: "extra_values".into() },
        ],
        vec![
            MaterialResource { name: "material_tint".into(), kind: MaterialResourceKind::UniformVec4, location: BindingLocation { group: 0, binding: 0 } },
            MaterialResource { name: "material_texture".into(), kind: MaterialResourceKind::Texture2dF32, location: BindingLocation { group: 0, binding: 1 } },
            MaterialResource { name: "extra_values".into(), kind: MaterialResourceKind::UniformVec4, location: BindingLocation { group: 0, binding: 2 } },
        ],
        BTreeMap::new(),
        SurfaceOutput::Color,
    );
    let store = AssetStore::<MaterialAsset, MaterialDefinition, ()>::new();
    let material = match store.acquire(&store.create().unwrap()).unwrap() {
        Acquire::Producer(p) => p.commit(definition, ResidentBytes::new(0)).unwrap(),
        _ => unreachable!(),
    };
    resources.material = SceneAssetRef::from_snapshot(&material);
    assets.materials = vec![material];
    (resources, assets)
}
