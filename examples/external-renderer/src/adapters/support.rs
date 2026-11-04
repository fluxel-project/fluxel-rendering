//! Asset, material, observation, and error-boundary helpers.

use super::*;
use fluxel_assets::{Acquire, AssetKind, AssetSnapshot, AssetStore, Production, ResidentBytes};
use fluxel_material::{
    BindingLocation, FragmentCall, MaterialAsset, MaterialCompiler, MaterialDefinition,
    MaterialFragment, MaterialParameter, MaterialResource, MaterialResourceKind, MaterialType,
    StaticValue, SurfaceOutput,
};
use fluxel_renderer::{
    CpuFeature, CpuPhase, FeatureError, RenderBounds, RenderTransform, SceneAssetRef, SceneDirty,
    SceneError,
};

pub(super) fn evidence_facts(actual: &ReferenceResult) -> EvidenceFacts {
    EvidenceFacts {
        initial_version: actual.initial.version,
        updated_version: actual.updated.version,
        retained_version: actual.retained.version,
        initial_material_generations: actual.initial.record_material_generations.clone(),
        updated_material_generations: actual.updated.record_material_generations.clone(),
        updated_extent: actual.updated.view_extents.clone(),
        updated_color: actual.features.updated_color,
        updated_label: actual.features.updated_label.clone(),
        feature_initial_version: actual.features.initial_version,
        feature_updated_version: actual.features.updated_version,
        version_mismatch: actual.features.version_mismatch.clone(),
        handle_mismatch: actual.features.handle_mismatch.clone(),
        cross_scene_mismatch: actual.features.cross_scene_mismatch.clone(),
        foreign_issuer_mismatch: actual.features.foreign_issuer_mismatch.clone(),
        cross_issuer_mismatch: actual.features.cross_issuer_mismatch.clone(),
        storage_issuer_mismatch: actual.features.storage_issuer_mismatch.clone(),
        consumed_contributions: actual.features.consumed_contributions,
        rejected_contribution: actual.features.rejected_contribution.clone(),
        storage_context_mismatch: actual.features.storage_context_mismatch.clone(),
        mixed_version_contribution: actual.features.mixed_version_contribution.clone(),
        replayed_contribution: actual.features.replayed_contribution.clone(),
    }
}

pub(super) fn feature_boundary_errors()
-> Result<(FeatureError, FeatureError), Box<dyn std::error::Error>> {
    let first = ExternalScene::new()?;
    let second = ExternalScene::new()?;
    let first_snapshot = first.snapshot();
    let second_snapshot = second.snapshot();
    let first_issuer = first.feature_issuer()?;
    let second_issuer = second.feature_issuer()?;
    let first_handle = first_issuer.issue("dense-colors")?;
    let second_handle = second_issuer.issue("dense-colors")?;
    let first_context = first_issuer.context(&first_snapshot, CpuPhase::Reference, first_handle)?;
    let mut first_storage = first_issuer.storage::<Vec<[u8; 4]>>();
    let input = first_storage.publish(&first_context, Vec::<[u8; 4]>::new())?;
    let feature = DenseColorFeature {
        handle: second_handle,
    };
    let cross_scene_mismatch = match feature.prepare(
        &second_issuer.context(&second_snapshot, CpuPhase::Reference, second_handle)?,
        &input,
    ) {
        Err(error) => error,
        Ok(_) => return Err("a matching version from another scene unexpectedly prepared".into()),
    };
    let foreign_issuer_mismatch =
        match second_issuer.context(&second_snapshot, CpuPhase::Reference, first_handle) {
            Err(error) => error,
            Ok(_) => return Err("a foreign issuer handle unexpectedly created context".into()),
        };
    Ok((cross_scene_mismatch, foreign_issuer_mismatch))
}

pub(super) fn observe(snapshot: &ExternalSnapshot) -> Result<SceneObservation, SceneError> {
    let record_material_slots = snapshot
        .record_ids()
        .map(|id| {
            snapshot
                .record(id)
                .map(|record| record.resources.material().id().slot())
        })
        .collect::<Result<_, _>>()?;
    let record_material_generations = snapshot
        .record_ids()
        .map(|id| {
            snapshot
                .record(id)
                .map(|record| record.resources.material().generation().get())
        })
        .collect::<Result<_, _>>()?;
    let view_extents = snapshot
        .view_ids()
        .map(|id| snapshot.view(id).map(|view| view.extent))
        .collect::<Result<_, _>>()?;
    let SceneDirty { records, views } = snapshot.dirty().clone();
    Ok(SceneObservation {
        version: snapshot.version(),
        record_material_slots,
        record_material_generations,
        view_extents,
        record_dirty: records,
        view_dirty: views,
    })
}

pub(super) fn sample_record(resources: ExternalResources) -> Result<ExternalRecord, SceneError> {
    Ok(ExternalRecord {
        resources,
        transform: RenderTransform::identity(),
        bounds: unit_bounds()?,
    })
}

pub(super) fn unit_bounds() -> Result<RenderBounds, SceneError> {
    RenderBounds::new([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
}

#[derive(Clone, Copy)]
pub(super) struct AssetReferences {
    pub(super) triangle_geometry: SceneAssetRef<TriangleGeometryAsset>,
    pub(super) quad_geometry: SceneAssetRef<QuadGeometryAsset>,
    pub(super) material_initial: MaterialReference,
    pub(super) material_replacement: MaterialReference,
}

pub(super) fn asset_references() -> Result<AssetReferences, Box<dyn std::error::Error>> {
    let triangle_store = AssetStore::<TriangleGeometryAsset, Vec<u8>, String>::new();
    let triangle_handle = triangle_store.create()?;
    let triangle = initial(&triangle_store, &triangle_handle, vec![1, 2, 3], 3)?;
    let quad_store = AssetStore::<QuadGeometryAsset, Vec<u8>, String>::new();
    let quad_handle = quad_store.create()?;
    let quad = initial(&quad_store, &quad_handle, vec![4, 5, 6, 7], 4)?;

    let material_store = AssetStore::<MaterialAsset, MaterialDefinition, String>::new();
    let material_handle = material_store.create()?;
    let material = initial(
        &material_store,
        &material_handle,
        material_definition(true),
        1,
    )?;
    let replacement = replace(
        &material_store,
        &material_handle,
        material_definition(false),
        1,
    )?;
    if material.id() != replacement.id() || material.generation() >= replacement.generation() {
        return Err("material replacement must retain identity and advance generation".into());
    }
    let initial_variant = MaterialCompiler::compile_for(&material)?;
    let replacement_variant = MaterialCompiler::compile_for(&replacement)?;
    if initial_variant.material()
        != Some(fluxel_material::MaterialAssetRef::from_snapshot(&material))
        || replacement_variant.material()
            != Some(fluxel_material::MaterialAssetRef::from_snapshot(
                &replacement,
            ))
        || initial_variant.key() == replacement_variant.key()
    {
        return Err("S3 material compilation did not preserve exact asset generations".into());
    }
    Ok(AssetReferences {
        triangle_geometry: SceneAssetRef::from_snapshot(&triangle),
        quad_geometry: SceneAssetRef::from_snapshot(&quad),
        material_initial: SceneAssetRef::from_snapshot(&material),
        material_replacement: SceneAssetRef::from_snapshot(&replacement),
    })
}

pub(super) fn initial<K: AssetKind, V: Send + Sync + 'static>(
    store: &AssetStore<K, V, String>,
    handle: &fluxel_assets::AssetHandle<K>,
    value: V,
    bytes: u64,
) -> Result<AssetSnapshot<K, V>, Box<dyn std::error::Error>> {
    match store.acquire(handle)? {
        Acquire::Producer(permit) => Ok(permit.commit(value, ResidentBytes::new(bytes))?),
        Acquire::Ready(_) | Acquire::Waiter(_) | Acquire::Failed(_) => {
            Err("new asset unexpectedly did not nominate a producer".into())
        }
    }
}

pub(super) fn replace<K: AssetKind, V: Send + Sync + 'static>(
    store: &AssetStore<K, V, String>,
    handle: &fluxel_assets::AssetHandle<K>,
    value: V,
    bytes: u64,
) -> Result<AssetSnapshot<K, V>, Box<dyn std::error::Error>> {
    match store.request_replacement(handle)? {
        Production::Producer(permit) => Ok(permit.commit(value, ResidentBytes::new(bytes))?),
        Production::Waiter(_) => Err("asset replacement unexpectedly waited for production".into()),
    }
}

pub(super) fn material_definition(enabled: bool) -> MaterialDefinition {
    MaterialDefinition::new(
        vec![
            MaterialFragment {
                name: String::from("sample_color"),
                inputs: Vec::new(),
                output: MaterialType::Vec4,
                calls: Vec::new(),
                wgsl: String::from(
                    "pub(super) fn sample_color() -> vec4<f32> { return textureLoad(material_texture, vec2<i32>(0, 0), 0); }",
                ),
            },
            MaterialFragment {
                name: String::from("custom_tint"),
                inputs: Vec::new(),
                output: MaterialType::Vec4,
                calls: vec![FragmentCall {
                    fragment: String::from("sample_color"),
                    arguments: Vec::new(),
                }],
                wgsl: String::from(
                    "pub(super) fn custom_tint() -> vec4<f32> { return sample_color() * tint; }",
                ),
            },
        ],
        "custom_tint",
        vec![MaterialParameter {
            name: String::from("tint"),
            ty: MaterialType::Vec4,
            resource: String::from("material_tint"),
        }],
        vec![
            MaterialResource {
                name: String::from("material_tint"),
                kind: MaterialResourceKind::UniformVec4,
                location: BindingLocation {
                    group: 0,
                    binding: 0,
                },
            },
            MaterialResource {
                name: String::from("material_texture"),
                kind: MaterialResourceKind::Texture2dF32,
                location: BindingLocation {
                    group: 0,
                    binding: 1,
                },
            },
        ],
        std::collections::BTreeMap::from([(
            String::from("MATERIAL_ENABLED"),
            StaticValue::Bool(enabled),
        )]),
        SurfaceOutput::Color,
    )
}
