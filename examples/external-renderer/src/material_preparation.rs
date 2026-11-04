//! Independent Material, Shader, geometry, and opaque-batch preparation consumer.

use std::{collections::BTreeMap, fs, path::Path};

use fluxel_assets::{Acquire, AssetSnapshot, AssetStore, ResidentBytes};
use fluxel_material::{
    BindingLocation, DynamicValue, MaterialAsset, MaterialDefinition, MaterialFragment,
    MaterialInstance, MaterialParameter, MaterialResource, MaterialResourceKind, MaterialType,
    StaticValue, SurfaceOutput,
};
use fluxel_renderer::{
    BoundsVisibility, ForwardCpuPipeline, FramePipeline, GeometryAttribute, GeometryLayout,
    GeometryStream, MaterialPreparationError, RenderBounds, RenderChange, RenderChangeSet,
    RenderRecord, RenderScene, RenderTransform, RenderView, SceneAssetRef, prepare_material,
};
use fluxel_rhi::api::{VertexFormat, VertexStepMode};
use fluxel_shader::{
    EntryPoint, ScalarType, ShaderComposer, ShaderCompositionRequest, ShaderStage, SourceUnit,
    VertexInputDeclaration, VertexType,
};

use crate::opaque;

#[derive(Clone)]
struct SceneInput {
    material: SceneAssetRef<MaterialAsset>,
    layout: usize,
    binding: String,
}

const TEMPLATE: &str = r#"
struct VertexIn { @location(0) position: vec3<f32>, @location(1) uv: vec2<f32>, @location(2) color: vec4<f32>, };
struct VertexOut { @builtin(position) position: vec4<f32>, @location(0) color: vec4<f32>, };
@vertex fn forward_vertex(input: VertexIn) -> VertexOut {
    var output: VertexOut;
    output.position = vec4<f32>(input.position.xy + input.uv * 0.01, input.position.z, 1.0);
    output.color = input.color;
    return output;
}
@fragment fn forward_fragment(input: VertexOut) -> @location(0) vec4<f32> {
    return evaluate_material() * input.color;
}
"#;

/// Runs one real Material and Shader product against three physical layouts.
pub(crate) fn run() -> Result<(), Box<dyn std::error::Error>> {
    let evidence = Path::new("target/external-renderer-evidence/s5-preparation");
    fs::create_dir_all(evidence)?;
    fs::write(evidence.join("input.wgsl"), TEMPLATE)?;
    fs::write(
        evidence.join("expected.txt"),
        "layouts=3\nlogical_shader=shared\nbatches=3\n",
    )?;

    let store = AssetStore::<MaterialAsset, MaterialDefinition, ()>::new();
    let handle = store.create()?;
    let snapshot = publish(&store, &handle, false)?;
    let material = SceneAssetRef::from_snapshot(&snapshot);
    let layouts = [interleaved(), separate(), mixed()];
    let (scene, view, ids) = scene(&material)?;
    let scene = scene.snapshot();
    let pipeline = ForwardCpuPipeline::new(BoundsVisibility);
    let frame = pipeline.prepare(&scene, view, RenderBounds::new([-1.0; 3], [1.0; 3])?)?;
    let mut composer = ShaderComposer::new();
    let mut products = Vec::new();
    let mut batch_inputs = Vec::new();
    for &id in frame.candidates() {
        let input = &frame.record(id)?.resources;
        if input.material != material {
            return Err("prepared scene record resolved a foreign material generation".into());
        }
        let layout = layouts
            .get(input.layout)
            .ok_or("scene record selected an unavailable geometry layout")?;
        let product = prepare_material(&snapshot, layout, &mut composer, request())?;
        batch_inputs.push((
            (
                product.material().key().clone(),
                input.layout,
                input.binding.clone(),
            ),
            id,
        ));
        products.push(product);
    }
    if products.len() != 4 {
        return Err("visible scene did not prepare all three layout records".into());
    }
    let prepared_interleaved = &products[0];
    let prepared_separate = &products[2];
    let prepared_mixed = &products[3];
    let first = instance(
        &snapshot,
        prepared_interleaved.material(),
        [1.0, 0.0, 0.0, 1.0],
    )?;
    let second = instance(
        &snapshot,
        prepared_interleaved.material(),
        [0.0, 1.0, 0.0, 1.0],
    )?;
    let replacement = replacement(&store, &handle)?;
    let replaced = prepare_material(&replacement, &layouts[0], &mut composer, request())?;
    if SceneAssetRef::from_snapshot(&replacement) == material {
        return Err("material replacement did not advance the scene-visible generation".into());
    }
    let batches = opaque::group(batch_inputs);
    fs::write(
        evidence.join("actual.txt"),
        format!(
            "visible_candidates={}\nlayouts=3\nshader_shared={}\ndynamic_variant_shared={}\nreplacement_variant_changed={}\nbatches={}\n",
            frame.candidates().len(),
            prepared_interleaved.shader().cache_key() == prepared_separate.shader().cache_key()
                && prepared_separate.shader().cache_key() == prepared_mixed.shader().cache_key(),
            first.variant() == second.variant(),
            replaced.material().key() != prepared_interleaved.material().key(),
            batches.len(),
        ),
    )?;
    if prepared_interleaved.shader().cache_key() != prepared_separate.shader().cache_key()
        || prepared_separate.shader().cache_key() != prepared_mixed.shader().cache_key()
        || prepared_interleaved.material().key() != prepared_separate.material().key()
    {
        return Err("compatible layouts did not share material and logical Shader products".into());
    }
    if first.variant() != second.variant() {
        return Err("dynamic values changed the shared material variant".into());
    }
    if replaced.material().key() == prepared_interleaved.material().key() {
        return Err("static material replacement reused its prior variant".into());
    }
    let expected_batches = [vec![ids[0], ids[1]], vec![ids[2]], vec![ids[3]]];
    if batches.len() != expected_batches.len()
        || batches[0].1 != expected_batches[0]
        || batches[1].1 != expected_batches[1]
        || batches[2].1 != expected_batches[2]
    {
        return Err("opaque compatibility batches differ from fixed expected order".into());
    }
    let mut missing_mapping = request();
    missing_mapping.vertex_inputs.clear();
    if !matches!(
        prepare_material(&snapshot, &layouts[0], &mut composer, missing_mapping),
        Err(MaterialPreparationError::MissingVertexSemantic { location: 0 })
    ) {
        return Err("unmapped reflected vertex input was accepted".into());
    }
    println!(
        "S5 preparation: Scene candidates resolved three layouts and three opaque batches"
    );
    Ok(())
}

fn scene(
    material: &SceneAssetRef<MaterialAsset>,
) -> Result<
    (
        RenderScene<SceneInput>,
        fluxel_renderer::RenderViewId,
        Vec<fluxel_renderer::RenderId>,
    ),
    Box<dyn std::error::Error>,
> {
    let mut scene = RenderScene::new()?;
    let mut changes = RenderChangeSet::new();
    for (layout, binding) in [
        (0, "external-uniform-a"),
        (0, "external-uniform-a"),
        (1, "external-uniform-b"),
        (2, "external-uniform-c"),
    ] {
        changes.push(RenderChange::Insert(RenderRecord {
            resources: SceneInput {
                material: *material,
                layout,
                binding: binding.into(),
            },
            transform: RenderTransform::identity(),
            bounds: RenderBounds::new([0.0; 3], [1.0; 3])?,
        }));
    }
    changes.push(RenderChange::InsertView(RenderView::new(
        RenderTransform::identity(),
        [640, 360],
    )?));
    let receipt = scene.apply(&changes)?;
    Ok((
        scene,
        receipt.inserted_views()[0],
        receipt.inserted().to_vec(),
    ))
}

fn request() -> ShaderCompositionRequest {
    let mut request =
        ShaderCompositionRequest::new(vec![SourceUnit::new("forward-template", TEMPLATE)]);
    request.entry_points = vec![
        EntryPoint::new("forward_vertex", ShaderStage::Vertex),
        EntryPoint::new("forward_fragment", ShaderStage::Fragment),
    ];
    request.vertex_inputs = vec![
        VertexInputDeclaration::new("POSITION", 0, VertexType::Vec3(ScalarType::F32)),
        VertexInputDeclaration::new("UV", 1, VertexType::Vec2(ScalarType::F32)),
        VertexInputDeclaration::new("COLOR", 2, VertexType::Vec4(ScalarType::F32)),
    ];
    request
}

fn definition(replacement: bool) -> MaterialDefinition {
    MaterialDefinition::new(
        vec![MaterialFragment {
            name: "base_color".into(),
            inputs: vec![],
            output: MaterialType::Vec4,
            calls: vec![],
            wgsl: "fn base_color() -> vec4<f32> { return tint; }".into(),
        }],
        "base_color",
        vec![MaterialParameter {
            name: "tint".into(),
            ty: MaterialType::Vec4,
            resource: "material_tint".into(),
        }],
        vec![MaterialResource {
            name: "material_tint".into(),
            kind: MaterialResourceKind::UniformVec4,
            location: BindingLocation {
                group: 0,
                binding: 0,
            },
        }],
        BTreeMap::from([("replacement".into(), StaticValue::Bool(replacement))]),
        SurfaceOutput::Color,
    )
}

fn publish(
    store: &AssetStore<MaterialAsset, MaterialDefinition, ()>,
    handle: &fluxel_assets::AssetHandle<MaterialAsset>,
    replacement: bool,
) -> Result<AssetSnapshot<MaterialAsset, MaterialDefinition>, Box<dyn std::error::Error>> {
    match store.acquire(handle)? {
        Acquire::Producer(producer) => {
            Ok(producer.commit(definition(replacement), ResidentBytes::new(0))?)
        }
        _ => Err("new material handle did not issue a producer".into()),
    }
}

fn replacement(
    store: &AssetStore<MaterialAsset, MaterialDefinition, ()>,
    handle: &fluxel_assets::AssetHandle<MaterialAsset>,
) -> Result<AssetSnapshot<MaterialAsset, MaterialDefinition>, Box<dyn std::error::Error>> {
    match store.request_replacement(handle)? {
        fluxel_assets::Production::Producer(producer) => {
            Ok(producer.commit(definition(true), ResidentBytes::new(0))?)
        }
        _ => Err("live material did not issue a replacement producer".into()),
    }
}

fn instance(
    snapshot: &AssetSnapshot<MaterialAsset, MaterialDefinition>,
    variant: &fluxel_material::CompiledMaterialVariant,
    tint: [f32; 4],
) -> Result<MaterialInstance, fluxel_material::MaterialError> {
    MaterialInstance::new(
        fluxel_material::MaterialAssetRef::from_snapshot(snapshot),
        variant,
        BTreeMap::from([("tint".into(), DynamicValue::Vec4(tint))]),
        BTreeMap::from([("material_tint".into(), "external-uniform".into())]),
    )
}

fn stream(stride: u64, attributes: Vec<GeometryAttribute>) -> GeometryStream {
    GeometryStream {
        stride,
        step_mode: VertexStepMode::Vertex,
        attributes,
    }
}

fn attribute(semantic: &str, offset: u64, format: VertexFormat) -> GeometryAttribute {
    GeometryAttribute {
        semantic: semantic.into(),
        offset,
        format,
    }
}

fn interleaved() -> GeometryLayout {
    GeometryLayout {
        streams: vec![stream(
            36,
            vec![
                attribute("POSITION", 0, VertexFormat::Float32(3)),
                attribute("UV", 12, VertexFormat::Float32(2)),
                attribute("COLOR", 20, VertexFormat::Float32(4)),
            ],
        )],
    }
}
fn separate() -> GeometryLayout {
    GeometryLayout {
        streams: vec![
            stream(12, vec![attribute("POSITION", 0, VertexFormat::Float32(3))]),
            stream(8, vec![attribute("UV", 0, VertexFormat::Float32(2))]),
            stream(16, vec![attribute("COLOR", 0, VertexFormat::Float32(4))]),
        ],
    }
}
fn mixed() -> GeometryLayout {
    GeometryLayout {
        streams: vec![
            stream(
                20,
                vec![
                    attribute("POSITION", 0, VertexFormat::Float32(3)),
                    attribute("UV", 12, VertexFormat::Float32(2)),
                ],
            ),
            stream(16, vec![attribute("COLOR", 0, VertexFormat::Float32(4))]),
        ],
    }
}
