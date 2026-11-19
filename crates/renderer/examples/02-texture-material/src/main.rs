use std::{
    collections::BTreeMap,
    error::Error,
    fs,
    path::{Path, PathBuf},
};

#[cfg(all(windows, any(feature = "dx12", feature = "native-gl-wgl")))]
#[path = "../../rendering-integration/src/diagnostics.rs"]
mod diagnostics;
mod evidence;
#[cfg(all(windows, any(feature = "dx12", feature = "native-gl-wgl")))]
#[allow(dead_code)]
#[path = "../../rendering-integration/src/capture.rs"]
mod integration_capture;
#[cfg(all(windows, any(feature = "dx12", feature = "native-gl-wgl")))]
mod native;
mod sample_oracle;
mod srgb_filter;
mod support;
#[cfg(all(windows, any(feature = "dx12", feature = "native-gl-wgl")))]
mod window_oracle;
use evidence::write_source_identity;
use fluxel_assets::{Acquire, AssetSnapshot, AssetStore, ResidentBytes};
use fluxel_material::{
    BindingLocation, FragmentInput, MaterialAsset, MaterialCodeNode, MaterialDefinition,
    MaterialNode, MaterialNodeGraph, MaterialNodeId, MaterialNodeInput, MaterialNodeInputSource,
    MaterialParameter, MaterialResource, MaterialResourceKind, MaterialType, SurfaceOutput,
};
use fluxel_renderer::{
    BoundsVisibility, ColorTextureAsset, ColorTextureData, DeferredCpuPipeline, ForwardAssets,
    ForwardCpuPipeline, ForwardResources, GeometryAsset, GeometryAttribute, GeometryData,
    GeometryLayout, GeometryStream, RenderBounds, RenderChange, RenderChangeSet, RenderRecord,
    RenderScene, RenderTransform, RenderView, Renderer, SceneAssetRef, TextureEncoding,
};
use fluxel_rhi::{
    api::{Device, VertexFormat, VertexStepMode},
    backend::{
        ExecutionBackend, MappingBackend, PipelineBackend, RasterBackend, ReadbackBackend,
        ResourceBackend, ShaderBackend,
    },
};
use fluxel_shader::SourceUnit;
use sample_oracle::{expected_probes, verify_probes, write_expected_probes, Probe};
use support::{block_on, write_probes};
const EXTENT: [u32; 2] = [640, 360];
const KTX_SHA256: &str = "38495B089915D38E2AE6CE14CDF872FF7D2DF6F79F3A1E5A7A7851CBE65882BD";
const LOCKED_RHI_REVISION: &str = "8400f8f35933a5d0464014a8671799890f67ff49";
const REFERENCE_RHI_REVISION: &str = "ebdcb614140296197fcf9abe2924ab0faf5cc490";

#[derive(Clone, Copy)]
enum Mode {
    Offscreen,
    Visible,
}

struct Options {
    mode: Mode,
    exit_after_capture: bool,
    asset: Option<PathBuf>,
    srgb_filter: bool,
    permuted_bindings: bool,
    output: PathBuf,
}
#[derive(Clone, Copy)]
struct SceneOptions {
    srgb_filter: bool,
    permuted_bindings: bool,
}

impl Options {
    fn scene_options(&self) -> SceneOptions {
        SceneOptions {
            srgb_filter: self.srgb_filter,
            permuted_bindings: self.permuted_bindings,
        }
    }
}
fn main() -> Result<(), Box<dyn Error>> {
    let options = parse_options()?;
    let texture = if options.srgb_filter {
        srgb_filter::texture()
    } else {
        load_texture(
            options
                .asset
                .as_deref()
                .ok_or("--asset is required without --srgb-filter")?,
        )?
    };
    fs::create_dir_all(&options.output)?;
    write_source_identity(&options, &texture)?;
    #[cfg(all(windows, feature = "dx12"))]
    return native::run_dx12(options, texture);
    #[cfg(all(windows, not(feature = "dx12"), feature = "native-gl-wgl"))]
    return native::run_wgl(options, texture);
    #[cfg(not(all(windows, any(feature = "dx12", feature = "native-gl-wgl"))))]
    Err("select --features dx12 or native-gl-wgl on Windows".into())
}

fn parse_options() -> Result<Options, Box<dyn Error>> {
    let mut mode = None;
    let mut exit_after_capture = false;
    let mut srgb_filter = false;
    let mut noasset = false;
    let mut permuted_bindings = false;
    let mut asset = None;
    let mut output = PathBuf::from("target/renderer-02-texture-material");
    let mut args = std::env::args_os().skip(1);
    while let Some(arg) = args.next() {
        match arg.to_string_lossy().as_ref() {
            "--offscreen" => mode = Some(Mode::Offscreen),
            "--visible" => mode = Some(Mode::Visible),
            "--exit-after-capture" => exit_after_capture = true,
            "--asset" => asset = Some(args.next().ok_or("--asset requires a path")?.into()),
            "--srgb-filter" => srgb_filter = true,
            "--noasset" => noasset = true,
            "--permuted-bindings" => permuted_bindings = true,
            "--output" => output = args.next().ok_or("--output requires a directory")?.into(),
            _ => return Err(format!("unsupported argument: {}", arg.to_string_lossy()).into()),
        }
    }
    if noasset && !srgb_filter {
        return Err("--noasset is only valid with --srgb-filter".into());
    }
    Ok(Options {
        mode: mode.ok_or("select exactly one of --offscreen or --visible")?,
        exit_after_capture,
        asset,
        srgb_filter,
        permuted_bindings,
        output,
    })
}

struct TextureInput {
    extent: [u32; 2],
    rgba8: Vec<u8>,
    sha256: String,
    encoding: TextureEncoding,
    source: &'static str,
}

fn load_texture(path: &Path) -> Result<TextureInput, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let hash = sha256(&bytes);
    if !hash.eq_ignore_ascii_case(KTX_SHA256) {
        return Err(
            format!("source KTX SHA-256 mismatch: expected {KTX_SHA256}, got {hash}").into(),
        );
    }
    if bytes.len() < 68
        || &bytes[..12] != b"\xABKTX 11\xBB\r\n\x1A\n"
        || &bytes[12..16] != [1, 2, 3, 4]
    {
        return Err("expected little-endian KTX1 input".into());
    }
    let word = |at| u32::from_le_bytes(bytes[at..at + 4].try_into().expect("validated KTX header"));
    let (width, height) = (word(36), word(40));
    if width == 0 || height == 0 || word(24) != 0x1908 || word(28) != 0x8058 {
        return Err("expected RGBA8 KTX1 input".into());
    }
    let offset = 64usize
        .checked_add(word(60) as usize)
        .ok_or("KTX key/value offset overflow")?;
    let size = width
        .checked_mul(height)
        .and_then(|x| x.checked_mul(4))
        .ok_or("KTX extent overflow")? as usize;
    let image_size = offset
        .checked_add(4)
        .and_then(|x| x.checked_add(size))
        .ok_or("KTX image range overflow")?;
    if image_size > bytes.len() || word(offset) as usize != size {
        return Err("KTX base-level RGBA byte range is invalid".into());
    }
    Ok(TextureInput {
        extent: [width, height],
        rgba8: bytes[offset + 4..image_size].to_vec(),
        sha256: hash,
        encoding: TextureEncoding::Linear,
        source: "controlled-KTX-RGBA8",
    })
}

fn material(permuted_bindings: bool) -> MaterialDefinition {
    let locations = if permuted_bindings {
        [2, 0, 1]
    } else {
        [0, 1, 2]
    };
    MaterialDefinition::new(Vec::new(), "unused", vec![MaterialParameter { name: "tint".into(), ty: MaterialType::Vec4, resource: "material_tint".into() }], vec![
        MaterialResource { name: "material_tint".into(), kind: MaterialResourceKind::UniformVec4, location: BindingLocation { group: 0, binding: locations[0] } },
        MaterialResource { name: "material_texture".into(), kind: MaterialResourceKind::Texture2dF32, location: BindingLocation { group: 0, binding: locations[1] } },
        MaterialResource { name: "material_sampler".into(), kind: MaterialResourceKind::Sampler { comparison: false }, location: BindingLocation { group: 0, binding: locations[2] } },
    ], BTreeMap::new(), SurfaceOutput::ColorUv0)
    .with_code_node(MaterialCodeNode { name: "sample_metalplate".into(), inputs: vec![FragmentInput { name: "uv".into(), ty: MaterialType::Vec2 }], output: MaterialType::Vec4, calls: Vec::new(), source: SourceUnit::new("renderer-02-texture-sample", "fn sample_metalplate(uv: vec2<f32>) -> vec4<f32> { return textureSample(material_texture, material_sampler, uv) * tint; }"), resources: Vec::new() })
    .with_node_graph(MaterialNodeGraph { nodes: vec![MaterialNode { id: MaterialNodeId(2), callable: "sample_metalplate".into(), inputs: vec![MaterialNodeInput { input: "uv".into(), source: MaterialNodeInputSource::Uv0 }] }], output: MaterialNodeId(2) })
}

fn geometry(srgb_filter: bool) -> GeometryData {
    // The synthetic path shifts U by half a framebuffer sample so pixel
    // (319, 0) has U=0.5 and clamps to texture row zero: an actual black/white
    // half-texel sample. The controlled KTX path retains its source UVs.
    let u0 = if srgb_filter {
        0.5 / EXTENT[0] as f32
    } else {
        0.0
    };
    let u1 = 1.0 + u0;
    let source = [
        ([-1.0_f32, -1.0, 0.0], [u0, 1.0]),
        ([1.0, -1.0, 0.0], [u1, 1.0]),
        ([-1.0, 1.0, 0.0], [u0, 0.0]),
        ([-1.0, 1.0, 0.0], [u0, 0.0]),
        ([1.0, -1.0, 0.0], [u1, 1.0]),
        ([1.0, 1.0, 0.0], [u1, 0.0]),
    ];
    let mut bytes = Vec::with_capacity(192);
    for (position, uv) in source {
        for value in position.into_iter().chain([0., 0., 1.]).chain(uv) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    GeometryData {
        layout: GeometryLayout {
            streams: vec![GeometryStream {
                stride: 32,
                step_mode: VertexStepMode::Vertex,
                attributes: vec![
                    GeometryAttribute {
                        semantic: "POSITION".into(),
                        offset: 0,
                        format: VertexFormat::Float32(3),
                    },
                    GeometryAttribute {
                        semantic: "NORMAL".into(),
                        offset: 12,
                        format: VertexFormat::Float32(3),
                    },
                    GeometryAttribute {
                        semantic: "UV0".into(),
                        offset: 24,
                        format: VertexFormat::Float32(2),
                    },
                ],
            }],
        },
        streams: vec![bytes],
        vertices: 0..6,
    }
}

struct Scene {
    snapshot: fluxel_renderer::FrameSnapshot<ForwardResources>,
    view: fluxel_renderer::RenderViewId,
    assets: ForwardAssets,
}
fn scene(texture: &TextureInput, options: SceneOptions) -> Result<Scene, Box<dyn Error>> {
    let geometries = AssetStore::<GeometryAsset, GeometryData, String>::new();
    let materials = AssetStore::<MaterialAsset, MaterialDefinition, String>::new();
    let textures = AssetStore::<ColorTextureAsset, ColorTextureData, String>::new();
    let geometry = commit(&geometries, geometry(options.srgb_filter), 192)?;
    let material = commit(&materials, material(options.permuted_bindings), 0)?;
    let image = commit(
        &textures,
        ColorTextureData {
            extent: texture.extent,
            rgba8: texture.rgba8.clone(),
            encoding: texture.encoding,
            mip_levels: 1,
        },
        texture.rgba8.len() as u64,
    )?;
    let mut scene = RenderScene::new()?;
    let mut changes = RenderChangeSet::new();
    changes.push(RenderChange::Insert(RenderRecord {
        resources: ForwardResources {
            geometry: SceneAssetRef::from_snapshot(&geometry),
            material: SceneAssetRef::from_snapshot(&material),
            textures: vec![fluxel_renderer::ForwardTextureBinding::asset(
                "material_texture",
                SceneAssetRef::from_snapshot(&image),
            )],
            samplers: Vec::new(),
            tint: [1.; 4],
        },
        transform: RenderTransform::identity(),
        bounds: RenderBounds::new([-1.; 3], [1.; 3])?,
    }));
    changes.push(RenderChange::InsertView(RenderView::new(
        RenderTransform::identity(),
        EXTENT,
    )?));
    let receipt = scene.apply(&changes)?;
    Ok(Scene {
        snapshot: scene.snapshot(),
        view: receipt.inserted_views()[0],
        assets: ForwardAssets {
            geometry: vec![geometry],
            materials: vec![material],
            textures: vec![image],
        },
    })
}

fn commit<K, T>(
    store: &AssetStore<K, T, String>,
    value: T,
    bytes: u64,
) -> Result<AssetSnapshot<K, T>, Box<dyn Error>>
where
    K: fluxel_assets::AssetKind,
    T: Send + Sync + 'static,
{
    let handle = store.create()?;
    match store.acquire(&handle)? {
        Acquire::Producer(producer) => Ok(producer.commit(value, ResidentBytes::new(bytes))?),
        _ => Err("new source asset did not yield a producer".into()),
    }
}

fn run_offscreen<B>(
    device: Device<B>,
    texture: &TextureInput,
    scene_options: SceneOptions,
    output: &Path,
) -> Result<(), Box<dyn Error>>
where
    B: ExecutionBackend
        + ResourceBackend
        + MappingBackend
        + ReadbackBackend
        + RasterBackend
        + PipelineBackend
        + ShaderBackend,
{
    let scene = scene(texture, scene_options)?;
    let bounds = RenderBounds::new([-1.; 3], [1.; 3])?;
    let mut renderer = Renderer::new(device)?;
    let pipeline = ForwardCpuPipeline::new(BoundsVisibility);
    let expected = expected_probes(texture, 1.0);
    write_expected_probes(output, "forward", texture, &expected, 1)?;
    let submission = block_on(renderer.submit_forward(
        &pipeline,
        scene.snapshot.clone(),
        scene.view,
        bounds,
        &scene.assets,
    ))?;
    let rgba = block_on(renderer.readback(&submission))?;
    if renderer.pending() != 0 {
        return Err("offscreen readback did not retire the submitted owner".into());
    }
    fs::write(output.join("forward-readback.rgba"), &rgba)?;
    write_probes(output, "forward", &rgba, EXTENT)?;
    verify_probes(&rgba, &expected, 1)?;
    let deferred_expected = expected_probes(texture, 1.0);
    // This flat unlit scene preserves the already quantized G-buffer color.
    // Its final one-to-one composite must not add half-texel filtering error.
    write_expected_probes(output, "deferred", texture, &deferred_expected, 1)?;
    let deferred = block_on(renderer.submit_deferred(
        &DeferredCpuPipeline::new(BoundsVisibility),
        scene.snapshot,
        scene.view,
        bounds,
        &scene.assets,
    ))?;
    let deferred_rgba = block_on(renderer.readback(&deferred))?;
    if renderer.pending() != 0 {
        return Err("Deferred readback did not retire the submitted owner".into());
    }
    fs::write(output.join("deferred-readback.rgba"), &deferred_rgba)?;
    write_probes(output, "deferred", &deferred_rgba, EXTENT)?;
    verify_probes(&deferred_rgba, &deferred_expected, 1)?;
    fs::write(
        output.join("result.json"),
        "{\"schemaVersion\":1,\"result\":\"Renderer MaterialNodeGraph/CodeNode Forward and Deferred readbacks completed\",\"scope\":\"opaque textured quad; no PBR, resolve, or window claim\"}\n",
    )?;
    support::write_output_hashes(output)?;
    Ok(())
}

fn is_srgb(texture: &TextureInput) -> bool {
    matches!(texture.encoding, TextureEncoding::Srgb)
}

fn encoding_name(encoding: TextureEncoding) -> &'static str {
    match encoding {
        TextureEncoding::Linear => "Linear",
        TextureEncoding::Srgb => "Srgb",
        TextureEncoding::SrgbEncodedUnorm => "SrgbEncodedUnorm",
    }
}

pub(crate) fn sha256(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut state = [0u32; 8];
    state.copy_from_slice(&[
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ]);
    let bit_len = (bytes.len() as u64).wrapping_mul(8);
    let mut input = bytes.to_vec();
    input.push(0x80);
    while (input.len() + 8) % 64 != 0 {
        input.push(0);
    }
    input.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in input.chunks_exact(64) {
        let mut words = [0u32; 64];
        for (i, word) in words[..16].iter_mut().enumerate() {
            *word = u32::from_be_bytes(chunk[i * 4..i * 4 + 4].try_into().unwrap());
        }
        for i in 16..64 {
            words[i] = words[i - 16]
                .wrapping_add(
                    words[i - 15].rotate_right(7)
                        ^ words[i - 15].rotate_right(18)
                        ^ (words[i - 15] >> 3),
                )
                .wrapping_add(words[i - 7])
                .wrapping_add(
                    words[i - 2].rotate_right(17)
                        ^ words[i - 2].rotate_right(19)
                        ^ (words[i - 2] >> 10),
                );
        }
        let mut v = state;
        for (i, word) in words.into_iter().enumerate() {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7]
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(
                    [
                        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
                        0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
                        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
                        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
                        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
                        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
                        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
                        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
                        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
                        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
                        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
                    ][i],
                )
                .wrapping_add(word);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [
                t1.wrapping_add(t2),
                v[0],
                v[1],
                v[2],
                v[3].wrapping_add(t1),
                v[4],
                v[5],
                v[6],
            ];
        }
        for i in 0..8 {
            state[i] = state[i].wrapping_add(v[i]);
        }
    }
    let mut out = String::new();
    for word in state {
        write!(&mut out, "{word:08X}").unwrap();
    }
    out
}

#[cfg(test)]
#[path = "tests/permuted_bindings.rs"]
mod permuted_bindings_tests;

#[cfg(all(windows, any(feature = "dx12", feature = "native-gl-wgl")))]
#[allow(dead_code)]
fn json_string(value: &str) -> String {
    format!("{value:?}")
}
