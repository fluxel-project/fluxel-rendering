//! Reproducible smoke and measurement harness for logical shader composition.
//!
//! `cargo +1.87.0 bench -p fluxel-shader --bench cold_compile -- --smoke`
//! exercises each workload once.  Omitting `--smoke` records raw timing samples;
//! pass `--samples N` to control the measurement count.

use std::{hint::black_box, sync::Arc, time::Instant};

use fluxel_shader::{
    BindingDeclaration, BindingKind, CallableDeclaration, EntryPoint, ScalarType, ShaderComposer,
    ShaderCompositionRequest, ShaderStage, SourceUnit, VertexInputDeclaration, VertexType,
};

const DEFAULT_SAMPLES: usize = 41;
const WARMUP_SAMPLES: usize = 5;

const SHARED: &str = r#"
struct Camera {
    transform: mat4x4<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var color_texture: texture_2d<f32>;
@group(0) @binding(2) var color_sampler: sampler;

fn apply_tint(color: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(color.rgb * vec3<f32>(0.9, 0.95, 1.0), color.a);
}
"#;

const VERTEX: &str = r#"
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
}

@vertex
fn vertex_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.position = camera.transform * vec4<f32>(input.position, 1.0);
    output.uv = input.uv;
    return output;
}
"#;

const FRAGMENT: &str = r#"
@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return apply_tint(textureSample(color_texture, color_sampler, input.uv));
}
"#;

fn representative_request() -> ShaderCompositionRequest {
    ShaderCompositionRequest::new(vec![
        SourceUnit::new("shared.wgsl", SHARED),
        SourceUnit::new("vertex.wgsl", VERTEX),
        SourceUnit::new("fragment.wgsl", FRAGMENT),
    ])
    .with_entry_point(EntryPoint::new("vertex_main", ShaderStage::Vertex))
    .with_entry_point(EntryPoint::new("fragment_main", ShaderStage::Fragment))
    .with_binding(BindingDeclaration::new(0, 0, BindingKind::Uniform))
    .with_binding(BindingDeclaration::new(0, 1, BindingKind::Texture))
    .with_binding(BindingDeclaration::new(0, 2, BindingKind::Sampler))
    .with_vertex_input(VertexInputDeclaration::new(
        "position",
        0,
        VertexType::Vec3(ScalarType::F32),
    ))
    .with_vertex_input(VertexInputDeclaration::new(
        "uv",
        1,
        VertexType::Vec2(ScalarType::F32),
    ))
    .with_callable(CallableDeclaration {
        name: "apply_tint".into(),
        arguments: vec![VertexType::Vec4(ScalarType::F32)],
        result: Some(VertexType::Vec4(ScalarType::F32)),
    })
    .with_static_option("color-space", "linear")
}

fn cold_compile(request: &ShaderCompositionRequest) {
    let artifact = ShaderComposer::new()
        .compose(request)
        .expect("representative request must compose");
    black_box(artifact);
}

fn cache_hit(
    composer: &mut ShaderComposer,
    request: &ShaderCompositionRequest,
    first: &Arc<fluxel_shader::ShaderArtifact>,
) {
    let second = composer
        .compose(request)
        .expect("cache-hit request must compose");
    assert!(
        Arc::ptr_eq(first, &second),
        "guard must take the cache-hit path"
    );
    black_box(second);
}

fn cache_hit_samples(request: &ShaderCompositionRequest, count: usize) -> Vec<u128> {
    let measure_once = || {
        let mut composer = ShaderComposer::new();
        let first = composer
            .compose(request)
            .expect("representative request must compose");
        let start = Instant::now();
        cache_hit(&mut composer, request, &first);
        start.elapsed().as_nanos()
    };
    for _ in 0..WARMUP_SAMPLES {
        black_box(measure_once());
    }
    (0..count).map(|_| measure_once()).collect()
}

fn samples(mut workload: impl FnMut(), count: usize) -> Vec<u128> {
    for _ in 0..WARMUP_SAMPLES {
        workload();
    }
    (0..count)
        .map(|_| {
            let start = Instant::now();
            workload();
            start.elapsed().as_nanos()
        })
        .collect()
}

fn median(values: &mut [u128]) -> u128 {
    values.sort_unstable();
    values[values.len() / 2]
}

fn print_result(role: &str, mut values: Vec<u128>) {
    let median_ns = median(&mut values);
    let minimum_ns = values[0];
    let maximum_ns = values[values.len() - 1];
    println!(
        "{{\"schema\":1,\"role\":\"{role}\",\"samples\":{},\"warmup_samples\":{WARMUP_SAMPLES},\"minimum_ns\":{minimum_ns},\"median_ns\":{median_ns},\"maximum_ns\":{maximum_ns},\"raw_ns\":{:?}}}",
        values.len(), values
    );
}

fn main() {
    let mut smoke = false;
    let mut sample_count = DEFAULT_SAMPLES;
    for argument in std::env::args().skip(1) {
        if argument == "--bench" {
            // Cargo passes this conventional marker to custom bench harnesses.
        } else if argument == "--smoke" {
            smoke = true;
        } else if let Some(value) = argument.strip_prefix("--samples=") {
            sample_count = value.parse().expect("--samples must be a positive integer");
        } else {
            panic!("unknown argument: {argument}");
        }
    }
    assert!(sample_count > 0, "--samples must be positive");

    let request = representative_request();
    if smoke {
        cold_compile(&request);
        let mut composer = ShaderComposer::new();
        let first = composer
            .compose(&request)
            .expect("representative request must compose");
        cache_hit(&mut composer, &request, &first);
        println!("smoke: representative cold compile and cache-hit guard passed");
        return;
    }

    print_result(
        "representative-cold-cache-miss",
        samples(|| cold_compile(&request), sample_count),
    );
    print_result("guard-cache-hit", cache_hit_samples(&request, sample_count));
}
