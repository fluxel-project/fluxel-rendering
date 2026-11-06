struct Tint {
    value: vec4f,
};

@group(0) @binding(0) var<uniform> tint: Tint;
@group(0) @binding(1) var source_tex: texture_2d<f32>;

struct VertexOut {
    @builtin(position) position: vec4f,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOut {
    var positions = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f(3.0, -1.0),
        vec2f(-1.0, 3.0),
    );
    var output: VertexOut;
    output.position = vec4f(positions[index], 0.0, 1.0);
    return output;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return textureLoad(source_tex, vec2i(0, 0), 0) * tint.value;
}
