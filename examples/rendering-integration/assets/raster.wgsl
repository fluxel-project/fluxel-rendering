struct VertexOut {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
};

@vertex
fn vs_main(
    @location(0) position: vec2f,
    @location(1) color: vec4f,
) -> VertexOut {
    var output: VertexOut;
    output.position = vec4f(position, 0.0, 1.0);
    output.color = color;
    return output;
}

@fragment
fn fs_main(input: VertexOut) -> @location(0) vec4f {
    return input.color;
}
