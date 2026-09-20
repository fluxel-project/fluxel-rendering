// Raster DXIL fixture source.  The checked-in blobs are the test inputs; this
// source only records their ABI and how to regenerate them.
struct VertexInput {
    float2 position : LOCATION0;
};

struct VertexOutput {
    float4 position : SV_POSITION;
};

VertexOutput vs_main(VertexInput input) {
    VertexOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    return output;
}

float4 ps_solid(VertexOutput input) : SV_TARGET {
    return float4(0.25, 0.5, 0.75, 1.0);
}

Texture2D source_texture : register(t0);
// Fluxel ABI 1.0 maps a binding's logical slot to its HLSL register number.
// The sampled texture occupies logical slot 0 (`t0`) and its sampler slot 1
// (`s1`), because portable slots are unique across binding classes.
SamplerState source_sampler : register(s1);

float4 ps_sampled(VertexOutput input) : SV_TARGET {
    return source_texture.SampleLevel(source_sampler, float2(0.5, 0.5), 0.0);
}
