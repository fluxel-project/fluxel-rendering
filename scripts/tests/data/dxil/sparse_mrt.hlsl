// Sparse-MRT DXIL fixture source.  The checked-in blobs are the test inputs;
// this source only records the ABI and how to regenerate them.
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

// One pixel shader with two outputs at locations 0 and 3, leaving holes at
// locations 1 and 2.  The sparse-MRT raster test drives a pipeline whose MRT
// span is four slots wide; this shader fills slots 0 and 3, and the scope fills
// the holes (1 and 2) with null RTVs.  Each output carries a distinct color so
// the two real targets can be told apart on readback.
struct SparseMrtOutputs {
    float4 low : SV_TARGET0;
    float4 high : SV_TARGET3;
};

SparseMrtOutputs ps_sparse_mrt(VertexOutput input) {
    SparseMrtOutputs output;
    output.low = float4(0.25, 0.5, 0.75, 1.0);
    output.high = float4(0.5, 0.25, 0.125, 1.0);
    return output;
}