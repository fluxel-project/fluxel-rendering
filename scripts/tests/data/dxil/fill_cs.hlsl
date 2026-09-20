// The fixture's source. Kept beside the blob it produced so that the bytecode
// can be regenerated rather than only trusted, and so that a reader can see the
// entry point and the thread-group size the blob was compiled for.
//
// Command line (see scripts/tests/data/dxil/README.md for the full note):
//   dxc -T cs_6_0 -E main -Fo fill_cs.dxil fill_cs.hlsl

RWByteAddressBuffer output : register(u0);

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    output.Store(id.x * 4, id.x);
}
