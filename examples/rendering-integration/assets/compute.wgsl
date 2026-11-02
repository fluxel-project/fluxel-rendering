@group(0) @binding(0)
var<storage, read_write> values: array<u32>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    if (index < arrayLength(&values)) {
        values[index] = values[index] * 3u + 7u;
    }
}
