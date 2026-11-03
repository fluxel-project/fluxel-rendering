use crate::{NagaCapability, capabilities::required_naga_capabilities};

const UNUSED_RESULT_INT64_ATOMIC_MIN: &str = r#"
@group(0) @binding(0)
var<storage, read_write> value: atomic<i64>;

@compute @workgroup_size(1)
fn main() {
    atomicMin(&value, 0li);
}
"#;

const INT64_ATOMIC_ADD: &str = r#"
@group(0) @binding(0)
var<storage, read_write> value: atomic<i64>;

@compute @workgroup_size(1)
fn main() {
    atomicAdd(&value, 1li);
}
"#;

#[test]
fn unused_storage_int64_atomic_min_prefers_min_max_over_all_ops() {
    let module = naga::front::wgsl::parse_str(UNUSED_RESULT_INT64_ATOMIC_MIN)
        .expect("fixture must be parseable WGSL");
    let requirements = required_naga_capabilities(&module)
        .expect("the min/max capability must validate the fixture");

    assert!(requirements.contains(&NagaCapability::ShaderInt64));
    assert!(requirements.contains(&NagaCapability::ShaderInt64AtomicMinMax));
    assert!(!requirements.contains(&NagaCapability::ShaderInt64AtomicAllOps));
}

#[test]
fn storage_int64_atomic_add_requires_all_ops() {
    let module = naga::front::wgsl::parse_str(INT64_ATOMIC_ADD)
        .expect("fixture must be parseable WGSL");
    let requirements = required_naga_capabilities(&module)
        .expect("the all-operations capability must validate the fixture");

    assert!(requirements.contains(&NagaCapability::ShaderInt64));
    assert!(requirements.contains(&NagaCapability::ShaderInt64AtomicAllOps));
    assert!(!requirements.contains(&NagaCapability::ShaderInt64AtomicMinMax));
}
