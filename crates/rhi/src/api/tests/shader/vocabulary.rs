//! Section 19.1: the stage set.
//!
//! Each stage gets its own bit in the mask. `use super::*` brings in the
//! fixtures.

use super::*;

// ---------------------------------------------------------------------------
// Section 19.1: the stage set.
// ---------------------------------------------------------------------------

#[test]
fn stage_mask_gives_each_stage_its_own_bit() {
    let masks = [
        (
            ShaderStage::Vertex,
            stage_mask(ShaderStage::Vertex),
            ShaderStages::VERTEX,
        ),
        (
            ShaderStage::Fragment,
            stage_mask(ShaderStage::Fragment),
            ShaderStages::FRAGMENT,
        ),
        (
            ShaderStage::Compute,
            stage_mask(ShaderStage::Compute),
            ShaderStages::COMPUTE,
        ),
    ];
    for (stage, mask, expected) in masks {
        assert_eq!(mask, expected, "{stage:?} maps to the wrong bit");
        assert!(!mask.is_empty());
    }

    // A stage set is a set: the vertex mask must not contain the fragment bit, and
    // the union of every mask must contain every one of them.
    assert!(!ShaderStages::VERTEX.contains(ShaderStages::FRAGMENT));
    let all = ShaderStages::VERTEX
        .union(ShaderStages::FRAGMENT)
        .union(ShaderStages::COMPUTE);
    for (_, mask, _) in masks {
        assert!(all.contains(mask));
    }
    assert!(all.contains(all));
}
