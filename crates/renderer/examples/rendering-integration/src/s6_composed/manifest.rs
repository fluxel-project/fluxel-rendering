//! Artifact manifest for the composed Forward workload.

pub(super) fn manifest() -> String {
    let artifacts = |names: &[&str]| -> String {
        names
            .iter()
            .flat_map(|name| {
                [
                    format!("\"s6-{name}-input.json\""),
                    format!("\"s6-{name}-expected.rgba\""),
                    format!("\"s6-{name}-expected.ppm\""),
                    format!("\"s6-{name}-actual.rgba\""),
                    format!("\"s6-{name}-actual.ppm\""),
                ]
            })
            .collect::<Vec<_>>()
            .join(",")
    };
    let original = format!(
        "\"s6-initial-mesh-geometry.txt\",\"s6-initial-mesh-geometry-stream-0.bin\",\"s6-initial-mesh-material.txt\",\"s6-initial-mesh-texture.rgba\",\"s6-initial-quad-geometry.txt\",\"s6-initial-quad-geometry-stream-0.bin\",\"s6-initial-quad-material.txt\",\"s6-initial-quad-texture.rgba\",\"s6-stable-mask.bin\",\"s6-composed-cache.json\",\"s6-composed-measurements.json\",{}",
        artifacts(&["frame-0", "frame-1", "frame-2"])
    );
    format!(
        "{{\"schemaVersion\":1,\"workload\":\"s6-composed-offscreen-typed-quad\",\"cases\":[{{\"name\":\"original-three-frame-cache\",\"artifacts\":[{original}]}},{{\"name\":\"enabled-disabled-enabled\",\"artifacts\":[{}]}},{{\"name\":\"independent-providers\",\"artifacts\":[{}]}},{{\"name\":\"three-pending-replacements\",\"artifacts\":[\"s6-pending-assets.json\",\"s6-pending-lifecycle.json\",\"s6-pending-new-quad-geometry-stream-0.bin\",\"s6-pending-new-quad-geometry.txt\",\"s6-pending-new-quad-material.txt\",\"s6-pending-new-quad-texture.rgba\",{}]}}]}}\n",
        artifacts(&["toggle-enabled", "toggle-disabled", "toggle-reenabled"]),
        artifacts(&["provider-marker-only", "provider-mesh-only"]),
        artifacts(&["pending-old", "pending-new", "pending-new-warm"])
    )
}
