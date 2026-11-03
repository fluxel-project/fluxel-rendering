# Renderer integration tests

Add public-API integration tests here as those APIs are implemented. Keep unit
test implementations under `src/tests/`, grouped by their production component.
Production modules retain only test-only module declarations that reference them.
Real-target support requires a runnable ecosystem consumer and retained evidence;
an empty scaffold or a successful build is not stage completion. The independent
`examples/external-renderer/` workspace currently supplies the S4 CPU reference
consumer for the public RenderScene and CPU-feature boundaries. It implements
the provisional Renderer-exported `CpuFeature` facade with issuer-bound typed
storage and controlled CPU phase contributions. It is not a GPU integration
test and does not establish a FramePipeline contract.
