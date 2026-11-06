# Renderer integration tests

Public-API integration tests belong here. Unit-test implementations belong under
`src/tests/`, grouped by their production component; production modules retain
only test-only declarations that reference them.

The independent `examples/external-renderer/` consumer exercises public
RenderScene, CPU Feature, and controlled FramePipeline boundaries. It remains a
CPU/public-contract consumer and does not create a GPU device or establish
backend support. Real-target behavior requires a runnable ecosystem consumer and
retained evidence; a successful build alone is insufficient.
