# ADR-0007: Historical fixed renderer recipes remain private

**Status:** Historical migration guidance.

Fixed raster recipes proved earlier vertical slices. They may remain private
implementation and evidence inputs while consumers migrate, but they do not
define a public RHI, material, shader, Graph, or renderer extension contract.

New rendering work follows the data-oriented Renderer target: materials and
Shader compose programs, FramePipeline chooses policy, Graph declares work, and
the selected RHI executes it. A fixed recipe must not become a substitute for a
general contract that a real consumer has not yet proved.
