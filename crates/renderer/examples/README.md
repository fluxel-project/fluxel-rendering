# Renderer examples

This directory contains standalone consumers of public Renderer APIs:

- `external-renderer/` checks RenderScene, CPU Feature, controlled Forward
  composition, and private-boundary compile failures without creating a device.
- `rendering-integration/` owns the Windows DX12 and desktop OpenGL/WGL fixture
  executables and their independent locks.

Platform event loops and Surface lifecycle composition stay in these consumers,
not in the Renderer library. They must use the public facade and cannot access
private Graph, allocator, encoder, or submission state.
