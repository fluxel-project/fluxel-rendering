# ADR-0017: Keep presentation frames separate from textures and completion

**Status:** Accepted

## Context

Swapchain images, WebGPU current textures, CAMetalDrawables, and GL default
framebuffers do not share normal texture ownership. Present timing also differs
from GPU completion.

## Decision

`FrameAttachment` is distinct from `Texture` and `TextureView`; an acquired
frame is non-cloneable and has one lifecycle. Presentation is planned before
submission, while acceptance, completion, and present outcome are separate.
`abandon()` terminates ownership without promising a cheap native release.

## Consequences

Backend lowering can normalize attachments privately without leaking a
swapchain model. WebGPU waits for browser current-texture expiry after abandon
before issuing a distinct frame identity.
