# ADR-0011: Build the GL family through three private layers

**Status:** Accepted

## Context

Desktop GL, GLES, and WebGL2 share a stateful execution model but have
different context creation, core versions, extensions, and entry points. A
single public browser/session or raw-context model would leak platform details
and duplicate resource ownership.

## Decision

`backend/gl` has three private responsibilities:

1. `api` normalizes GL-family feature routes and structured failures;
2. `state` is the sole desired/applied/unknown state authority, including
   dependency-invalidated structural caches;
3. native/browser providers and lowering own context calls and translate RHI
   command snapshots through that state authority.

Desktop GL, GLES, and adopted WebGL2 create separate backend-private providers
and terminal device identities. A capability exists only when the selected
core-or-extension route and every required function entry point are present.
No raw context, extension object, browser session, or GL name enters public
RHI. Context loss is device loss.

## Consequences

The backend can use the highest available GL 4.x or ES 3.x context while
retaining the GL 4.0 / ES 3.0 / WebGL2 floor. State caching is an optimization,
not a second command semantics: cache misses, raw access, retirement, and loss
must invalidate facts conservatively. Runtime evidence remains required per
provider route and belongs in conformance results, not this ADR.
