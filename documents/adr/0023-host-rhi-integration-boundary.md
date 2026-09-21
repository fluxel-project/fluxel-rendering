# ADR-0023: Keep host windows separate from RHI presentation integration

**Status:** Accepted

## Context

`fluxel-host` is a pure platform leaf. It owns native-window lifetimes and
reports portable lifecycle facts through standard `raw-window-handle` traits;
it does not depend on RHI. Browser canvas/DOM lifecycle has the same role in
`@fluxel/browser`.

RHI presentation has a different lifetime. A `PresentationTarget` may be
supplied during adapter preflight before a device exists, may be examined by
more than one provider, and later obtains backend-private per-device backing.
Its native backing cannot safely be a copied `HWND`, `ANativeWindow*`,
`UIView*`, canvas, or context: those handles are borrowed from the host and
their validity ends at a platform lifecycle boundary. Making one of them a
public RHI field would either leak native types or permit a safe-looking target
to outlive its host object.

## Decision

Keep three ownership domains distinct:

```text
fluxel-host / @fluxel-browser
    owns native window or canvas lifetime and lifecycle facts
    does not depend on RHI

host-to-RHI integration adapter
    depends on host plus RHI
    borrows standard raw handles only while constructing/retiring an opaque
    PresentationTarget; retains the host object for the configured-surface lease

fluxel-rhi
    owns opaque PresentationTarget / configuration / acquired-frame semantics
    keeps backend-native target state private
    does not depend on fluxel-host or a browser session type
```

The adapter is the only layer permitted to translate a host's raw handle into
the backend-private target registration path. Its public result is an ordinary
RHI `PresentationTarget`; examples and conformance workloads then use only
portable RHI operations. A host destruction event first retires the adapter's
surface lease, then invalidates future acquire/configure requests through the
existing structured RHI outcome. It must never leave an opaque target pointing
at a freed host window.

The integration adapter is a consumer-facing convenience layer, not a new RHI
execution domain. It does not introduce a browser session/token, resource
manager, render graph, or native GPU object into `api`.

## Consequences

- `crates/rhi/tests/common` contains one portable workload per contract and
  `crates/rhi/tests/harness` owns async terminal/outcome policy. Native host
  setup belongs to adapter fixtures, not to backend-local test bodies.
- `crates/rhi/examples` may use the adapter so examples do not name
  HWND, `ANativeWindow`, UIKit, DOM canvas, Vulkan surfaces, or DXGI objects.
- Windows, Android, iOS, and browser adapters can mature independently; a
  platform without a host adapter is `Skipped`, not a reason to weaken an RHI
  conformance case.
- The adapter must add positive (live host target), negative (destroyed or
  foreign target), and boundary (resize/suspend/destroy ordering) tests before
  publishing a platform route.
