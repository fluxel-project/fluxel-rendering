# PBR environment material acceptance

This standalone workspace is a renderer consumer for the initial split-sum
specular IBL contract. It supplies an exact binary16 lat-long mip chain and a
deliberately artificial binary16 BRDF-LUT control table, then executes the
same metallic-roughness material through both Forward and Deferred routes.

The LUT is not a generated BRDF integration result and the radiance chain is
not evidence of an environment prefiltering pipeline. This consumer verifies
resource admission, mip selection, camera-dependent reflection, and the
height-correlated split-sum calculation only.

On Windows run one native backend, for example:

```text
cargo run --locked --features dx12 -- --offscreen --output target/ibl-dx12
```

Each case saves its source input and independent payload-derived oracle before
that case allocates resources, uploads, records, or submits work. Both routes
retain raw companion readbacks and check nine interior pixels against the
oracle after writing those bytes, with tolerance 5 for RGBA8 G-buffer
normal/roughness quantization and float arithmetic.

Visible Surface acceptance is available on Windows:

```text
cargo run --locked --features dx12 -- --visible --exit-after-capture --output target/ibl-visible-dx12
```

It creates a hidden window and configures the supported opaque FIFO Surface.
DX12 shows it in the background after its first presentation; WGL maps it in
the background before its first present because that driver requires a mapped
drawable. Every Forward and Deferred Surface submission has a same-input
offscreen companion. Windows Graphics Capture saves the client frame and
compares dense RGB exactly with that companion outside the fixed two 16×16 DWM
bottom-corner masks (512 pixels total); alpha is ignored for the opaque
Surface. DX12 and WGL use their own host/device lifetimes and diagnostics. No
capture operation changes focus, z-order, or visibility.
