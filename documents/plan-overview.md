# Fluxel × Blender Integration Overview Plan

## 1. Goal

Build one complete authoring-to-runtime loop:

```text
Blender
   ↓ author
Fluxel Material / RenderScene
   ↓ compile
Fluxel Renderer
   ↓
Blender Viewport Preview
   ↓ export
Fluxel Assets
   ↓
Game Runtime
```

The main acceptance criterion is:

> The same Blender scene and material produce the same Fluxel result in Blender preview and in the exported Fluxel runtime.

The goal is not:

```text
Fluxel == EEVEE
Fluxel == Cycles
```

The goal is:

```text
Fluxel in Blender == Fluxel in Runtime
```

Blender is the authoring environment.

Fluxel owns the rendering semantics.

---

## 2. Architecture

```text
Blender
│
├─ Scene Objects
├─ Mesh / UV
├─ Material Nodes
├─ Texture Assets
└─ Camera / Lights
        │
        ▼
Fluxel Blender Adapter
        │
        ├─ Scene -> RenderScene
        ├─ Material Nodes -> MaterialGraph
        └─ Assets -> Fluxel asset references
        │
        ▼
Fluxel Core
│
├─ MaterialGraph
│    ↓
├─ Material IR
│    ↓
├─ Shader Composition
│    ↓
├─ ShaderArtifact
│
├─ RenderScene
│
└─ FramePipeline
     ↓
 RenderGraph
     ↓
    RHI
```

Blender is only an authoring frontend.

Rendering is always performed by Fluxel.

---

## 3. Repository and Crate Layout

Recommended long-term workspace:

```text
fluxel-rendering/
├─ fluxel-rhi
├─ fluxel-rendergraph
├─ fluxel-material
├─ fluxel-shader
└─ fluxel-renderer
```

Blender integration should live separately:

```text
fluxel-blender
```

Responsibilities:

```text
Blender Python add-on
Blender Node -> Fluxel MaterialGraph
Blender Scene -> RenderScene
Fluxel viewport integration
Fluxel asset export
Diagnostics
```

Dependency direction:

```text
Blender Adapter
      ↓
material / renderer
      ↓
shader
      ↓
rendergraph
      ↓
rhi
```

---

## 4. Stage 0 — Finish the Foundation

Prerequisites:

```text
RHI
RenderGraph
Capture / Replay foundation
```

Finish the current `0.16-0.20` foundation train first.

Do not start Blender integration yet.

The RHI foundation must already support:

```text
ShaderArtifact
ShaderInterface
PipelineInterface
RasterPipeline
ComputePipeline
Bindings
```

Otherwise the material system has no stable lower-layer contract.

---

## 5. Stage 1 — Minimal Material Core

Create:

```text
fluxel-material
```

Start with a very small semantic model:

```text
MaterialDomain::Surface

MaterialValueType

Float
Vec2
Vec3
Color3
Color4
Normal3
Texture2D
```

Minimal node set:

```text
Constant
Parameter
TexCoord
Texture2D
SampleTexture2D

Add
Multiply
Mix
Clamp
Normalize
NormalMap
```

Initial output contract:

```text
SurfaceOutput
├─ base_color
├─ metallic
├─ roughness
├─ normal
├─ emissive
└─ opacity
```

Acceptance:

```text
Rust code
-> MaterialGraph
-> Material IR
```

No Blender integration yet.

---

## 6. Stage 2 — Shader Composition

Create:

```text
fluxel-shader
```

Inputs:

```text
Material IR
+
Pass shader template
+
Geometry interface
+
Target profile
```

Outputs:

```text
ShaderArtifact
ShaderInterface
MaterialVariantKey
```

Main responsibilities:

```text
Feature analysis
Variant canonicalization
Permutation pruning
Shader cache
```

Do not generate every possible feature combination.

Use:

```text
Material IR
    ↓
RequiredFeatures
    ↓
Canonical VariantKey
    ↓
Shader Composition
```

Acceptance:

Two materials with different runtime parameters but the same structure should share the same shader variant.

---

## 7. Stage 3 — Minimal RenderScene + FramePipeline

Implement:

```text
RenderScene
RenderObject
RenderView
CullingService
SortingService
FramePipeline
```

The first implementation only needs:

```text
Transform
Bounds
GeometryHandle
MaterialHandle
Camera
one basic light
```

A first pipeline can be very small:

```text
Cull
-> Sort
-> Main Raster Pass
```

The purpose is not to provide a default renderer.

The purpose is to prove:

```text
RenderScene
-> FramePipeline
-> RenderGraph
```

as the official path.

---

## 8. Stage 4 — Material Runtime

Add:

```text
MaterialAsset
MaterialInstance
CompiledMaterialVariant
```

Keep these responsibilities separate:

```text
MaterialGraph        authoring
Material IR          compiler input
Compiled Variant     renderer input
MaterialInstance     runtime parameters
```

Dynamic parameters may include:

```text
base_color
roughness
metallic
texture
emissive
```

Changing them must not require shader recompilation.

Acceptance:

```text
material.set("roughness", 0.8)
```

only updates runtime parameter data.

---

## 9. Stage 5 — Blender Material Import

Start Blender integration only after the Fluxel material path works without Blender.

Support a deliberately small Blender node subset:

```text
Material Output

Principled BSDF:
- Base Color
- Metallic
- Roughness
- Normal
- Emission
- Alpha

Image Texture
Texture Coordinate
Normal Map
RGB
Value
Math: Add / Multiply
Mix
```

Translation:

```text
Blender ShaderNodeTree
        ↓
Fluxel Blender Adapter
        ↓
MaterialGraph
        ↓
Material IR
```

Unsupported nodes must fail explicitly.

Do not silently approximate unsupported behavior.

---

## 10. Stage 6 — Blender Scene Import

Support:

```text
Mesh
Transform
UV
Camera
Material assignment
basic Light
```

Translation:

```text
Blender Scene
        ↓
RenderScene
```

Do not import unrelated systems yet:

```text
gameplay
physics
game logic
navigation
full animation system
```

The first target is rendering data only.

---

## 11. Stage 7 — Blender Viewport Preview

This is the first major end-to-end milestone.

Implement Blender viewport integration:

```text
Blender Scene
        ↓
RenderScene

Blender Material Nodes
        ↓
MaterialGraph
        ↓
Material IR

        ↓
Fluxel FramePipeline
        ↓
RenderGraph
        ↓
RHI
        ↓
Blender Viewport
```

At this point Blender becomes:

```text
Scene Editor
Material Editor
Fluxel Preview Tool
```

Fluxel does not need its own scene or material editor.

---

## 12. Stage 8 — Asset Export

Add:

```text
Export Fluxel Project
```

Exported data may include:

```text
RenderScene assets
Mesh assets
Texture assets
Material assets
Compiled / cooked material variants
```

The runtime must not depend on Blender.

```text
.blend
   ↓ build/export
Fluxel assets
   ↓
Game Runtime
```

---

## 13. Stage 9 — Preview / Runtime Equivalence

Build automated validation around the same input:

```text
Scene
Material
Camera
Light
```

Run it through:

```text
Blender Fluxel Preview
```

and:

```text
Standalone Fluxel Runtime
```

Compare the framebuffer.

Both paths must share:

```text
same Material IR
same shader composition
same shader variant
same material parameters
same textures
same FramePipeline
same color management
```

There must not be two independent shader implementations.

---

## 14. Stage 10 — Expand Blender Compatibility

Only after the full loop works, gradually add:

```text
more Math nodes
Vector Math
ColorRamp
Mapping
procedural textures
more Principled inputs
transparent / masked materials
more normal features
```

Rule:

> Every supported Blender node must map to an explicitly defined Fluxel semantic.

Do not implement Blender nodes merely because Blender exposes them.

---

## 15. Later — JS Runtime Material Control

After `MaterialInstance` is stable:

```text
JS
 ↓
MaterialInstance
```

Example:

```js
material.set("roughness", 0.6);
material.set("damage", 0.8);
```

CSS is still unnecessary at this stage.

---

## 16. Much Later — Style System

Only add a CSS-like layer when there is a real need for:

```text
object classification
material assignment
parameter overrides
state-dependent styling
```

The style layer should only do:

```text
selector
-> material selection
-> MaterialInstance parameter overrides
```

It must not rebuild the MaterialGraph.

The intended separation is:

```text
Blender
    = material structure

Style
    = material assignment / parameters

JS
    = runtime mutation
```

---

## 17. DevTools

DevTools should come last.

Possible inspection path:

```text
RenderObject
├─ matched style
├─ material
├─ parameters
├─ variant key
├─ ShaderArtifact
├─ RenderGraph passes
└─ GPU resources
```

This is a debugging tool, not a material editor.

---

## 18. Recommended Execution Order

```text
0. RHI / RenderGraph foundation
        ↓
1. Material semantic model
        ↓
2. MaterialGraph
        ↓
3. Material IR
        ↓
4. Shader Composition
        ↓
5. RenderScene + FramePipeline
        ↓
6. Material Runtime
        ↓
7. Blender Material Adapter
        ↓
8. Blender Scene Adapter
        ↓
9. Fluxel Blender Viewport
        ↓
10. Export / Runtime
        ↓
11. Preview == Runtime validation
        ↓
12. Expand Blender node coverage
        ↓
13. JS material mutation
        ↓
14. CSS-like Style
        ↓
15. DevTools
```

---

## 19. First Major Product Milestone

The first milestone worth treating as a product-level proof is:

```text
In Blender:

Create a Cube
↓
Connect Principled BSDF
↓
Add a texture
↓
Set Roughness / Metallic
↓
Select Fluxel Render Engine
↓
See the Fluxel result in the Viewport
↓
Export
↓
Open in standalone Fluxel Runtime
↓
Get the same rendered result
```

Once this works, the architecture is proven.

Everything after that is incremental expansion:

```text
more nodes
more materials
more pipelines
JS
CSS
DevTools
```

rather than another foundation rewrite.
