use external_renderer::{
    quad_geometry_data, quad_marker_material_definition, QuadAssets, QuadInput, QuadMarker,
};
use fluxel_assets::{Acquire, AssetSnapshot, AssetStore, Production, ResidentBytes};
use fluxel_material::{
    BindingLocation, MaterialAsset, MaterialDefinition, MaterialFragment, MaterialParameter,
    MaterialResource, MaterialResourceKind, MaterialType, SurfaceOutput,
};
use fluxel_renderer::{
    ColorTextureAsset, ColorTextureData, ForwardAssets, ForwardResources, GeometryAsset,
    GeometryAttribute, GeometryData, GeometryLayout, GeometryStream, RenderBounds, RenderRecord,
    RenderTransform, SceneAssetRef,
};
use fluxel_rhi::api::{VertexFormat, VertexStepMode};
use std::{collections::BTreeMap, error::Error, fs, path::Path};

pub(super) const MESH: [u8; 4] = [0, 255, 0, 255];
pub(super) const QUAD: [u8; 4] = [32, 32, 223, 255];

pub(super) struct Assets {
    pub(super) geometry: AssetStore<GeometryAsset, GeometryData, String>,
    pub(super) material: AssetStore<MaterialAsset, MaterialDefinition, String>,
    pub(super) texture: AssetStore<ColorTextureAsset, ColorTextureData, String>,
    pub(super) mesh_g: AssetSnapshot<GeometryAsset, GeometryData>,
    pub(super) quad_g: AssetSnapshot<GeometryAsset, GeometryData>,
    pub(super) mesh_m: AssetSnapshot<MaterialAsset, MaterialDefinition>,
    pub(super) quad_m: AssetSnapshot<MaterialAsset, MaterialDefinition>,
    pub(super) mesh_t: AssetSnapshot<ColorTextureAsset, ColorTextureData>,
    pub(super) quad_t: AssetSnapshot<ColorTextureAsset, ColorTextureData>,
}
impl Assets {
    pub(super) fn new() -> Result<Self, Box<dyn Error>> {
        let geometry = AssetStore::new();
        let material = AssetStore::new();
        let texture = AssetStore::new();
        Ok(Self {
            mesh_g: geometry_snapshot(&geometry, triangle())?,
            quad_g: geometry_snapshot(&geometry, quad_geometry_data())?,
            mesh_m: material_snapshot(&material, mesh_material())?,
            quad_m: material_snapshot(&material, quad_marker_material_definition())?,
            mesh_t: texture_snapshot(&texture, MESH)?,
            quad_t: texture_snapshot(&texture, [255; 4])?,
            geometry,
            material,
            texture,
        })
    }
    pub(super) fn forward(&self) -> ForwardAssets {
        ForwardAssets {
            geometry: vec![self.mesh_g.clone(), self.quad_g.clone()],
            materials: vec![self.mesh_m.clone(), self.quad_m.clone()],
            textures: vec![self.mesh_t.clone(), self.quad_t.clone()],
        }
    }
    pub(super) fn record(&self, out: bool) -> RenderRecord<ForwardResources> {
        RenderRecord {
            resources: ForwardResources {
                geometry: SceneAssetRef::from_snapshot(&self.mesh_g),
                material: SceneAssetRef::from_snapshot(&self.mesh_m),
                texture: SceneAssetRef::from_snapshot(&self.mesh_t),
                tint: [1.0; 4],
            },
            transform: if out {
                translated()
            } else {
                RenderTransform::identity()
            },
            bounds: if out {
                RenderBounds::new([2.0; 3], [3.0; 3]).unwrap()
            } else {
                RenderBounds::new([-0.5, -0.5, 0.0], [0.5, 0.5, 0.0]).unwrap()
            },
        }
    }
    pub(super) fn input(&self, out: bool, order: i64) -> QuadInput {
        QuadInput {
            markers: vec![QuadMarker {
                id: 7,
                center: if out { [2.0, 0.0, 0.0] } else { [0.0; 3] },
                half_extent: [0.55; 2],
                tint: [1.0, 0.0, 0.0, 1.0],
                assets: QuadAssets {
                    geometry: SceneAssetRef::from_snapshot(&self.quad_g),
                    material: SceneAssetRef::from_snapshot(&self.quad_m),
                    texture: SceneAssetRef::from_snapshot(&self.quad_t),
                },
                order,
            }],
        }
    }
    pub(super) fn input_with(
        &self,
        g: &AssetSnapshot<GeometryAsset, GeometryData>,
        m: &AssetSnapshot<MaterialAsset, MaterialDefinition>,
        t: &AssetSnapshot<ColorTextureAsset, ColorTextureData>,
        order: i64,
    ) -> QuadInput {
        QuadInput {
            markers: vec![QuadMarker {
                id: 7,
                center: [0.0; 3],
                half_extent: [0.55; 2],
                tint: [1.0, 0.0, 0.0, 1.0],
                assets: QuadAssets {
                    geometry: SceneAssetRef::from_snapshot(g),
                    material: SceneAssetRef::from_snapshot(m),
                    texture: SceneAssetRef::from_snapshot(t),
                },
                order,
            }],
        }
    }
    pub(super) fn write(&self, output: &Path) -> Result<(), Box<dyn Error>> {
        for (name, g) in [("mesh", &self.mesh_g), ("quad", &self.quad_g)] {
            for (i, bytes) in g.value().streams.iter().enumerate() {
                fs::write(
                    output.join(format!("s6-initial-{name}-geometry-stream-{i}.bin")),
                    bytes,
                )?;
            }
            fs::write(
                output.join(format!("s6-initial-{name}-geometry.txt")),
                format!("{:#?}\n", g.value()),
            )?;
        }
        for (name, m) in [("mesh", &self.mesh_m), ("quad", &self.quad_m)] {
            fs::write(
                output.join(format!("s6-initial-{name}-material.txt")),
                format!("{:#?}\n", m.value()),
            )?;
        }
        for (name, t) in [("mesh", &self.mesh_t), ("quad", &self.quad_t)] {
            fs::write(
                output.join(format!("s6-initial-{name}-texture.rgba")),
                &t.value().rgba8,
            )?;
        }
        Ok(())
    }
}
pub(super) fn replace_geometry(
    store: &AssetStore<GeometryAsset, GeometryData, String>,
    old: &AssetSnapshot<GeometryAsset, GeometryData>,
    data: GeometryData,
) -> Result<AssetSnapshot<GeometryAsset, GeometryData>, Box<dyn Error>> {
    match store.request_replacement(old.handle())? {
        Production::Producer(p) => {
            let bytes = data.streams.iter().map(|x| x.len() as u64).sum();
            Ok(p.commit(data, ResidentBytes::new(bytes))?)
        }
        Production::Waiter(_) => Err("S6 geometry replacement unexpectedly waited".into()),
    }
}
pub(super) fn replace_material(
    store: &AssetStore<MaterialAsset, MaterialDefinition, String>,
    old: &AssetSnapshot<MaterialAsset, MaterialDefinition>,
    data: MaterialDefinition,
) -> Result<AssetSnapshot<MaterialAsset, MaterialDefinition>, Box<dyn Error>> {
    match store.request_replacement(old.handle())? {
        Production::Producer(p) => Ok(p.commit(data, ResidentBytes::new(0))?),
        Production::Waiter(_) => Err("S6 material replacement unexpectedly waited".into()),
    }
}
pub(super) fn replace_texture(
    store: &AssetStore<ColorTextureAsset, ColorTextureData, String>,
    old: &AssetSnapshot<ColorTextureAsset, ColorTextureData>,
    data: ColorTextureData,
) -> Result<AssetSnapshot<ColorTextureAsset, ColorTextureData>, Box<dyn Error>> {
    let bytes = data.rgba8.len() as u64;
    match store.request_replacement(old.handle())? {
        Production::Producer(p) => Ok(p.commit(data, ResidentBytes::new(bytes))?),
        Production::Waiter(_) => Err("S6 texture replacement unexpectedly waited".into()),
    }
}
fn geometry_snapshot(
    s: &AssetStore<GeometryAsset, GeometryData, String>,
    d: GeometryData,
) -> Result<AssetSnapshot<GeometryAsset, GeometryData>, Box<dyn Error>> {
    let h = s.create()?;
    match s.acquire(&h)? {
        Acquire::Producer(p) => {
            let n = d.streams.iter().map(|x| x.len() as u64).sum();
            Ok(p.commit(d, ResidentBytes::new(n))?)
        }
        _ => Err("S6 geometry producer missing".into()),
    }
}
fn material_snapshot(
    s: &AssetStore<MaterialAsset, MaterialDefinition, String>,
    d: MaterialDefinition,
) -> Result<AssetSnapshot<MaterialAsset, MaterialDefinition>, Box<dyn Error>> {
    let h = s.create()?;
    match s.acquire(&h)? {
        Acquire::Producer(p) => Ok(p.commit(d, ResidentBytes::new(0))?),
        _ => Err("S6 material producer missing".into()),
    }
}
fn texture_snapshot(
    s: &AssetStore<ColorTextureAsset, ColorTextureData, String>,
    c: [u8; 4],
) -> Result<AssetSnapshot<ColorTextureAsset, ColorTextureData>, Box<dyn Error>> {
    let h = s.create()?;
    match s.acquire(&h)? {
        Acquire::Producer(p) => Ok(p.commit(
            ColorTextureData {
                extent: [1, 1],
                rgba8: c.to_vec(),
            },
            ResidentBytes::new(4),
        )?),
        _ => Err("S6 texture producer missing".into()),
    }
}
fn triangle() -> GeometryData {
    let mut b = Vec::new();
    for p in [
        [-0.45_f32, -0.40, 0.0],
        [0.45, -0.40, 0.0],
        [0.0, 0.45, 0.0],
    ] {
        for v in p.into_iter().chain([0.0, 0.0]).chain([1.0; 4]) {
            b.extend_from_slice(&v.to_le_bytes())
        }
    }
    GeometryData {
        layout: GeometryLayout {
            streams: vec![GeometryStream {
                stride: 36,
                step_mode: VertexStepMode::Vertex,
                attributes: vec![
                    GeometryAttribute {
                        semantic: "POSITION".into(),
                        offset: 0,
                        format: VertexFormat::Float32(3),
                    },
                    GeometryAttribute {
                        semantic: "UV".into(),
                        offset: 12,
                        format: VertexFormat::Float32(2),
                    },
                    GeometryAttribute {
                        semantic: "COLOR".into(),
                        offset: 20,
                        format: VertexFormat::Float32(4),
                    },
                ],
            }],
        },
        streams: vec![b],
        vertices: 0..3,
    }
}
fn mesh_material() -> MaterialDefinition {
    MaterialDefinition::new(vec![MaterialFragment{name:"s6_mesh".into(),inputs:vec![],output:MaterialType::Vec4,calls:vec![],wgsl:"fn s6_mesh() -> vec4<f32> { return textureLoad(material_texture, vec2<i32>(0,0),0)*tint; }".into()}],"s6_mesh",vec![MaterialParameter{name:"tint".into(),ty:MaterialType::Vec4,resource:"material_tint".into()}],vec![MaterialResource{name:"material_tint".into(),kind:MaterialResourceKind::UniformVec4,location:BindingLocation{group:0,binding:0}},MaterialResource{name:"material_texture".into(),kind:MaterialResourceKind::Texture2dF32,location:BindingLocation{group:0,binding:1}}],BTreeMap::new(),SurfaceOutput::Color)
}
fn translated() -> RenderTransform {
    RenderTransform::from_columns([
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [2., 0., 0., 1.],
    ])
    .unwrap()
}

pub(super) fn small_quad() -> GeometryData {
    let mut data = quad_geometry_data();
    for vertex in data.streams[0].chunks_exact_mut(36) {
        for offset in [0, 4, 8] {
            let mut bytes = [0; 4];
            bytes.copy_from_slice(&vertex[offset..offset + 4]);
            let scaled = f32::from_le_bytes(bytes) * 0.5;
            vertex[offset..offset + 4].copy_from_slice(&scaled.to_le_bytes());
        }
    }
    data
}
pub(super) fn replacement_quad_material() -> MaterialDefinition {
    MaterialDefinition::new(vec![MaterialFragment { name:"s6_replacement_quad".into(), inputs:vec![], output:MaterialType::Vec4, calls:vec![], wgsl:"fn s6_replacement_quad() -> vec4<f32> { return textureLoad(quad_texture, vec2<i32>(0,0),0) * vec4<f32>(tint.rgb * 0.5 + vec3<f32>(0.25), tint.a); }".into() }], "s6_replacement_quad", vec![MaterialParameter { name:"tint".into(), ty:MaterialType::Vec4, resource:"quad_tint".into() }], vec![MaterialResource { name:"quad_tint".into(), kind:MaterialResourceKind::UniformVec4, location:BindingLocation { group:0,binding:0 } }, MaterialResource { name:"quad_texture".into(), kind:MaterialResourceKind::Texture2dF32, location:BindingLocation { group:0,binding:1 } }], BTreeMap::new(), SurfaceOutput::Color)
}
