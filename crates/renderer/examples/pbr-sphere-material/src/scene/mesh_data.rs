use fluxel_renderer::{GeometryAttribute, GeometryData, GeometryLayout, GeometryStream};
use fluxel_rhi::api::{VertexFormat, VertexStepMode};

pub const VERTEX_BYTES: usize = 48;
pub const VERTICES_PER_SPHERE: u32 = 240;

pub fn sphere_centres() -> [[f32; 3]; 6] {
    [
        [-2., 0.85, 0.],
        [0., 0.85, 0.],
        [2., 0.85, 0.],
        [-2., -1.05, 0.],
        [0., -1.05, 0.],
        [2., -1.05, 0.],
    ]
}

pub fn mesh() -> GeometryData {
    let streams = vec![all_sphere_bytes()];
    GeometryData {
        layout: layout(),
        streams,
        vertices: 0..VERTICES_PER_SPHERE * 6,
    }
}

pub fn individual_meshes() -> Vec<GeometryData> {
    all_sphere_bytes()
        .chunks_exact(VERTICES_PER_SPHERE as usize * VERTEX_BYTES)
        .zip(sphere_centres())
        .map(|(bytes, centre)| GeometryData {
            layout: layout(),
            streams: vec![localise_positions(bytes, centre)],
            vertices: 0..VERTICES_PER_SPHERE,
        })
        .collect()
}

fn localise_positions(world_bytes: &[u8], centre: [f32; 3]) -> Vec<u8> {
    let mut local_bytes = world_bytes.to_vec();
    for vertex in local_bytes.chunks_exact_mut(VERTEX_BYTES) {
        for axis in 0..3 {
            let offset = axis * 4;
            let world =
                f32::from_le_bytes(vertex[offset..offset + 4].try_into().expect("position"));
            vertex[offset..offset + 4].copy_from_slice(&(world - centre[axis]).to_le_bytes());
        }
    }
    local_bytes
}

fn layout() -> GeometryLayout {
    GeometryLayout {
        streams: vec![GeometryStream {
            stride: VERTEX_BYTES as u64,
            step_mode: VertexStepMode::Vertex,
            attributes: vec![
                GeometryAttribute {
                    semantic: "POSITION".into(),
                    offset: 0,
                    format: VertexFormat::Float32(3),
                },
                GeometryAttribute {
                    semantic: "NORMAL".into(),
                    offset: 12,
                    format: VertexFormat::Float32(3),
                },
                GeometryAttribute {
                    semantic: "UV0".into(),
                    offset: 24,
                    format: VertexFormat::Float32(2),
                },
                GeometryAttribute {
                    semantic: "TANGENT".into(),
                    offset: 32,
                    format: VertexFormat::Float32(4),
                },
            ],
        }],
    }
}

fn all_sphere_bytes() -> Vec<u8> {
    let mut bytes = Vec::new();
    for (index, centre) in sphere_centres().into_iter().enumerate() {
        let orm_u = (index as f32 + 0.5) / 6.;
        for x in 0..8 {
            let u0 = x as f32 / 8.;
            let u1 = (x + 1) as f32 / 8.;
            push_vertex(&mut bytes, centre, u0, 0., orm_u);
            push_vertex(&mut bytes, centre, u1, 1. / 6., orm_u);
            push_vertex(&mut bytes, centre, u0, 1. / 6., orm_u);
        }
        for y in 1..5 {
            for x in 0..8 {
                let u0 = x as f32 / 8.;
                let u1 = (x + 1) as f32 / 8.;
                let v0 = y as f32 / 6.;
                let v1 = (y + 1) as f32 / 6.;
                for (u, v) in [(u0, v0), (u1, v0), (u0, v1), (u0, v1), (u1, v0), (u1, v1)] {
                    push_vertex(&mut bytes, centre, u, v, orm_u);
                }
            }
        }
        for x in 0..8 {
            let u0 = x as f32 / 8.;
            let u1 = (x + 1) as f32 / 8.;
            push_vertex(&mut bytes, centre, u0, 5. / 6., orm_u);
            push_vertex(&mut bytes, centre, u1, 5. / 6., orm_u);
            push_vertex(&mut bytes, centre, u1, 1., orm_u);
        }
    }
    bytes
}

fn push_vertex(bytes: &mut Vec<u8>, centre: [f32; 3], u: f32, v: f32, orm_u: f32) {
    let azimuth = u * core::f32::consts::TAU;
    let polar = v * core::f32::consts::PI;
    let normal = [
        polar.sin() * azimuth.cos(),
        polar.cos(),
        polar.sin() * azimuth.sin(),
    ];
    let position = [
        centre[0] + normal[0] * 0.72,
        centre[1] + normal[1] * 0.72,
        centre[2] + normal[2] * 0.72,
    ];
    let tangent = [-azimuth.sin(), 0., azimuth.cos(), 1.];
    for value in position
        .into_iter()
        .chain(normal)
        .chain([orm_u, 0.5])
        .chain(tangent)
    {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}
