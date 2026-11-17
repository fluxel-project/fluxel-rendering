//! Deterministic source-faithful data for RHI example 04.
//!
//! This module does not create a GPU object.  The renderer consumer expands the
//! original index buffer but preserves its triangle order and per-vertex colour.

pub const GRID_SIDE: u32 = 5;
pub const CUBE_COUNT: u32 = GRID_SIDE * GRID_SIDE * GRID_SIDE;

/// Source interleaved cube vertices: position followed by RGB.
pub const CUBE_VERTICES: [[f32; 6]; 8] = [
    [-1.0, -1.0, 1.0, 1.0, 0.0, 0.0],
    [1.0, -1.0, 1.0, 0.0, 1.0, 0.0],
    [1.0, 1.0, 1.0, 0.0, 0.0, 1.0],
    [-1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    [-1.0, -1.0, -1.0, 1.0, 0.0, 0.0],
    [1.0, -1.0, -1.0, 0.0, 1.0, 0.0],
    [1.0, 1.0, -1.0, 0.0, 0.0, 1.0],
    [-1.0, 1.0, -1.0, 0.0, 0.0, 0.0],
];

/// Exact 36-element index order from the RHI source.
pub const CUBE_INDICES: [u32; 36] = [
    0, 1, 2, 2, 3, 0, 1, 5, 6, 6, 2, 1, 7, 6, 5, 5, 4, 7, 4, 0, 3, 3, 7, 4, 4, 5, 1, 1, 0, 4, 3, 2,
    6, 6, 7, 3,
];

/// Column-major local-to-world transform for one source lattice cube.
pub fn model(index: u32, time: f32) -> [[f32; 4]; 4] {
    assert!(index < CUBE_COUNT);
    let x = index / 25;
    let y = (index / 5) % 5;
    let z = index % 5;
    let position = [
        -10.0 + x as f32 * 5.0,
        -10.0 + y as f32 * 5.0,
        -10.0 + z as f32 * 5.0,
    ];
    let mut rng = CubeRng::for_cube(index);
    let initial: [f32; 3] = core::array::from_fn(|_| rng.signed() * 2.0 * core::f32::consts::PI);
    let speed: [f32; 3] = core::array::from_fn(|_| rng.signed());
    let angles: [f32; 3] = core::array::from_fn(|i| initial[i] + time * speed[i]);
    multiply(
        translation(position),
        multiply(
            rotation_z(angles[2]),
            multiply(rotation_y(angles[1]), rotation_axis_xy(angles[0])),
        ),
    )
}

/// The reference's right-handed camera with WebGPU depth mapping.
pub fn camera(extent: [u32; 2]) -> [[f32; 4]; 4] {
    let aspect = extent[0].max(1) as f32 / extent[1].max(1) as f32;
    let f = 1.0 / 30f32.to_radians().tan();
    let projection = [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, 256.0 / -255.9, -1.0],
        [0.0, 0.0, -25.6 / 255.9, 0.0],
    ];
    multiply(projection, translation([0.0, 0.0, -30.0]))
}

pub fn multiply(left: [[f32; 4]; 4], right: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut product = [[0.0; 4]; 4];
    for column in 0..4 {
        for row in 0..4 {
            product[column][row] = (0..4)
                .map(|inner| left[inner][row] * right[column][inner])
                .sum();
        }
    }
    product
}

pub fn encode_matrix(matrix: [[f32; 4]; 4]) -> [u8; 64] {
    let mut bytes = [0; 64];
    for (i, value) in matrix.into_iter().flatten().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn translation(position: [f32; 3]) -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [position[0], position[1], position[2], 1.0],
    ]
}
fn rotation_axis_xy(angle: f32) -> [[f32; 4]; 4] {
    let (s, c) = angle.sin_cos();
    let half = 0.5 * (1.0 - c);
    let diagonal = c + half;
    let xy = half;
    let z = s * core::f32::consts::FRAC_1_SQRT_2;
    [
        [diagonal, xy, -z, 0.0],
        [xy, diagonal, z, 0.0],
        [z, -z, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}
fn rotation_y(angle: f32) -> [[f32; 4]; 4] {
    let (s, c) = angle.sin_cos();
    [
        [c, 0.0, -s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [s, 0.0, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}
fn rotation_z(angle: f32) -> [[f32; 4]; 4] {
    let (s, c) = angle.sin_cos();
    [
        [c, s, 0.0, 0.0],
        [-s, c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

struct CubeRng(u32);
impl CubeRng {
    fn for_cube(index: u32) -> Self {
        let mut seed = index.wrapping_add(0x9e37_79b9);
        seed ^= seed >> 16;
        seed = seed.wrapping_mul(0x85eb_ca6b);
        seed ^= seed >> 13;
        seed = seed.wrapping_mul(0xc2b2_ae35);
        Self(seed ^ (seed >> 16))
    }
    fn signed(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((self.0 >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0
    }
}
