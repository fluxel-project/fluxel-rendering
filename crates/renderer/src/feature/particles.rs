//! Typed CPU particle state and the bounded M1 GPU particle shader ABI.
//!
//! The renderer owns the persistent ping-pong buffers, Graph declarations,
//! dispatch, vertex binding and completion retirement. This module owns only
//! the portable state bytes and WGSL kernels shared by those later owners.

use fluxel_rhi::api::VertexFormat;

/// Camera and fade values for one GPU particle draw.
///
/// The selected renderer view supplies the existing clip-from-world transform.
/// This value supplies the corresponding world-to-view transform so the shader
/// can rotate quad corners in view space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuParticleView {
    /// The selected view's world-to-view transform, before projection.
    pub view_from_world: crate::RenderTransform,
    /// Billboard rotation pivot in quad coordinates.
    pub translation_pivot: [f32; 2],
    /// Linear RGBA color at the end of a particle's life.
    pub dead_color: [f32; 4],
}

impl GpuParticleView {
    /// Uses Babylon's zero pivot and opaque-black dead color with this camera.
    pub const fn new(view_from_world: crate::RenderTransform) -> Self {
        Self {
            view_from_world,
            translation_pivot: [0.0; 2],
            dead_color: [0.0, 0.0, 0.0, 1.0],
        }
    }
}

/// Bytes occupied by one GPU particle state.
///
/// The state is five 16-byte `vec4<f32>` records. It is valid both as WGSL
/// storage data and as an instance-rate vertex stream.
pub const GPU_PARTICLE_STATE_STRIDE: u64 = 80;

/// One instance-rate vertex input in the particle-state ABI.
#[derive(Clone, Copy, Debug)]
pub struct GpuParticleVertexAttribute {
    /// Semantic consumed by the particle render template.
    pub semantic: &'static str,
    /// Byte offset within one [`GpuParticle`] state record.
    pub offset: u64,
    /// Physical RHI format at this offset.
    pub format: VertexFormat,
}

/// Instance-rate attributes exposed by one packed particle-state buffer.
///
/// The caller supplies this stream after the static quad stream. Locations are
/// resolved by ordinary Shader reflection during preparation.
pub const GPU_PARTICLE_VERTEX_ATTRIBUTES: [GpuParticleVertexAttribute; 5] = [
    GpuParticleVertexAttribute {
        semantic: "PARTICLE_POSITION_AGE",
        offset: 0,
        format: VertexFormat::Float32(4),
    },
    GpuParticleVertexAttribute {
        semantic: "PARTICLE_VELOCITY_LIFETIME",
        offset: 16,
        format: VertexFormat::Float32(4),
    },
    GpuParticleVertexAttribute {
        semantic: "PARTICLE_SIZE_AXES_ANGLE",
        offset: 32,
        format: VertexFormat::Float32(4),
    },
    GpuParticleVertexAttribute {
        semantic: "PARTICLE_COLOR",
        offset: 48,
        format: VertexFormat::Float32(4),
    },
    GpuParticleVertexAttribute {
        semantic: "PARTICLE_ANGULAR_SPEED",
        offset: 64,
        format: VertexFormat::Float32(4),
    },
];

/// Immutable externally seeded particle state in Fluxel world coordinates.
///
/// `position` and `velocity` use the existing RHI world-space convention.
/// `lifetime` is strictly positive. `age` may equal or exceed it so a source
/// can explicitly seed an expired particle. `size` is the overall scale and
/// `axis_size` scales the billboard's local X and Y axes. `color` stays linear
/// and is faded by the render helper as the particle reaches its lifetime.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuParticle {
    /// Current world position.
    pub position: [f32; 3],
    /// Current world velocity.
    pub velocity: [f32; 3],
    /// Positive lifespan in seconds.
    pub lifetime: f32,
    /// Elapsed lifetime in seconds.
    pub age: f32,
    /// Overall billboard scale.
    pub size: f32,
    /// Per-axis billboard scale after the overall scale.
    pub axis_size: [f32; 2],
    /// Linear RGBA source color.
    pub color: [f32; 4],
    /// Billboard rotation in radians.
    pub angle: f32,
    /// Radians added to `angle` each second.
    pub angular_speed: f32,
}

/// Rejection reason for CPU particle state before allocation or upload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuParticleError {
    /// At least one input component is not finite.
    NonFinite,
    /// The lifetime is not strictly positive.
    NonPositiveLifetime,
    /// Age is negative.
    NegativeAge,
    /// Overall or axis scale is negative.
    NegativeSize,
}

impl GpuParticle {
    /// Checks the CPU source state before the renderer allocates GPU storage.
    pub fn validate(&self) -> Result<(), GpuParticleError> {
        if !self
            .position
            .iter()
            .chain(&self.velocity)
            .chain(&[self.lifetime, self.age, self.size])
            .chain(&self.axis_size)
            .chain(&self.color)
            .chain(&[self.angle, self.angular_speed])
            .all(|value| value.is_finite())
        {
            return Err(GpuParticleError::NonFinite);
        }
        if self.lifetime <= 0.0 {
            return Err(GpuParticleError::NonPositiveLifetime);
        }
        if self.age < 0.0 {
            return Err(GpuParticleError::NegativeAge);
        }
        if self.size < 0.0 || self.axis_size.iter().any(|value| *value < 0.0) {
            return Err(GpuParticleError::NegativeSize);
        }
        Ok(())
    }

    /// Packs five little-endian `vec4<f32>` values in the published ABI order.
    pub fn pack(&self) -> Result<[u8; GPU_PARTICLE_STATE_STRIDE as usize], GpuParticleError> {
        self.validate()?;
        let vectors = [
            [
                self.position[0],
                self.position[1],
                self.position[2],
                self.age,
            ],
            [
                self.velocity[0],
                self.velocity[1],
                self.velocity[2],
                self.lifetime,
            ],
            [self.size, self.axis_size[0], self.axis_size[1], self.angle],
            self.color,
            [self.angular_speed, 0.0, 0.0, 0.0],
        ];
        let mut bytes = [0; GPU_PARTICLE_STATE_STRIDE as usize];
        for (vector_index, vector) in vectors.into_iter().enumerate() {
            for (component_index, component) in vector.into_iter().enumerate() {
                let offset = vector_index * 16 + component_index * 4;
                bytes[offset..offset + 4].copy_from_slice(&component.to_le_bytes());
            }
        }
        Ok(bytes)
    }
}

/// Uniform values used by one bounded particle update dispatch.
///
/// The owner packs these as a single 32-byte uniform record. `active_count`
/// restricts the workgroup invocation range. `delta_seconds` and `gravity`
/// must be finite and `delta_seconds` cannot be negative.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuParticleUpdate {
    /// Number of initialized state records to update.
    pub active_count: u32,
    /// Elapsed time for this update.
    pub delta_seconds: f32,
    /// World-space acceleration applied after position integration.
    pub gravity: [f32; 3],
}

/// Rejection reason for update uniforms before dispatch allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuParticleUpdateError {
    /// Delta time or an acceleration component is not finite.
    NonFinite,
    /// Delta time is negative.
    NegativeDelta,
}

impl GpuParticleUpdate {
    /// Checks the update values before they are made visible to the GPU.
    pub fn validate(&self) -> Result<(), GpuParticleUpdateError> {
        if !self.delta_seconds.is_finite() || !self.gravity.iter().all(|value| value.is_finite()) {
            return Err(GpuParticleUpdateError::NonFinite);
        }
        if self.delta_seconds < 0.0 {
            return Err(GpuParticleUpdateError::NegativeDelta);
        }
        Ok(())
    }

    /// Packs the exact 32-byte WGSL uniform layout in little-endian order.
    ///
    /// Bytes zero through three hold `active_count`. Bytes four through fifteen
    /// are zero padding. Bytes sixteen through nineteen hold `delta_seconds`,
    /// followed by `gravity.x`, `gravity.y`, and `gravity.z`.
    pub fn pack(&self) -> Result<[u8; 32], GpuParticleUpdateError> {
        self.validate()?;
        let mut bytes = [0; 32];
        bytes[0..4].copy_from_slice(&self.active_count.to_le_bytes());
        bytes[16..20].copy_from_slice(&self.delta_seconds.to_le_bytes());
        for (index, value) in self.gravity.into_iter().enumerate() {
            let offset = 20 + index * 4;
            bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        Ok(bytes)
    }
}

/// Shared WGSL particle-state declarations.
///
/// Compose this source unit before either [`PARTICLE_UPDATE_WGSL`] or
/// [`PARTICLE_BILLBOARD_WGSL`]. Keeping the declaration separate lets compute
/// and raster pipelines consume the same storage and vertex ABI without each
/// defining an incompatible particle type.
pub(crate) const PARTICLE_STATE_WGSL: &str = r#"
struct FluxelParticle {
    position_age: vec4<f32>,
    velocity_lifetime: vec4<f32>,
    size_axes_angle: vec4<f32>,
    color: vec4<f32>,
    angular_speed: vec4<f32>,
};

struct FluxelParticleStates {
    particles: array<FluxelParticle>,
};
"#;

/// WGSL compute source for the M1 particle update.
///
/// Compose [`PARTICLE_STATE_WGSL`] before this source. Bind group zero binds a
/// 32-byte `FluxelParticleUpdate` uniform at binding zero, the current state
/// as read-only storage at binding one, and the next ping-pong state as
/// read-write storage at binding two. The 64-wide dispatch and source-first
/// position integration follow Babylon revision
/// `daae8d9483dcbb6308ebd0f6ca5b218783614771`,
/// `gpuUpdateParticles.compute.fx`. Gradient, emitter, noise and flow-map
/// branches are deliberately outside this M1 kernel.
pub(crate) const PARTICLE_UPDATE_WGSL: &str = r#"
struct FluxelParticleUpdate {
    count_padding: vec4<u32>,
    delta_gravity: vec4<f32>,
};

@group(0) @binding(0) var<uniform> particle_update: FluxelParticleUpdate;
@group(0) @binding(1) var<storage, read> particle_source: FluxelParticleStates;
@group(0) @binding(2) var<storage, read_write> particle_product: FluxelParticleStates;

@compute @workgroup_size(64)
fn fluxel_particle_update(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    if (index >= particle_update.count_padding.x) {
        return;
    }
    let source = particle_source.particles[index];
    var product = source;
    let age = source.position_age.w;
    let lifetime = source.velocity_lifetime.w;
    let delta = particle_update.delta_gravity.x;
    let gravity = particle_update.delta_gravity.yzw;
    product.position_age = vec4<f32>(
        source.position_age.xyz + source.velocity_lifetime.xyz * delta,
        age + delta,
    );
    product.velocity_lifetime = vec4<f32>(
        source.velocity_lifetime.xyz + gravity * delta,
        lifetime,
    );
    product.size_axes_angle = vec4<f32>(
        source.size_axes_angle.xyz,
        source.size_axes_angle.w + source.angular_speed.x * delta,
    );
    particle_product.particles[index] = product;
}
"#;

/// WGSL helpers for the default spherical billboard branch.
///
/// Compose [`PARTICLE_STATE_WGSL`] before this source. The render template
/// passes a unit-quad corner and a caller-selected view and projection. The
/// helpers produce world and view positions only. The caller performs Fluxel's
/// established clip projection and supplies the final transparent-phase
/// composition. A source lifetime at or below zero, or an age at or above the
/// lifetime, returns `inactive`. The supplied `dead_color` is mixed using
/// Babylon's default particle color branch. Babylon's base default is opaque
/// black, while a transparent phase can explicitly select a transparent dead
/// color.
pub(crate) const PARTICLE_BILLBOARD_WGSL: &str = r#"
struct FluxelParticleBillboard {
    world_position: vec3<f32>,
    view_position: vec4<f32>,
    color: vec4<f32>,
    inactive: bool,
};

fn fluxel_particle_billboard(
    particle: FluxelParticle,
    corner: vec2<f32>,
    translation_pivot: vec2<f32>,
    dead_color: vec4<f32>,
    view: mat4x4<f32>,
    inverse_view: mat4x4<f32>,
) -> FluxelParticleBillboard {
    let age = particle.position_age.w;
    let lifetime = particle.velocity_lifetime.w;
    let inactive = lifetime <= 0.0 || age >= lifetime;
    let lifetime_ratio = clamp(age / select(1.0, lifetime, lifetime > 0.0), 0.0, 1.0);
    let size = particle.size_axes_angle;
    let local = (corner - translation_pivot) * size.yz * size.x;
    let angle = particle.size_axes_angle.w;
    let rotated = vec4<f32>(
        local.x * cos(angle) - local.y * sin(angle) + translation_pivot.x,
        local.x * sin(angle) + local.y * cos(angle) + translation_pivot.y,
        0.0,
        0.0,
    );
    let center = view * vec4<f32>(particle.position_age.xyz, 1.0);
    let view_position = center + rotated;
    return FluxelParticleBillboard(
        (inverse_view * view_position).xyz,
        view_position,
        particle.color * (1.0 - lifetime_ratio) + dead_color * lifetime_ratio,
        inactive,
    );
}
"#;
