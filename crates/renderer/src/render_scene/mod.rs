//! Render-only records and transactional scene changes.
//! The scene owns no gameplay identity, ECS query, GPU object, or Graph work.

mod camera;
pub use camera::ShadingCamera;
mod directional_light;
pub use directional_light::DirectionalLight;
mod types;
pub use types::*;
