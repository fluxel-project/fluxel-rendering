//! Render-only records and transactional scene changes.
//! The scene owns no gameplay identity, ECS query, GPU object, or Graph work.

mod camera;
pub use camera::ShadingCamera;
mod types;
pub use types::*;
