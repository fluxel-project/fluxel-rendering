//! Fluxel RHI — the portable GPU execution contract.
//!
//! This crate is the boundary between Fluxel's engine layers and the GPU APIs
//! beneath them. It exposes one portable vocabulary for device ownership,
//! resources, shaders, recording, submission, completion, and presentation, and
//! it owns the rules that decide whether a portable operation is legal without
//! asking a driver.
//!
//! # What this crate is
//!
//! ```text
//! portable GPU execution vocabulary
//! + instance capability facts
//! + strict validation
//! + opaque logical handles
//! + explicit hazard/dependency validation
//! + logical submission/completion/presentation
//! + portable logical statistics / inventory
//! ```
//!
//! # What this crate is not
//!
//! ```text
//! a wrapper over one native API
//! the greatest common denominator of all platforms
//! a native-handle escape hatch
//! ```
//!
//! It is capability-layered rather than levelled down: a backend contributes the
//! facts about what it can do, and the portable vocabulary stays whole. A caller
//! therefore never learns whether it is on DX12, Vulkan, Metal, WebGPU, or the
//! GL family in order to write correct code — but a caller that needs a
//! capability asks the device for the fact rather than probing for a type.
//!
//! # Layering
//!
//! ```text
//! Renderer / material graph / custom pipeline policy
//!     -> RenderGraph declaration and object recipes
//!     -> GraphExecutionPlan
//!     -> RHI RecordedWork + SubmissionPlan
//!     -> backend-private lowering
//!     -> DX12 | Vulkan | Metal | WebGPU | GL family
//! ```
//!
//! RenderGraph owns declarations, versions, dependencies, culling, scheduling,
//! logical lifetime, and presentation intent. This crate owns portable
//! execution, device validation, submission, completion, presentation,
//! retirement, logical observation, and backend lowering. Keeping those apart is
//! what stops the graph from becoming a second source of truth about hazards.
//!
//! # Where the rules live
//!
//! The normative sources are `documents/design-rhi.md` and the numbered modules
//! under `documents/rhi-design/`. This crate is written from them, not the other
//! way round: a disagreement between this code and those modules is a defect in
//! this code.
//!
//! # Status
//!
//! The crate is being rebuilt contract-first. [`api`] holds the public surface,
//! written from the specification with its validation and refusal paths fixed
//! before any lowering exists; a verb whose body is `unimplemented!()` has its
//! contract settled and its implementation still to arrive. Backends land under
//! `backend/`, and the shared implementation they draw on under `base/`.

#![deny(missing_docs)]

pub mod api;
