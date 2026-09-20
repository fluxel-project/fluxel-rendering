//! Vulkan command recording and submission.
//!
//! This chapter is intentionally a narrow vertical slice.  It owns the native
//! queue, command-pool and completion bookkeeping, but it does not pretend that
//! a recorded portable payload has been lowered before a corresponding Vulkan
//! lowering exists.  `spine` therefore refuses every currently-unlowered
//! payload during Phase A.  That is important: accepting a plan and omitting a
//! draw, copy, upload, or debug command would be a silent substitute.
//!
//! Future `copy`, `raster`, `compute`, and `transfer` siblings replace the
//! individual refusal arms; they do not change the transaction boundary here.

mod compute;
mod raster;
pub(crate) mod spine;
mod transfer;
