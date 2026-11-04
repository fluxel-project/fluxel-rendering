//! Deterministic ownership tests for renderer submissions.
//!
//! The witness backend reports scripted logical facts only; DX12 and WGL
//! evidence belongs to the integration consumer.

use std::{
    future::Future,
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use fluxel_assets::{Acquire, AssetSnapshot, AssetStore, ResidentBytes};
use fluxel_rhi::{api::*, backend::*};

use super::*;

#[path = "lifecycle/backend.rs"]
mod backend;
#[path = "lifecycle/execution.rs"]
mod execution;
#[path = "lifecycle/support.rs"]
mod support;
