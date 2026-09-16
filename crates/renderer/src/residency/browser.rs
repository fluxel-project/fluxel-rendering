//! WebGL2 realization of the closed logical-asset residency table.
//!
//! This is the browser counterpart of [`super::native`]: the same typed
//! logical keys, the same accept/advance/retire rules, driving the RHI's
//! explicit WebGL2 executor instead of a native device. What differs is only
//! the shape of an upload -- see [`Decided`] -- and the fact that the executor
//! is reached through a handle the caller also submits frames with, because a
//! browser context is one object both halves must name.
//!
//! The physical buffers are not here. They stay inside the RHI's resident
//! tokens, which this layer keys and hands back; nothing in this module holds
//! a WebGL handle, and nothing here invents a logical identity -- every key
//! comes from an [`AssetSnapshot`] the asset store minted.

use core::fmt;
use std::cell::RefCell;
use std::rc::Rc;

use fluxel_assets::AssetSnapshot;
use fluxel_rendergraph::DeviceIdentity;
use fluxel_rhi::adapter::webgl2::{
    WebGl2ResidentImage, WebGl2ResidentMesh, WebGl2Session, WebGl2SessionError,
};

use crate::{Geometry, Rgba8Image};

use super::MeshAsset;
use super::cache::{MeshStartError, MeshStatus, ResidencyBackend, Table, UploadPoll};

/// The browser executor, reached through the handle the frame path also holds.
struct WebGl2Backend {
    session: Rc<RefCell<WebGl2Session>>,
}

impl WebGl2Backend {
    fn new(session: Rc<RefCell<WebGl2Session>>) -> Self {
        Self { session }
    }
}

/// One browser upload whose outcome was already decided when it started.
///
/// The native backend's uploads are genuinely asynchronous: it starts a copy
/// and learns later whether that copy completed. The WebGL2 executor has no
/// such stage. `upload_resident_mesh` creates and fills the buffers inside the
/// call, and every reason it can report for not doing so is known before it
/// has retained anything -- which is why this backend's failure type is
/// uninhabited and its start error carries the whole of what can go wrong.
///
/// The type exists so that the browser half runs the table's own accept,
/// advance and retire rules rather than a second copy of them.
struct Decided<T> {
    token: T,
}

/// The failure a browser upload cannot report, because it has no stage after
/// the start in which to report one.
///
/// A future browser executor with a genuinely asynchronous copy would give
/// this type a variant along with the state that can produce it; until then
/// the table's failure path is unreachable here by construction rather than by
/// convention. `clippy::empty_enum` would prefer `!`, which is not yet
/// nameable in an associated type position, so the lint stays allow-by-default
/// rather than being silenced here.
#[derive(Clone, Debug)]
enum NeverFailed {}

impl ResidencyBackend for WebGl2Backend {
    type MeshPending = Decided<WebGl2ResidentMesh>;
    type MeshResident = WebGl2ResidentMesh;
    type MeshStartError = WebGl2SessionError;
    type MeshFailure = NeverFailed;
    type ImagePending = Decided<WebGl2ResidentImage>;
    type ImageResident = WebGl2ResidentImage;
    type ImageStartError = WebGl2SessionError;
    type ImageFailure = NeverFailed;

    fn device(&self) -> DeviceIdentity {
        self.session.borrow().device_identity()
    }

    fn begin_mesh(
        &mut self,
        geometry: &Geometry,
    ) -> Result<Decided<WebGl2ResidentMesh>, WebGl2SessionError> {
        let mut session = self.session.borrow_mut();
        session
            .upload_resident_mesh(geometry.positions(), geometry.indices())
            .map(|token| Decided { token })
    }

    fn poll_mesh(
        &mut self,
        pending: &mut Decided<WebGl2ResidentMesh>,
    ) -> UploadPoll<WebGl2ResidentMesh, NeverFailed> {
        // Idempotent on purpose: a resident token is a reference-counted
        // handle, so answering twice costs a refcount, and the state machine
        // never has to know that this backend resolves on the first poll.
        UploadPoll::Ready(pending.token.clone())
    }

    fn begin_image(
        &mut self,
        image: &Rgba8Image,
    ) -> Result<Decided<WebGl2ResidentImage>, WebGl2SessionError> {
        let mut session = self.session.borrow_mut();
        session
            .upload_resident_image(image.extent(), image.pixels())
            .map(|token| Decided { token })
    }

    fn poll_image(
        &mut self,
        pending: &mut Decided<WebGl2ResidentImage>,
    ) -> UploadPoll<WebGl2ResidentImage, NeverFailed> {
        UploadPoll::Ready(pending.token.clone())
    }
}

/// Why a browser mesh realization could not be produced.
#[derive(Debug)]
#[non_exhaustive]
pub enum WebGl2ResidencyError {
    /// The context refused to start the upload.
    ///
    /// Every reason an upload does not become resident arrives here: the
    /// session was not active, the geometry was rejected, or the browser
    /// failed while creating or filling the buffers.
    UploadRejected(WebGl2SessionError),
    /// The snapshot predates the newest content generation already observed
    /// for the same logical mesh.
    StaleMeshGeneration,
    /// The shared residency state machine did not reach a committed
    /// realization.
    ///
    /// The WebGL2 executor cannot enter this state, because it decides an
    /// upload inside the start call and reports a failure before retaining
    /// any work. The variant exists so that a change on either side of that
    /// boundary fails closed -- as a reported error -- instead of dropping a
    /// realization the caller believes it owns.
    NotResident,
}

impl fmt::Display for WebGl2ResidencyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UploadRejected(error) => {
                write!(formatter, "browser mesh residency did not start: {error}")
            }
            Self::StaleMeshGeneration => {
                formatter.write_str("browser residency rejected a stale content generation")
            }
            Self::NotResident => formatter.write_str("browser mesh residency is not resident"),
        }
    }
}

impl std::error::Error for WebGl2ResidencyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::UploadRejected(error) => Some(error),
            Self::StaleMeshGeneration | Self::NotResident => None,
        }
    }
}

/// The renderer's residency for one explicit browser context.
///
/// Resolution is synchronous here in a way it is not on the native path: the
/// caller asks with a logical generation and gets back either the resident
/// token for that exact generation or an error saying why not. The rules
/// behind that answer are the shared table's -- one upload per logical
/// identity, content generation and device; a newer generation replaces the
/// older realization rather than joining it; a distinct device is a distinct
/// key, so a restored context re-uploads instead of inheriting tokens whose
/// buffers are gone.
///
/// The context handle is shared with the caller that submits frames. Each side
/// borrows it for the length of one call and never across one.
pub struct WebGl2AssetResidency {
    table: Table<WebGl2Backend>,
}

impl WebGl2AssetResidency {
    /// Creates residency for one browser context.
    ///
    /// A restored context is a new device, so the caller replaces this value
    /// rather than reusing it: every key it holds names the context that was
    /// lost, and none of them can be matched again.
    #[must_use]
    pub fn new(session: Rc<RefCell<WebGl2Session>>) -> Self {
        let table = Table::new(WebGl2Backend::new(session));
        Self { table }
    }

    /// Returns the context identity every realization is keyed on.
    #[must_use]
    pub fn device(&self) -> DeviceIdentity {
        self.table.device()
    }

    /// Resolves the exact mesh generation on this context, uploading it once.
    ///
    /// Repeated calls with the same snapshot return the token already resident
    /// for that logical identity, generation and device rather than uploading
    /// again, which is the whole of what the RHI's unconditional upload verb
    /// leaves to its caller.
    pub fn prepare_mesh(
        &mut self,
        mesh: &AssetSnapshot<MeshAsset, Geometry>,
    ) -> Result<WebGl2ResidentMesh, WebGl2ResidencyError> {
        match self.table.prepare_mesh(mesh.clone()) {
            Ok(MeshStatus::Ready(resident)) => Ok(resident),
            Ok(MeshStatus::Pending) => Err(WebGl2ResidencyError::NotResident),
            Ok(MeshStatus::Failed(impossible)) => match impossible {},
            Err(MeshStartError::Upload(error)) => Err(WebGl2ResidencyError::UploadRejected(error)),
            Err(MeshStartError::Stale) => Err(WebGl2ResidencyError::StaleMeshGeneration),
        }
    }
}
