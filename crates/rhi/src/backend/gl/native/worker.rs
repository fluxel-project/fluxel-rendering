//! Dedicated owner-thread scheduler for WGL/EGL native contexts.
//!
//! The context descriptor crosses the thread boundary; the context and its
//! `NativeGlProvider` are constructed *inside* the worker and never leave it.
//! This is the only shape compatible with both GL currentness and the public
//! driver's `Send + Sync` requirement.

use std::sync::Mutex;
use std::sync::mpsc::{self, Sender};

/// Failure while constructing an owner-thread worker.
#[derive(Debug)]
pub(crate) enum WorkerStartupError {
    Factory(String),
    WorkerExited,
}

type Job<O> = Box<dyn FnOnce(&mut O) + Send + 'static>;

/// Synchronous command gateway to an object confined to one dedicated thread.
///
/// `O` does not need `Send`: the factory creates it after the thread starts.
/// A caller's closure and result are `Send`, which prevents references to the
/// owner-context GL objects from escaping in either direction.
pub(crate) struct NativeOwnerWorker<O: 'static> {
    sender: Mutex<Option<Sender<Job<O>>>>,
}

impl<O: 'static> NativeOwnerWorker<O> {
    /// Starts an owner and returns immutable discovery information produced on
    /// that same owner thread.  Native GL facts/name must be observed only
    /// after the WGL/EGL context is current; this avoids moving a `!Send`
    /// context merely to discover it on the caller thread.
    pub(crate) fn spawn_with_info<I: Send + 'static>(
        factory: impl FnOnce() -> Result<(O, I), String> + Send + 'static,
    ) -> Result<(Self, I), WorkerStartupError> {
        let (jobs, receiver) = mpsc::channel::<Job<O>>();
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        std::thread::Builder::new()
            .name("fluxel-gl-owner".into())
            .spawn(move || match factory() {
                Ok((mut owner, info)) => {
                    let _ = ready_tx.send(Ok(info));
                    while let Ok(job) = receiver.recv() {
                        job(&mut owner);
                    }
                }
                Err(message) => {
                    let _ = ready_tx.send(Err(message));
                }
            })
            .map_err(|_| WorkerStartupError::WorkerExited)?;
        match ready_rx
            .recv()
            .map_err(|_| WorkerStartupError::WorkerExited)?
        {
            Ok(info) => Ok((
                Self {
                    sender: Mutex::new(Some(jobs)),
                },
                info,
            )),
            Err(message) => Err(WorkerStartupError::Factory(message)),
        }
    }

    /// Starts a thread and constructs its WGL/EGL context/provider there.
    ///
    /// The factory captures only a verified, owned platform descriptor (for
    /// example HWND/HDC values normalized to integer descriptors or an EGL
    /// display/config descriptor). It must not capture an already-current
    /// `glow::Context`, which would move context affinity across threads.
    pub(crate) fn spawn(
        factory: impl FnOnce() -> Result<O, String> + Send + 'static,
    ) -> Result<Self, WorkerStartupError> {
        let (jobs, receiver) = mpsc::channel::<Job<O>>();
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        std::thread::Builder::new()
            .name("fluxel-gl-owner".into())
            .spawn(move || match factory() {
                Ok(mut owner) => {
                    let _ = ready_tx.send(Ok(()));
                    while let Ok(job) = receiver.recv() {
                        job(&mut owner);
                    }
                }
                Err(message) => {
                    let _ = ready_tx.send(Err(message));
                }
            })
            .map_err(|_| WorkerStartupError::WorkerExited)?;
        match ready_rx
            .recv()
            .map_err(|_| WorkerStartupError::WorkerExited)?
        {
            Ok(()) => Ok(Self {
                sender: Mutex::new(Some(jobs)),
            }),
            Err(message) => Err(WorkerStartupError::Factory(message)),
        }
    }

    /// Runs one owned command on the context owner and waits for its result.
    /// A disconnected worker is terminal rather than retried on another
    /// thread: retrying would issue GL calls against a different context.
    pub(crate) fn call<R: Send + 'static>(
        &self,
        command: impl FnOnce(&mut O) -> R + Send + 'static,
    ) -> Result<R, WorkerStartupError> {
        let (reply_tx, reply_rx) = mpsc::sync_channel(1);
        let job: Job<O> = Box::new(move |owner| {
            let _ = reply_tx.send(command(owner));
        });
        let sender = self.sender.lock().unwrap_or_else(|p| p.into_inner());
        let Some(sender) = sender.as_ref() else {
            return Err(WorkerStartupError::WorkerExited);
        };
        sender
            .send(job)
            .map_err(|_| WorkerStartupError::WorkerExited)?;
        reply_rx
            .recv()
            .map_err(|_| WorkerStartupError::WorkerExited)
    }
}

impl<O: 'static> Drop for NativeOwnerWorker<O> {
    fn drop(&mut self) {
        self.sender.lock().unwrap_or_else(|p| p.into_inner()).take();
    }
}

#[cfg(test)]
mod tests {
    use super::NativeOwnerWorker;

    #[test]
    fn owner_is_created_and_mutated_only_on_worker() {
        let worker = NativeOwnerWorker::spawn(|| Ok::<_, String>(0u32)).unwrap();
        assert_eq!(
            worker
                .call(|value| {
                    *value += 1;
                    *value
                })
                .unwrap(),
            1
        );
        assert_eq!(
            worker
                .call(|value| {
                    *value += 1;
                    *value
                })
                .unwrap(),
            2
        );
    }
}
