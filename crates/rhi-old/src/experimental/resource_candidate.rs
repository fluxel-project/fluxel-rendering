//! Browser-independent construction ownership for closed resource recipes.
//!
//! The closed browser resource recipes create several live GPU objects before
//! the whole set is usable. Construction ownership must therefore be total at
//! every step: a candidate owns exactly what was created so far, destroys all
//! of it when any later creation or write step fails, and releases the set to
//! its registry only after the final fallible step, so a failed construction
//! can never leave destroyed objects behind or half-update a registry.
//! Like `webgpu_state`, this module is portable on purpose: host tests inject
//! failures at any creation or write step without pretending a host build has
//! a WebGPU implementation. It owns no lifecycle states, tickets, registries,
//! or browser FFI.

/// Total cleanup ownership of the values one recipe created so far.
///
/// Register every successfully created value with [`Self::keep`] immediately
/// after its creation call returns. Any `?` failure in a later step drops the
/// candidate, which runs `dispose` on each kept value exactly once. [`Self::commit`]
/// ends that ownership after the last fallible step; the fully initialized
/// set then moves to its registry owner and is never disposed by the candidate.
pub(crate) struct CreationCandidate<T, F: FnMut(&T)> {
    kept: Vec<T>,
    dispose: F,
    committed: bool,
}

impl<T, F: FnMut(&T)> CreationCandidate<T, F> {
    /// Creates an uncommitted candidate around one cleanup operation.
    pub(crate) fn new(dispose: F) -> Self {
        Self {
            kept: Vec::new(),
            dispose,
            committed: false,
        }
    }

    /// Registers one created value. The recipe keeps its own handle for later
    /// steps; this copy exists only so a later failure can destroy the object.
    pub(crate) fn keep(&mut self, value: &T)
    where
        T: Clone,
    {
        self.kept.push(value.clone());
    }

    /// Ends cleanup ownership after the last fallible construction step.
    pub(crate) fn commit(mut self) {
        self.committed = true;
    }
}

impl<T, F: FnMut(&T)> Drop for CreationCandidate<T, F> {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        for value in &self.kept {
            (self.dispose)(value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::RefCell, collections::HashMap, rc::Rc};

    /// Counting no-GPU backend. Every creation and write is one numbered step;
    /// `fail_creation`/`fail_write` (1-based step numbers) inject the review
    /// seam's failures. `None` never fails.
    #[derive(Default)]
    struct Backend {
        creations: u32,
        writes: u32,
        fail_creation: Option<u32>,
        fail_write: Option<u32>,
    }

    impl Backend {
        fn create(&mut self) -> Result<u64, &'static str> {
            self.creations += 1;
            if self.fail_creation == Some(self.creations) {
                return Err("injected creation failure");
            }
            Ok(u64::from(self.creations))
        }

        fn write(&mut self, _value: &u64) -> Result<(), &'static str> {
            self.writes += 1;
            if self.fail_write == Some(self.writes) {
                return Err("injected write failure");
            }
            Ok(())
        }
    }

    type Registry = HashMap<u64, (u64, u64)>;

    /// A registry that already owns an unrelated entry. A failed upload must
    /// leave it exactly as it was: no half update, no lost entry.
    fn registry_with_unrelated_entry() -> Registry {
        HashMap::from([(999, (10, 11))])
    }

    fn destroy_log() -> Rc<RefCell<Vec<u64>>> {
        Rc::new(RefCell::new(Vec::new()))
    }

    /// Mirrors the closed WebGPU resident-mesh recipe ordering: create the
    /// position buffer, create the index buffer, write the position upload,
    /// write the index upload, and only then install into the registry.
    fn upload_mesh(
        backend: &mut Backend,
        destroyed: &Rc<RefCell<Vec<u64>>>,
        registry: &mut Registry,
        key: u64,
    ) -> Result<(), &'static str> {
        let log = Rc::clone(destroyed);
        let mut candidate =
            CreationCandidate::new(move |value: &u64| log.borrow_mut().push(*value));
        let position = backend.create()?;
        candidate.keep(&position);
        let index = backend.create()?;
        candidate.keep(&index);
        backend.write(&position)?;
        backend.write(&index)?;
        candidate.commit();
        registry.insert(key, (position, index));
        Ok(())
    }

    /// Mirrors the resident per-frame recipe ordering: create the uniform
    /// buffer, create its bind group, then install the pair.
    fn create_uniform_and_binding(
        backend: &mut Backend,
        destroyed: &Rc<RefCell<Vec<u64>>>,
    ) -> Result<(u64, u64), &'static str> {
        let log = Rc::clone(destroyed);
        let mut candidate =
            CreationCandidate::new(move |value: &u64| log.borrow_mut().push(*value));
        let uniform = backend.create()?;
        candidate.keep(&uniform);
        let bind_group = backend.create()?;
        candidate.commit();
        Ok((uniform, bind_group))
    }

    fn destroyed_values(destroyed: &Rc<RefCell<Vec<u64>>>) -> Vec<u64> {
        destroyed.borrow().clone()
    }

    #[test]
    fn first_creation_failure_creates_and_destroys_nothing() {
        let mut backend = Backend {
            fail_creation: Some(1),
            ..Backend::default()
        };
        let destroyed = destroy_log();
        let mut registry = registry_with_unrelated_entry();
        assert!(upload_mesh(&mut backend, &destroyed, &mut registry, 7).is_err());
        assert_eq!(destroyed_values(&destroyed), Vec::<u64>::new());
        assert_eq!(registry, registry_with_unrelated_entry());
    }

    #[test]
    fn second_creation_failure_destroys_exactly_the_first_object() {
        let mut backend = Backend {
            fail_creation: Some(2),
            ..Backend::default()
        };
        let destroyed = destroy_log();
        let mut registry = registry_with_unrelated_entry();
        assert!(upload_mesh(&mut backend, &destroyed, &mut registry, 7).is_err());
        assert_eq!(destroyed_values(&destroyed), vec![1]);
        assert_eq!(registry, registry_with_unrelated_entry());
    }

    #[test]
    fn first_write_failure_destroys_every_created_object() {
        let mut backend = Backend {
            fail_write: Some(1),
            ..Backend::default()
        };
        let destroyed = destroy_log();
        let mut registry = registry_with_unrelated_entry();
        assert!(upload_mesh(&mut backend, &destroyed, &mut registry, 7).is_err());
        assert_eq!(destroyed_values(&destroyed), vec![1, 2]);
        assert_eq!(registry, registry_with_unrelated_entry());
    }

    #[test]
    fn second_write_failure_destroys_every_created_object() {
        let mut backend = Backend {
            fail_write: Some(2),
            ..Backend::default()
        };
        let destroyed = destroy_log();
        let mut registry = registry_with_unrelated_entry();
        assert!(upload_mesh(&mut backend, &destroyed, &mut registry, 7).is_err());
        assert_eq!(destroyed_values(&destroyed), vec![1, 2]);
        assert_eq!(registry, registry_with_unrelated_entry());
    }

    #[test]
    fn successful_upload_installs_once_and_destroys_nothing() {
        let mut backend = Backend::default();
        let destroyed = destroy_log();
        let mut registry = registry_with_unrelated_entry();
        upload_mesh(&mut backend, &destroyed, &mut registry, 7).expect("upload succeeds");
        assert_eq!(destroyed_values(&destroyed), Vec::<u64>::new());
        assert_eq!(registry.get(&7), Some(&(1, 2)));
        assert_eq!(registry.get(&999), Some(&(10, 11)));
    }

    #[test]
    fn committed_candidates_never_dispose_their_values() {
        let mut backend = Backend::default();
        let destroyed = destroy_log();
        create_uniform_and_binding(&mut backend, &destroyed)
            .expect("full uniform construction succeeds");
        assert_eq!(destroyed_values(&destroyed), Vec::<u64>::new());
        assert_eq!(backend.creations, 2);
    }

    #[test]
    fn uniform_bind_group_failure_destroys_only_the_uniform() {
        let mut backend = Backend {
            fail_creation: Some(2),
            ..Backend::default()
        };
        let destroyed = destroy_log();
        assert!(create_uniform_and_binding(&mut backend, &destroyed).is_err());
        assert_eq!(destroyed_values(&destroyed), vec![1]);
    }
}
