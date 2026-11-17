//! Structural authorization for same-Graph Deferred material aliases.

use super::{record_view::DeferredRecordView, *};
use crate::graph::definition::TextureHandle;

/// Native product identity is checked by the combined capture owner before
/// recording. This check preserves logical roles and prevents an asset import
/// or CPU upload from being silently replaced by a capture dependency.
pub(super) fn validate<B: ResourceBackend>(
    frame: &DeferredRecordView<'_, B>,
    aliases: Option<&[Option<TextureHandle>]>,
) -> Result<()> {
    let count = frame.resident.textures.len();
    if aliases.is_some_and(|items| items.len() != count) {
        return Err(Error::InvalidRange);
    }
    let mut roles = vec![None; count];
    for draw in frame.draws {
        if draw.textures.len() != draw.capture_textures.len() {
            return Err(Error::InvalidRange);
        }
        for (&index, &capture) in draw.textures.iter().zip(&draw.capture_textures) {
            let role = roles.get_mut(index).ok_or(Error::InvalidRange)?;
            if role.is_some_and(|previous| previous != capture) {
                return Err(Error::InvalidState);
            }
            *role = Some(capture);
        }
    }
    for (index, role) in roles.into_iter().enumerate() {
        let capture = role == Some(true);
        let alias = aliases.and_then(|items| items[index]);
        if capture != alias.is_some()
            || (capture
                && frame
                    .resident_upload
                    .textures
                    .iter()
                    .any(|copy| copy.texture == index))
        {
            return Err(Error::InvalidState);
        }
    }
    Ok(())
}
