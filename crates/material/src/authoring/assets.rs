//! Material asset identities.

use fluxel_assets::{AssetId, AssetKind, AssetSnapshot, ContentGeneration};

/// Marker for material assets stored through `fluxel-assets`.
pub struct MaterialAsset;

impl AssetKind for MaterialAsset {}

/// Opaque logical identity for one material asset.
pub type MaterialAssetId = AssetId<MaterialAsset>;

/// The durable identity of one material content generation.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct MaterialAssetRef {
    id: MaterialAssetId,
    generation: ContentGeneration,
}

impl MaterialAssetRef {
    /// Captures identity and content generation from an immutable asset snapshot.
    #[must_use]
    pub fn from_snapshot<V>(snapshot: &AssetSnapshot<MaterialAsset, V>) -> Self {
        Self {
            id: snapshot.id(),
            generation: snapshot.generation(),
        }
    }

    /// Returns the logical asset identity.
    #[must_use]
    pub const fn id(self) -> MaterialAssetId {
        self.id
    }

    /// Returns the immutable content generation represented by this reference.
    #[must_use]
    pub const fn generation(self) -> ContentGeneration {
        self.generation
    }
}
