import numpy as np

FEATURE_DIM = 60

def extract_features(patch: np.ndarray) -> np.ndarray:
    patch = np.asarray(patch, dtype=np.float32)
    if patch.shape != (15, 35, 35):
        raise ValueError(f"Expected patch shape (15,35,35), got {patch.shape}")

    spectral = patch[:12]
    spec_mean = spectral.mean(axis=(1, 2))
    spec_std  = spectral.std(axis=(1, 2))
    spec_min  = spectral.min(axis=(1, 2))
    spec_max  = spectral.max(axis=(1, 2))

    terrain = patch[12:15]
    terr_mean = terrain.mean(axis=(1, 2))
    terr_std  = terrain.std(axis=(1, 2))
    terr_min  = terrain.min(axis=(1, 2))
    terr_max  = terrain.max(axis=(1, 2))

    return np.concatenate(
        [spec_mean, spec_std, spec_min, spec_max,
         terr_mean, terr_std, terr_min, terr_max]
    ).astype(np.float32)

def extract_features_batch(patches: np.ndarray) -> np.ndarray:
    patches = np.asarray(patches, dtype=np.float32)
    if patches.ndim != 4 or patches.shape[1:] != (15, 35, 35):
        raise ValueError(f"Expected patches shape (N,15,35,35), got {patches.shape}")

    spectral = patches[:, :12]
    spec_mean = spectral.mean(axis=(2, 3))
    spec_std  = spectral.std(axis=(2, 3))
    spec_min  = spectral.min(axis=(2, 3))
    spec_max  = spectral.max(axis=(2, 3))

    terrain = patches[:, 12:15]
    terr_mean = terrain.mean(axis=(2, 3))
    terr_std  = terrain.std(axis=(2, 3))
    terr_min  = terrain.min(axis=(2, 3))
    terr_max  = terrain.max(axis=(2, 3))

    feats = np.concatenate(
        [spec_mean, spec_std, spec_min, spec_max,
         terr_mean, terr_std, terr_min, terr_max],
        axis=1
    )
    return feats.astype(np.float32)
