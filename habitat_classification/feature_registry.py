import os

def get_feature_fns():
    v = os.getenv("HABITAT_FEATURES", "v36").strip().lower()
    if v == "v36":
        from features_v36 import extract_features, extract_features_batch, FEATURE_DIM
        return extract_features, extract_features_batch, FEATURE_DIM
    if v == "v60":
        from features_v60 import extract_features, extract_features_batch, FEATURE_DIM
        return extract_features, extract_features_batch, FEATURE_DIM
    raise ValueError(f"Unknown HABITAT_FEATURES={v}. Use v36 or v60.")
