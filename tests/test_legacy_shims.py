from feature_engineering import FeatureEngineeringTransformer, FeaturePruner
from preprocessing import (
    CategoricalLevelManager,
    FrequencyEncoder,
    MissingValueHandler,
    SkewedFeatureTransformer,
    drop_useless_features,
    handle_infinite_and_nan,
    optimize_memory,
)


def test_feature_engineering_shim_exports_legacy_symbols():
    assert FeatureEngineeringTransformer.__module__ == "legacy.feature_engineering"
    assert FeaturePruner.__module__ == "legacy.feature_engineering"


def test_preprocessing_shim_exports_legacy_symbols():
    assert handle_infinite_and_nan.__module__ == "legacy.preprocessing"
    assert optimize_memory.__module__ == "legacy.preprocessing"
    assert drop_useless_features.__module__ == "legacy.preprocessing"
    assert MissingValueHandler.__module__ == "legacy.preprocessing"
    assert SkewedFeatureTransformer.__module__ == "legacy.preprocessing"
    assert CategoricalLevelManager.__module__ == "legacy.preprocessing"
    assert FrequencyEncoder.__module__ == "legacy.preprocessing"
