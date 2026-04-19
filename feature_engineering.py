"""Legacy compatibility shim for feature engineering helpers.

The active, canonical feature-engineering pipeline used by DVC and training
stages lives in :mod:`src.feature_engineering`.

This shim keeps historical imports working without leaving a second large
implementation at the repository root.
"""

from legacy.feature_engineering import FeatureEngineeringTransformer, FeaturePruner

__all__ = ["FeatureEngineeringTransformer", "FeaturePruner"]
