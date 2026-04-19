"""Legacy compatibility shim for preprocessing helpers.

The canonical preprocessing entrypoint for the project is :mod:`src.preprocess`.
This module only preserves the older root-level imports for notebooks and
backward compatibility.
"""

from legacy.preprocessing import (
    CategoricalLevelManager,
    FrequencyEncoder,
    MissingValueHandler,
    SkewedFeatureTransformer,
    drop_useless_features,
    handle_infinite_and_nan,
    optimize_memory,
)

__all__ = [
    "CategoricalLevelManager",
    "FrequencyEncoder",
    "MissingValueHandler",
    "SkewedFeatureTransformer",
    "drop_useless_features",
    "handle_infinite_and_nan",
    "optimize_memory",
]
