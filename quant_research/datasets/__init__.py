"""Dataset builders for model research."""

from quant_research.datasets.intraday_features import (
    IntradayFeatureConfig,
    build_intraday_feature_matrix,
)
from quant_research.datasets.manifests import (
    DatasetPartitionManifest,
    file_sha256,
    read_dataset_manifest,
    write_dataset_manifest,
)
from quant_research.datasets.splits import (
    PurgedTimeSplitConfig,
    WalkForwardWindow,
    purged_time_split,
    walk_forward_time_splits,
)
from quant_research.datasets.supervised import (
    ForwardReturnLabelConfig,
    add_cross_sectional_label_rank,
    build_alpha_feature_matrix,
    build_forward_return_labels,
    build_multi_horizon_forward_return_labels,
    join_alpha_features_and_labels,
)

__all__ = [
    "ForwardReturnLabelConfig",
    "IntradayFeatureConfig",
    "DatasetPartitionManifest",
    "PurgedTimeSplitConfig",
    "WalkForwardWindow",
    "add_cross_sectional_label_rank",
    "build_alpha_feature_matrix",
    "build_intraday_feature_matrix",
    "build_forward_return_labels",
    "build_multi_horizon_forward_return_labels",
    "file_sha256",
    "join_alpha_features_and_labels",
    "purged_time_split",
    "read_dataset_manifest",
    "walk_forward_time_splits",
    "write_dataset_manifest",
]
