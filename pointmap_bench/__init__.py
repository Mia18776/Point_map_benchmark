"""pointmap-benchmark: compare feed-forward 3D point-map models on equal footing.

Supported models (all driven through MapAnything's unified model interface):
MapAnything, MASt3R (+ sparse global alignment), Pi3 and Depth Anything 3.
"""

from .data import Scene, discover_scenes, list_images, load_ground_truth, load_views
from .metrics import evaluate_with_gt, evaluate_without_gt
from .models import (
    DEFAULT_MODELS,
    MODEL_REGISTRY,
    ModelSpec,
    ModelUnavailable,
    check_availability,
    load_model,
    run_model,
)
from .prediction import Prediction
from .report import aggregate_by_model, write_reports
from .sources import FolderViewSource, PreparedViews, ViewSource, WaiViewSource

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_MODELS",
    "MODEL_REGISTRY",
    "FolderViewSource",
    "ModelSpec",
    "ModelUnavailable",
    "PreparedViews",
    "Prediction",
    "Scene",
    "ViewSource",
    "WaiViewSource",
    "aggregate_by_model",
    "check_availability",
    "discover_scenes",
    "evaluate_with_gt",
    "evaluate_without_gt",
    "list_images",
    "load_ground_truth",
    "load_model",
    "load_views",
    "run_model",
    "write_reports",
]
