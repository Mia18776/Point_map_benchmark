# map-anything's mapanything/utils/metrics.py (evaluate_ate's alignment) calls
# np.linalg.linalg.svd, a private alias numpy 2.x removed. It is the same
# function as np.linalg.svd, so restoring the alias changes no result.
import numpy as np

if not hasattr(np.linalg, "linalg"):
    np.linalg.linalg = np.linalg
