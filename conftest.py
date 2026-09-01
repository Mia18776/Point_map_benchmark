"""Make the package and the scripts importable when running ``pytest`` directly.

``python -m pytest`` puts the working directory on ``sys.path`` but a bare
``pytest`` invocation does not, so the tests would fail to import
``pointmap_bench`` without this.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
for path in (ROOT, os.path.join(ROOT, "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)
