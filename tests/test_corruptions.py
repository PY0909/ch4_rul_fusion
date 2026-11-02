import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from rul_fusion.corruptions import (inject_bias, inject_drift, inject_missing,
                                    inject_noise)


def test_corruptions_are_deterministic_and_preserve_shape():
    x = np.ones((4, 5, 3), dtype=np.float32)
    for function, argument in ((inject_missing, 0.2), (inject_noise, 0.1),
                               (inject_bias, 0.2), (inject_drift, 0.2)):
        a, meta_a = function(x, argument, np.random.default_rng(4))
        b, meta_b = function(x, argument, np.random.default_rng(4))
        assert a.shape == x.shape
        assert np.array_equal(a, b)
        assert meta_a == meta_b
