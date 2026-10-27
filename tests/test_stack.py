"""Import the shared scientific stack. No training."""

from __future__ import annotations

import os


def test_scientific_stack_imports():
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy
    import seaborn as sns
    import sklearn

    assert np.__version__
    assert pd.__version__
    assert scipy.__version__
    assert sklearn.__version__
    assert sns.__version__
    assert plt is not None
