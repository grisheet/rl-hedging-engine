"""Shared type aliases used across rlhedge."""
from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

# Option kind
OptionKind = Literal["call", "put"]

# Generic array-like accepted by pricing functions
ArrayLike = npt.ArrayLike

# Concrete numpy array
FloatArray = npt.NDArray[np.float64]
