"""Unit tests run on CPU; experiment workers own the GPUs."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "2")
