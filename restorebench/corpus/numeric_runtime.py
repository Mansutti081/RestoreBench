# ABOUTME: Freezes numerical libraries to one thread before dataset-generation imports.
# ABOUTME: Prevents small benchmark-network solves from paying BLAS thread-management overhead.
from __future__ import annotations

import os


NUMERIC_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
)


def configure_single_threaded_blas() -> None:
    """Force every supported numerical runtime to one thread per process."""
    for variable in NUMERIC_THREAD_ENV_VARS:
        os.environ[variable] = "1"


# Importing this module is the generation bootstrap. It must happen before pandapower/numpy.
configure_single_threaded_blas()
