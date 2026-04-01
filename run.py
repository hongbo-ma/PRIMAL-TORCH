#!/usr/bin/env python
"""
Wrapper that sets all env vars BEFORE any torch import happens,
then launches driver.py in the same process.
"""
import os
os.environ["TORCH_DISABLE_ONEDNN"] = "1"
os.environ["ONEDNN_PRIMITIVE_CACHE_CAPACITY"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Now it's safe to import torch and run
import runpy
runpy.run_path("driver.py", run_name="__main__")
