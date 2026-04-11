"""
Wide+deep CNN: 4→8→8→4 with fused GEMM+bias+ReLU on TinyTPU.

Profile with:
    PYTHONPATH=tinygrad python scripts/profile_tpu.py --from-tinygrad scripts/models/cnn_4_8_8_4.py

Visualize:
    open scripts/viz_pipeline.html   # Load JSON → trace.json
"""
import os, numpy as np
os.environ.setdefault("TINYTPU_SIM", "build/mkTbTinyTPURuntime.bexe")
os.environ["DISABLE_COMPILER_CACHE"] = "1"
from tinygrad import Tensor

np.random.seed(99)

x  = Tensor(np.random.randint(-2, 3, size=(4, 4), dtype=np.int32), dtype="int32", device="TINYTPU")
Wa = Tensor(np.random.randint(-1, 2, size=(4, 8), dtype=np.int32), dtype="int32", device="TINYTPU")
ba = Tensor(np.random.randint(-1, 2, size=8,      dtype=np.int32), dtype="int32", device="TINYTPU")
Wb = Tensor(np.random.randint(-1, 2, size=(8, 8), dtype=np.int32), dtype="int32", device="TINYTPU")
bb = Tensor(np.random.randint(-1, 2, size=8,      dtype=np.int32), dtype="int32", device="TINYTPU")
Wc = Tensor(np.random.randint(-1, 2, size=(8, 4), dtype=np.int32), dtype="int32", device="TINYTPU")
bc = Tensor(np.random.randint(-1, 2, size=4,      dtype=np.int32), dtype="int32", device="TINYTPU")

t = ((x  @ Wa) + ba).relu().realize()   # Layer 1: 4→8 + relu
t = ((t  @ Wb) + bb).relu().realize()   # Layer 2: 8→8 + relu
t = ((t  @ Wc) + bc).realize()          # Layer 3: 8→4

print(t.numpy())
