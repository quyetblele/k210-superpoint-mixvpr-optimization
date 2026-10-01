"""CPU-only, bounded resource configuration shared by torch stages."""
import os, sys
os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
import cv2, torch
from .settings import WORKSPACE
sys.path.insert(0, str(WORKSPACE / 'scripts'))
import create_superpoint_k210_w0 as w0
from superpoint_kpu_wrapper import SuperPointKPUWrapper
sys.path.insert(0, str(w0.EDGE / 'third_party'))
from SuperGluePretrainedNetwork.models import superpoint as ops
cv2.setNumThreads(1)
torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)
