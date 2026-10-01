import os, sys
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
