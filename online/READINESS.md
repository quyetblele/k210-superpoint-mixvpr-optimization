# Frozen online integration

`run_frozen.py` runs independent raw-frame MixVPR/SuperPoint branches,
bounded retrieval and direct 2D–3D matching, then CPU PnP. Outputs are JSONL
positions/failure states and a latency summary; no visual renderer is required.
This is a PC reference, not K210 firmware or board validation.

```bash
.venv-mixvpr-cuda/bin/python online/run_frozen.py \
  --model SELECTED_DEV_CHECKPOINT.pt --candidate B_C160_H96 \
  --map VERIFIED_RUNTIME_MAP --config online/config_laptop.json \
  --calibration VERIFIED_CAMERA.json --video DEVELOPMENT_VIDEO.mp4 \
  --output online/runs/dev_001.jsonl --max-frames 100
```

Select A or B matching the checkpoint. Frozen SuperPoint loads its hash-checked
CW90 raw-head ONNX, restores detector channel/spatial orientation, applies the
release's greedy NMS and sampling, and returns original-frame coordinates.
Reference local descriptors must be regenerated with this exact recipe before
assigning its identity. Never relabel legacy descriptors to bypass the guard.

Required remaining inputs:

- DEV-selected completed MixVPR checkpoint and rebuilt reference global bank.
- Runtime map with frozen SuperPoint descriptors associated with valid 3D tracks.
  `online/maps` is currently empty; old stair map instructions are retired.
- Real live/video calibration and explicit camera model. Floor6 reconstruction
  uses RADIAL_FISHEYE, which cannot be passed as OpenCV pinhole distortion.
  A verified rectification/conversion is required before this runner accepts it.

Synthetic software smoke does not establish localization accuracy. Use
development video to validate operation; keep GT1 out of parameter selection.
K210 transport, portable PnP and physical-board timing/memory remain separate
gates. No teacher/student training process is started or modified by this runner.
