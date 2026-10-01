# Map preparation quality audit

Scope: existing stair development map only. Not Floor6/Floor7 independent test.
Queries are the 41 registered project DEV images; subtract each query descriptor contribution from map centroids. Use stored COLMAP pixel coordinates (feature coordinates checked within 1.01 px), per-image SIMPLE_RADIAL intrinsics, top160 legacy keypoints, full-map block scan and deterministic PnP. Query geometry remains in the reconstruction, so results are optimistic internal checks.

| Variant | Good pose | Median rotation (deg) | Median center (native units) | Median matches |
|---|---:|---:|---:|---:|
| FP32 | 41/41 | 0.1171 | 0.007744 | 54 |
| FP16 | 41/41 | 0.1171 | 0.007744 | 54 |
| INT8_per_descriptor | 41/41 | 0.1116 | 0.008289 | 55 |
| PRUNE_track5_error1p75_FP32 | 40/41 | 0.1055 | 0.007777 | 42 |

Good pose means at least 8 inliers, <=5 degrees and <=0.1 native units relative to reconstructed pose; not meters or external ground truth.
Point reprojection-error median 1.465 px, p90 1.873 px; track median 5 observations. These are filtered reconstructed points, not an unbiased accuracy estimate.

Decision: retain all 7,770 points and FP32 XYZ. Provisional INT8 descriptor database reduces descriptor+scale payload from 7,956,480 to 2,020,200 bytes (74.61%). It passes the internal paired no-regression guard. FP16 is a fallback. Do not interpret tiny median differences as real improvements.
Pruning to track>=5 and error<=1.75 loses a baseline good-pose query, so is rejected. Initial track>=3/error<=2 filter was already applied in the source and did not prune anything.

Candidate directory uses symlinks to avoid copying geometry; packaging for microSD must materialize referenced files. Global descriptors are still unbound. Local extractor compatibility remains unverified. INT8 values are normalized in float32 by the PC matcher: this is database quantization, not an integer-only runtime or model PTQ.

Next test: bind frozen global model/reference database, validate local feature compatibility and calibration, then compare FP32 vs candidate on locked independent video queries with labels. No additional pruning is warranted by current evidence.
