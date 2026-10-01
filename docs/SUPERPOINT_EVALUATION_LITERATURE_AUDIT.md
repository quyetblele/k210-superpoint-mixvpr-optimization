# SuperPoint evaluation: literature audit and corrected protocol proposal

Status: literature/code audit complete; upstream baseline reproduction and revised experiments NOT YET RUN. Existing experiments remain historical engineering diagnostics. No training or architecture changes were made during this audit.

## Primary-source findings

**SuperPoint (2018), sections7.2–7.3.** Detector repeatability uses240x320 images,300points,3pixel correspondence tolerance. Homography evaluation uses480x640 and up to1000points, with correctness reported at1/3/5pixels. It reports detector and descriptor metrics separately. Our42synthetically transformed indoor pairs are not an HPatches reproduction, and correct-match count alone is not the paper's complete evaluation. [Paper](https://arxiv.org/pdf/1712.07629)

**HPatches.** The original descriptor benchmark uses patches to separate descriptor evaluation from detector choices. Its tasks include verification, image matching and patch retrieval. Distinguish that benchmark from the full-image homography evaluation commonly performed on HPatches sequences. [Authors' paper](https://openaccess.thecvf.com/content_cvpr_2017/html/Balntas_HPatches_A_Benchmark_CVPR_2017_paper.html)

**HF-Net (CVPR2019), sections3 and5.3.** Hierarchical retrieval followed by local2D–3D matching is valid. Crucially, the authors construct feature-specific models using reference pair matching, geometric filtering and triangulation with known reference poses. Thus preserving camera/world geometry does not imply old keypoint tracks are interchangeable between detectors. [Paper](https://arxiv.org/pdf/1812.03506)

**HLoc implementation.** Query/reference keypoint matches are linked to Point3D IDs through each reference image's observations. `localize_sfm.py` converts query coordinates by adding0.5 and uses pycolmap pose estimation/refinement. Its current entry-point RANSAC default is12pixels, not a universal optimal threshold. It also has a retrieval-pose fallback in one path: a pose written to file is not automatically a successful PnP estimate. [Code](https://raw.githubusercontent.com/cvg/Hierarchical-Localization/master/hloc/localize_sfm.py)

**Official HLoc SuperPoint presets.** `superpoint_aachen` specifies4096keypoints and maximum side1024; InLoc uses maximum side1600. These are examples of research localization presets, not requirements for every device. A184x320/160point control must be labeled resource-constrained, not full-strength public SuperPoint performance. [Feature configuration](https://raw.githubusercontent.com/cvg/Hierarchical-Localization/master/hloc/extract_features.py)

**Matcher semantics.** HLoc's `NN-superpoint` preset uses mutual checking and Euclidean descriptor distance0.7. Its `NN-ratio` preset uses a ratio0.8. These differ from our cosine>=0.8 plus additive margin>=0.05. For normalized descriptors, distance squared equals2(1−cosine), so distance0.7 corresponds to cosine>=0.755; the mutual check still matters. [Presets](https://raw.githubusercontent.com/cvg/Hierarchical-Localization/master/hloc/match_features.py), [matcher](https://raw.githubusercontent.com/cvg/Hierarchical-Localization/master/hloc/matchers/nearest_neighbor.py)

**Descriptor sampling.** HLoc explicitly documents an original sampling issue and offers `fix_sampling`, retaining False by default for compatibility. Changing this is a separate intervention requiring consistent query/reference features; it must not silently invalidate existing caches. [Extractor](https://raw.githubusercontent.com/cvg/Hierarchical-Localization/master/hloc/extractors/superpoint.py)

**Evaluation roles.** The long-term localization benchmark distinguishes reference/database images from query images and evaluates position and orientation. Those roles are not synonyms for neural-network TRAIN/DEV/TEST splits. [Benchmark organizers](https://www.visuallocalization.net/)

**System-level comparison.** The Image Matching Benchmark embeds methods in full geometry pipelines rather than relying only on intermediate local-feature scores, and makes evaluation settings explicit. [Authors' benchmark](https://github.com/ubc-vision/image-matching-benchmark)

## Audit of this workspace: observations, not literature claims

1. We imposed TRAIN-only references, removing an entire consecutive DEV segment. Reference-track coverage fell to median13.74%. That is a difficult coverage-limited experiment, not a neutral measure of intrinsic feature quality.
2. Alternating DEV support/query restored teacher performance, but nearby-frame20query sanity is not independent evaluation. Neither0/41 nor20/20 can replace a published/public benchmark.
3. Raw correct MNN count, full pose success, solver acceptance, and teacher/student retention ratios have sometimes been discussed too closely together. They answer different questions and need separate result tables.
4. Student descriptors were attached to old observations by nearest-keypoint association or by dense resampling. Neither establishes that the student's own detections produce valid multi-view tracks. This is an interoperability/map-association experiment, not yet a complete feature-specific localization baseline.
5. Caps160/600/120, top3 references, additive matching margin and100RANSAC iterations are custom deployment choices. They should be evaluated as hardware-budget interventions after a trusted baseline works.
6. The0.1native-unit/5degree good-pose criterion is an engineering criterion. Map units have not been established as meters; it is not interchangeable with benchmark metric thresholds.
7. Local HLoc commit is `c13273bd0ecc2917a35910fd843712a1c6243193`. Its code confirms the0.5coordinate conversion and sampling option. Current custom map code transforms `im.xys` directly into network coordinates and uses OpenCV PnP. There is a coordinate-convention compatibility question to test; this audit does not prove a bug or attribute0/41 to a half-pixel offset.
8. Teacher low-resolution controls use original NMS while the latest students use greedy/corner tie handling. That is appropriate for an explicit postprocessing experiment, but not an isolated network-only comparison.

The following evidence remains useful: frozen actual-kmodel homography comparisons isolated a large NMS contribution; reconstruction known-correspondence PnP passed41/41 internally; per-image descriptors improved the covered low-resolution teacher9→16/20. These remain scoped engineering findings, not paper-level model rankings.

## Proposed replacement: three separate evaluation tracks

### A. Upstream reference reproduction

Pin published weights, upstream code commit, dependencies and configuration. Run an official full-image HPatches evaluator on a documented subset first, then the full benchmark when available. Verify geometry transformation after resizing, coordinate axes, masks, interpolation, NMS, sampling and point-count rules. Report correctness at the specified thresholds, repeatability, localization error and descriptor/matching metrics. Record expected differences between published paper settings and the released implementation; do not promise identical numbers from different weights/software.

For localization, use a pinned HLoc NN configuration as the PC reference implementation. No LightGlue/SuperGlue installation is required for an NN baseline. Capture actual solver status separately from fallback poses. Use standard baseline settings initially; bounded batches/threads may be changed for stability without changing numerical protocol.

### B. Controlled teacher–student comparison

Use the same query/reference roles, resolution/FOV, keypoint cap, postprocessing, descriptor sampling and matcher/solver when isolating architecture quality. Run teacher high-resolution quality reference in a separate column, not as an equal-budget comparison. If NMS must differ for deployment, report both common-postprocessing and deployed-postprocessing comparisons.

Compare frozen FP32 and actual nncase outputs on identical inputs and preprocessing identities. Any detector/descriptor hybrid is a diagnostic only; reference and query descriptor spaces must use the same donor network. Do not compare a student query vector directly against teacher reference vectors just because both have256dimensions.

### C. Localization and hardware-budget ablations

Lock separate database/query manifests with coverage statistics and intended capture separation. Keep unsupported queries in overall results; optionally report a predeclared coverage-conditioned diagnostic without choosing the filter from model outcomes. Never remove hard queries until teacher performance looks good.

Preserve existing reference poses as the geometric anchor. Establish feature-consistent reference keypoint indices, verified pair matches and triangulated tracks for each method; compare shared-old-map resampling only as an explicitly named secondary experiment. Store image/feature-index/Point3D associations and fail on identity mismatch.

After reference reproduction, introduce one budget restriction at a time: resolution, point cap, retrieval breadth, active-map cap, matcher/solver budget, then actual INT8. Choose any configurable recipe on DEV under equal search budgets; freeze before final evaluation. Record both position and orientation errors, success curves, per-query failures and actual runtime/memory scopes. No board claims from PC timing.

## One next gate

**UPSTREAM_EVALUATOR_AND_COORDINATE_CONTRACT_REPRODUCTION.** Before further student training or quality claims, validate the pinned upstream extractor/matcher/pose path and coordinate/sampling conventions on a small fixed sample. Then freeze a revised protocol. Do not turn HLoc's12pixel default into a post-hoc fix for our4pixel failures, or change split/point budgets to force a high teacher score.

No new numerical experiment was run as part of this literature audit. The next protocol is a proposal, not a claim of upstream reproduction PASS.
