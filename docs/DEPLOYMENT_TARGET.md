# Deployment target and execution order

User-confirmed target: Sipeed Maix Bit K210 kit with LCD and camera, identified from the supplied product image. Exact hardware revision, camera sensor, firmware and SDK are not yet verified. No specifications are inferred from the listing beyond the identified product.

User preference: finish development, training decisions, computer-side evaluation and deployment preparation on the PC before moving to board trials. Do not require an attached board for those stages or flash hardware in this phase.

Offline PC: training/distillation, map and descriptor-bank construction, conversion, faithful kmodel simulation, replay of the complete online algorithm, resource-budget analysis and runtime/package preparation.

Final online target: camera → MixVPR → retrieval/active map → SuperPoint → matching → PnP-RANSAC → pose output on K210. Achievability of the complete chain remains to be demonstrated. PC numerical references are not a deployable implementation of CPU-side stages by themselves.

Computer-phase completion requires qualified feature/localization quality under a frozen protocol, model-compatible map data, compiled model artifacts, explicit runtime/firmware compatibility, and a reviewable package with test inputs and expected outputs. Current state does not meet all these conditions; see EXPERIMENT_STATUS.md.

Board phase: confirm runtime I/O, compare saved test vectors, integrate the camera, then measure frame-to-pose latency, peak RAM and sustained stability. Those measurements remain NOT_MEASURED until actual hardware tests. Simulator results and static memory estimates must not be presented as measured board performance.
