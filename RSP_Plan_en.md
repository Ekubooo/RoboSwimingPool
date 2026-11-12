This repository is a clone of the latest version of MimicKit, with an initial integration of a custom fluid solver into the Newton engine layer.

The setup is compatible with the latest stable versions of the Newton engine, Warp DSL, and MuJoCo Warp. A Conda environment named `warpsim` is currently being used.

For the experimental implementation, please refer to the following projects:
1. Fluid Solver: `D:\ComputerScience\PROJECT\WarpLab`. Focus on the `PBF` and `simready` directories at the root; the `demo` folder contains archived legacy code and should be ignored.
2. MimicKit (adapted for the new Newton engine), its associated tools, and relevant fixes: `D:/ComputerScience/PROJECT/MimicKit`.
3. Documentation regarding the previous project structure and general conventions: `D:\ComputerScience\PROJECT\MimicKit\NEWTON_PBF_SIMREADY_INTEGRATION_PLAN.md`.

Current Status and Tasks:
1. The PBF fluid solver is integrated as a custom solver, utilizing a "volume map" method to achieve weak two-way coupling between the fluid and rigid bodies.
2. The custom solver backend for the Newton engine is largely complete, currently held together by a temporary "glue" layer. The next steps involve phasing out this glue layer, refactoring the asset pipeline and renderer, and refining the overall program workflow.
3. The current objective is to run the "dodgeball" task (within the MimicKit SMP pipeline) by loading an existing reinforcement learning (RL) policy into the fluid environment to observe the behavior of a humanoid robot. The focus is primarily on visual output and program execution; machine learning training and multi-environment simulation are not currently in scope.

Important Notes:
1. Preservation: Avoid modifying—or at least minimize changes to—the existing architecture of the MimicKit Newton backend, the PBF solver, and the core methods of the underlying solvers it invokes.
2. Accuracy: Written descriptions may be imprecise; all project documentation and records should be treated as references only—always verify against the actual source code.
3. Design Approach: Do not rely on existing code; instead, derive the architecture and implementation based on Newton engine specifications, related open-source Newton libraries, and other open-source robotics simulation software. In other words, by using the practices and requirements of major open-source libraries as a benchmark, we provide an analysis and offer recommendations.

Pending follow-up: asset orchestration and Newton adaptation (2026-11-12)

The current priority is to complete and validate low-level asset capabilities. The following upper-level tasks remain pending:

- [ ] Unify rigid/container asset preparation: compose existing geometry, collision and boundary tools through consistent preparation interfaces and artifact references. Generate only requested outputs; reuse the boundary pipeline for PBF assets and keep runtime body IDs outside shared asset descriptions and cache keys.
- [ ] Isolate Newton asset adaptation: extract Newton body-geometry reading and ModelBuilder population from scene asset recipes into a small adapter. Retain a thin startup coordinator and preserve current backend/solver locations, asset paths, cache identities and numerical methods.

Asset capability inventory (2026-11-12)

Status refers to capabilities integrated in this repository, not every feature available in its dependencies. Missing capabilities are candidates for future work, not an instruction to implement all of them now. Rigid geometry processing already works; unifying preparation entry points is a separate orchestration task.

| Category | Capability | Status | Current scope or gap |
|---|---|---|---|
| Input and transforms | OBJ geometry IO, explicit scaling and up-axis conversion, preservation of the local origin | Implemented | Primarily triangle-mesh OBJ inputs. |
| Primitives | Parametric primitive mesh generation | Partially implemented | Boxes and spheres have a generic entry point; capsules are already used by the scene; cylinders and other types are not integrated. |
| Mesh processing | Exact-position welding, solid-component splitting, Boolean union and simplification | Implemented | Intended for valid closed solids, not arbitrary damaged meshes. |
| Basic validation | Indices, finite values, duplicate/degenerate faces, closure, manifold topology, winding and validity after float32 conversion | Implemented | Existing checks reject invalid inputs; they are not a general geometric self-intersection test. |
| Geometric self-intersections | Independent triangle self-intersection detection and defect localization | Not implemented | Topological validity alone does not prove the absence of self-intersections. |
| Mesh repair | Bad-face cleanup, hole filling and normal/winding correction | Not implemented | Exact-position welding exists; a complete, explicitly enabled repair workflow does not. |
| Containers and cavities | Container geometry construction and validation | Partially implemented | Current validation assumes a cuboid exterior and one cavity; arbitrary exteriors, multiple cavities and general hollow rigid bodies are not uniformly supported. |
| Collision geometry | CoACD decomposition, hull validation and error statistics | Implemented | Uses a fixed recipe; diagnostics include sampled distances and volume differences. |
| SDF | Sparse solid/container SDF baking and support-range handling | Implemented | CUDA, float32 and the existing input constraints. |
| Volume maps | Independent volume-map generation from an SDF, support radius and quadrature settings | Implemented | Leaves the input SDF unchanged; computation is separate from file IO. |
| Boundary data validation | Grid transforms, actual tile coverage, support metadata and local bounds | Implemented | Some helpers live in asset modules; moving them into a library would be organization work, not new algorithms. |
| Storage infrastructure | OBJ/NVDB/hull NPZ IO, fingerprints, staging, replacement publication, failure rollback and path protection | Implemented | Supports the existing sequential build workflow. |
| Visual resources | Complete packaging and validation of material/texture dependencies | Not implemented | Deferred until visual-asset integration. |

Humanoid and motion validation clarification:

- Implemented in MimicKit: character loading, forward kinematics, motion loading/interpolation and velocity processing. Existing input locations and these implementations should be reused.
- Partially implemented in the native loading path: model/root and supported-joint checks, kinematic-array lengths, initial-pose DOF count, paired simulation/kinematic body-name order, loop-mode enumeration and nonnegative motion weights. The body-name comparison uses zip without an explicit equal-length check. Native motion loading lacks a complete up-front check of positive finite FPS, minimum frame count, exact frame width, finite frame values and positive finite total sampling weight.
- Implemented in repository offline tools: tools/motion_analysis/data/motion.py checks required fields, two-dimensional frames with at least two samples and seven channels, finite values, positive finite FPS and loop mode. data/resampling.py additionally checks character DOF count when the FPS actually changes. These checks are not automatically used by the native demo loader.
- Partially implemented for SMP prior compatibility: file existence, observation settings, key-body count and control frequency are checked; SHA-256 is checked only when configured, and exact key-body order is checked in the LaFAN-adapter path. These do not constitute general skeleton/DOF semantic validation for every motion or policy.
- Not implemented as a common RSP asset capability: shared preflight validation connecting those existing checks to complete motion-to-character compatibility. Native motion records contain frames, FPS and loop mode, without a joint-name/order declaration; matching channel counts alone cannot establish joint semantics. Reuse the existing validators before adding missing checks.

Collision-cache manifests and corrupt-cache recovery belong to asset-pipeline policy, not missing convex-decomposition algorithms. Physics stepping and training loops remain outside the asset library.
