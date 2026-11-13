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