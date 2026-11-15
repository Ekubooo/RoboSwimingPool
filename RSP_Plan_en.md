This repository contains the latest version of MimicKit, with a custom fluid solver integrated into the Newton physics engine layer.

The environment is configured (using a Conda environment named "warpsim") to support the latest stable versions of the Newton engine, NVIDIA Warp DSL, and MuJoCo Warp.

For reference regarding experimental or previous implementations, please consult the following projects:
1. Fluid Solver: `D:\ComputerScience\PROJECT\WarpLab`. Focus on the PBF and `simready` components in the root directory; ignore the `demo` folder, as it contains archived legacy code.
2. MimicKit (adapted for the new Newton engine) and its associated tools/fixes: `D:/ComputerScience/PROJECT/MimicKit`.

Current Status and Tasks:
1. The PBF (Position Based Fluids) solver is integrated into the Newton engine as a custom solver. It utilizes a "volume map" method to achieve two-way coupling between the fluid and solid objects.
2. The integration of the custom solver with the Newton engine backend is largely complete. The asset pipeline, rendering integration, global time-stepping logic, and a high-performance, lightweight observation system for single-environment Reinforcement Learning (RL) are all functional. The current demo demonstrates good overall performance.
3. The system is currently held together by a temporary "glue" layer; the next step is to gradually phase out this layer to refine the overall program architecture.
4. The primary objective is to run existing RL policies—specifically those developed for the "dodgeball" task within the MimicKit SMP framework—by loading a humanoid robot into the fluid-solid coupled environment to observe its behavior. The focus is on visual demonstration and system execution; machine learning training and multi-environment parallel execution are not currently in scope.

Important Notes:
1. Preservation: It is critical to maintain—or at least minimize disruption to—the existing architecture of the MimicKit-Newton backend, the PBF solver, the core solver methods/structures, and the established asset pipeline.
2. Accuracy: Written documentation and records may not always be precise; they should be used only as a reference. Always verify details by analyzing the actual source code. 3. Without referencing existing code, conduct an analysis and review of the architectural design and code implementation by drawing upon standards and practices found in the Newton engine specifications, Newton-related open-source libraries, and other large-scale open-source robotics simulation software. In other words, provide analysis and recommendations based on the conventions and requirements established by these prominent open-source projects.