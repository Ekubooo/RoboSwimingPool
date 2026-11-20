This project is based on a recent clone of the Mimickit repository, with a custom fluid solver integrated into the NVIDIA Newton physics engine layer.

The project is currently configured to work with recent stable versions of the Newton engine, NVIDIA Warp DSL, and MuJoCo Warp, utilizing a Conda environment named "warpsim."

Current Status and Objectives:
1. The project is largely complete in its overall architecture. It encompasses a custom solver, coupling with a rigid-body solver, an asset pipeline, basic rendering capabilities, scene assembly, and the loading of reinforcement learning (RL) policies for humanoid agents. The project structure is well-defined, with clear, logical module responsibilities and a self-consistent set of features.
2. The primary objective is to execute existing RL policies—specifically those developed for the "dodgeball" task within Mimickit’s SMP (Simulation/Motion/Physics) pipeline—by loading humanoid robots into a fluid-structure interaction (FSI) environment to observe their behavior in fluid. The focus is on visual output and program execution; machine learning training, multi-environment parallel execution, and high-precision physical force calculations are currently out of scope.

Key Guidelines:
1. Minimize disruption: Avoid altering the existing overall program architecture unless absolutely necessary.
2. Exercise caution with documentation: Written descriptions may be inaccurate; all project documentation, records, and test logs should be treated as reference material only—the actual source code must be analyzed directly.
3. Independent analysis: Do not rely on or execute existing test code. Instead, evaluate the architecture and implementation by benchmarking against established standards found in the Newton engine specifications, related open-source Newton libraries, and other prominent open-source robotics simulation software. Provide recommendations based on the conventions and requirements typical of such large-scale, open-source projects.