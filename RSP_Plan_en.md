The current repository clones the latest MimicKit, initially integrating a custom fluid solver into the Newton engine layer and improving asset pipeline capabilities.

Currently, it supports the latest stable versions of the Newton engine, Nvidia warp dsl, and mujoco warp, using the conda environment "warpsim".

Experimental/previous implementations can be found in two other projects:

1. Fluid Solver. D:\ComputerScience\PROJECT\WarpLab. The fluid project mainly refers to PBF and simready in its root directory. Ignore the demo folder, which contains archived code.

2. MimicKit and its tools adapted for the new Newton, and MimicKit Newton fixes. D:/ComputerScience/PROJECT/MimicKit

Current Status and Tasks:

1. The PBF fluid program currently integrates with the Newton engine using a custom solver. PBF uses the volume map method to achieve two-way coupling between weak fluids and solids.

2. The custom solver backend of the Newton engine is now largely complete, as are the asset pipeline capabilities and definitions. The upper layer is currently assembled using a temporary glue layer. The next step is to gradually remove the glue layer, implement multi-renderer integration, and achieve fluid billboard rendering to refine the overall program flow.

3. The current operational goal is primarily to run the dodgeball task under the Mimickit SMP path, loading existing RL policies into the fluid environment to observe the humanoid robot's behavior. The focus is mainly on visual presentation and program execution; machine learning training and multi-environment operation are not currently considered.

Important Information:

1. Non-destructive/Minimized Destruction/Unnecessary Destruction: This refers to the current architecture of the Mimickit Newton backend, the PBF algorithm solver and its related subordinate main solver methods and architectures, and the completed asset pipeline architecture.

2. Textual descriptions may be inaccurate. All project documents and records are for reference only; actual code must be analyzed and verified.

3. Without referencing existing code, analysis and review should consider the architecture design and code implementation from the perspective of the Newton engine specification, related Newton open-source libraries, and other large open-source robot simulation software libraries. In other words, analysis and suggestions should be provided based on the practices and requirements of major open-source libraries.