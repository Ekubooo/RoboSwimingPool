# RoboLab

RoboLab is a robotics simulation workspace **developed from
[MimicKit](https://github.com/xbpeng/MimicKit)**. It retains MimicKit's motion,
policy and training infrastructure and adds Newton 1.7 fluid/rigid simulation,
shared solver components, and PCA motion analysis. These additions are maintained
in three direct Git submodules. RoboLab is not the upstream MimicKit project.

## Modules

| Repository / Python package | Role |
| --- | --- |
| [RobotSwimingPool / RSP](https://github.com/Ekubooo/RobotSwimingPool) | Scene assembly, policy playback, rendering and asset preparation |
| [Solver4Newton / Solver](https://github.com/Ekubooo/Solver4Newton) | PBF, rigid/fluid coupling, scheduling and shared volume-map boundaries |
| [PCA4Mimickit / PCA](https://github.com/Ekubooo/PCA4Mimickit) | Motion PCA, reconstruction, comparison playback and analysis reports |

```text
Robot_Lab/                   # host repository (RoboLab)
  Robot_Lab/                 # ordinary grouping directory
    PCA/                     # PCA4Mimickit submodule
    Solver/                  # Solver4Newton submodule
    RSP/                     # RobotSwimingPool submodule
  mimickit/                  # retained framework and engine integrations
  args/                      # command-line presets
  data/                      # configs, original assets and generated runtime assets
  docs/                      # retained MimicKit usage and algorithm guides
  tools/                     # host launchers and development tools
```

Use the canonical imports `RSP`, `Solver`, and `PCA`. The submodules currently
run within this host checkout; they are not standalone application installations.

## Setup

Clone the host and its published submodules:

```powershell
git clone --recurse-submodules git@github.com:Ekubooo/Robot_Lab.git
cd Robot_Lab
```

For an existing clone, use `git submodule update --init --recursive` to check out
the exact child commits recorded by the host. SSH access to GitHub must be configured;
HTTPS cloning is also supported using the corresponding repository URL.

The validated environment is Windows with an NVIDIA CUDA GPU, Python 3.11,
**Newton 1.7.0 and Warp 1.18.0**. The root
[requirements.txt](requirements.txt) is the unified installation list for MimicKit,
RSP, Solver and PCA; `pyproject.toml` reads this same list. It includes the
simulation backends, viewer UI and asset-preparation dependencies.

For a new environment, install a compatible CUDA-enabled PyTorch build for your
GPU first (the validated environment uses `torch 2.10.0+cu128`), then run:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

The requirements do not choose a CUDA PyTorch build for you. The optional
[constraints-warpsim.txt](constraints-warpsim.txt) retains the same four backend
pins for use when adding packages to an existing environment; it is not a separate
installation step. The RSP asset package retains a smaller component dependency
list for reference, but you do not need to install it separately. The submodules
currently share this host environment; independent package/environment manifests
can be added when standalone installation is supported.

In the **already configured** WarpSim environment, register the checkout without
changing installed dependencies:

```powershell
conda activate WarpSim
python -m pip install -e . --no-deps --no-build-isolation
```

The latter command is not a complete environment installation. This documentation
update was checked in the existing WarpSim environment, not a fresh installation.
Run the following examples from the host root.

## Resources

Download upstream assets and motion data as described in the
[MimicKit integration guide](docs/MimicKit.md#data-and-policies). Local downloaded
assets, motions, models and `output/` are intentionally excluded from Git.

RSP's default profiles additionally reference locally trained task resources:

| Profile | Required local task files |
| --- | --- |
| Humanoid | `output/a5000_dodgeball_2048_full_r01/agent_config.yaml` and `model.pt` |
| G1 | `output/g1_lafan_full_20261107_r01/env_config.yaml`, `agent_config.yaml` and `model.pt` |

Provide those files and the prior/config/asset dependencies they reference, or
adapt `args/rsp_pbf_demo_args.txt` / `args/rsp_pbf_g1_args.txt` to a compatible
checkpoint and its matching configuration. A generic pretrained policy is not an
interchangeable replacement. Cloning the repositories alone does not provide these
local checkpoints. PCA's example requires `data/motions/humanoid/humanoid_getup_faceup.pkl`
and the matching humanoid assets.

## Run RSP

```powershell
python tools/run_rsp.py
python tools/run_rsp.py --profile humanoid --direct
```

The first command opens scene selection. The second launches the default Humanoid
scene directly; `--profile g1 --direct` selects G1 after its resources are prepared.
Use the playback panel for pause, step and reset. Startup prepares missing or
outdated runtime assets under `data/assets_RSP/runtime` from the original sources;
it reuses valid cached assets. Optional explicit preparation:

```powershell
python tools/prepare_rsp_assets.py
```

See the [RSP guide](https://github.com/Ekubooo/RobotSwimingPool) and
[asset pipeline guide](https://github.com/Ekubooo/RobotSwimingPool/blob/main/sim_ready/README.md).

<!-- RSP screenshot slot: paste GitHub-hosted runtime image Markdown here. Keep local image files out of Git. -->

## Run PCA

```powershell
python -m PCA.run_pca --motion humanoid/humanoid_getup_faceup.pkl --components 4 --play
```

PCA compares the original motion with its four-component reconstruction. New
analysis results and caches are written under `output/pca/`. The player supports
comparison, local and classroom views. See the
[PCA guide](https://github.com/Ekubooo/PCA4Mimickit) for other analysis workflows.

<!-- PCA screenshot slot: paste GitHub-hosted runtime image Markdown here. Keep local image files out of Git. -->

## MimicKit functionality

The retained framework still provides training, policy testing, motion viewing
and retargeting tools. See the [integration guide](docs/MimicKit.md), including the
seven algorithm guides. RSP itself focuses on single-scene playback of existing
policies; it is not a replacement for those training workflows.

## Working with submodules

All four repositories use `main`. A recursive clone normally leaves submodules at
the recorded commits with detached HEADs. Before editing a clean, synchronized
checkout, switch each child to its development branch and update it:

```powershell
git -C Robot_Lab/PCA switch main
git -C Robot_Lab/PCA pull --ff-only origin main
git -C Robot_Lab/Solver switch main
git -C Robot_Lab/Solver pull --ff-only origin main
git -C Robot_Lab/RSP switch main
git -C Robot_Lab/RSP pull --ff-only origin main
```

This deliberately selects the latest child development commits; use
`git submodule update --init --recursive` instead when reproducing the host's
recorded versions. For normal development in the same checkout, commit in each
changed child first, then stage and commit the changed child pointers in the host.
No intervening host pull is required just to record local child commits.

Publish child commits before the host:

```powershell
git -C Robot_Lab/PCA push origin main
git -C Robot_Lab/Solver push origin main
git -C Robot_Lab/RSP push origin main
git push --recurse-submodules=check origin main
```

Proceed to the last command only after the child pushes succeed.

## License and acknowledgments

The [Apache-2.0 license](LICENSE) and original MimicKit copyright notice are
retained. RoboLab adds and modifies integrations, tools and documentation; it does
not claim authorship of the original framework. Each child repository carries its
own inherited license and source-history notes. Earlier module history remains in
this host repository.

Data and third-party assets retain their respective terms and attribution; see
[data/LICENSE](data/LICENSE) and the
[RSP asset provenance](https://github.com/Ekubooo/RobotSwimingPool/blob/main/sim_ready/PROVENANCE.md).
The project license is not a blanket grant for externally obtained assets or policies.

For work using MimicKit, see its [paper](https://arxiv.org/abs/2510.13794) and the
preserved [upstream citation metadata](docs/CITATION_MimicKit.cff). That metadata
identifies MimicKit, not a separate RoboLab publication.
