# Newton-PBF 初期接入（2026-11-10）

已新增独立 `newton_pbf` 后端及 `pbf_play` 环境，用原 SMP 权重持续运行单环境流固耦合场景。原 `newton_engine.py`、`newton_1_7_engine.py`、训练配置、模型、运动数据及 PBF 核心/helper 保持原文。没有提交或推送 Git。

运行环境是 Conda WarpSim：Newton 1.7.0、Warp 1.18.0、MuJoCo/MuJoCo Warp 3.15.0；单个 `cuda:0` 环境、种子 1234，策略 30 Hz、物理 240 Hz。

在项目根目录启动：

```powershell
conda activate WarpSim
python -B mimickit/run.py --arg_file args/rsp_pbf_demo_args.txt
```

无窗口运行：

```powershell
python -B mimickit/run.py --arg_file args/rsp_pbf_demo_args.txt --visualize false
```

程序持续运行，关闭窗口或 Ctrl+C 退出。`R`、原生 Reset 按钮和 `Reset scene (R)` 同时重置整场景。Enter 暂停/继续，Space 单步。左侧 `Policy: scene sphere` 切换策略末尾六维观测的来源；配置项为 `data/envs/pbf_play_env.yaml` 中的 `policy_use_scene_sphere`，默认 false。球始终参与物理模拟。

| 模块 | 职责 |
| --- | --- |
| `mimickit/engines/newton_pbf_engine.py` | 完整复制 1.7 后端后独立修改，管理共享 Model/State、粒子双缓冲、控制步、CUDA Graph、接触接口与重置 |
| `mimickit/engines/newton_pbf_runtime.py` | 粒子初始化、边界绑定、坐标转换、布局检查、GSI 后整体摆放 |
| `mimickit/sim_ready/pipelines/pbf_play.py` | 临时资产生产：静态池/bunny、原生 body 局部几何、SDF、代理和内容缓存 |
| `mimickit/solvers/pbf/coupling.py` | 边界姿态同步 → PBF → 反力/力矩相加 → MuJoCo → 同步新姿态 |
| `mimickit/envs/pbf_play_env.py` | 复用 SMP 动作、基础观测、运动学和 GSI；零奖励、无自动终止、无躲避球任务实体 |
| `mimickit/engines/newton_pbf_viewer.py` | GPU 实例化立方体粒子，边长等于粒子直径 0.025 m；原生显示开关与资源释放 |

共享应用代码仅增加两处工厂注册。已有 PBF 文件唯一改动为 `boundary_model.py` 中 `Sim_Ready` → `sim_ready` 的导入适配。未增加 PBF observables，接触力及执行器力请求委托 MuJoCo。禁用 Newton 的粒子-形状碰撞标志，避免与 PBF 重复反馈；原刚体碰撞仍由 MuJoCo 处理。

复制后端的必要调整包括：奇数子步复制粒子状态；CUDA Graph 记录 PBF 预测缓冲交换关系、预分配 HashGrid、捕获后恢复 Python 计数；去除组合求解器错误触发的通用 IK 回写；重置使用 MuJoCo `flags=0` 保留新 GSI 关节状态。重置恢复两份粒子状态、工作缓冲、边界姿态和三个自由刚体，不重新烘焙。

场景采用 Z-up，SimReady 通用程序仍保持 Y-up。仅临时场景适配器执行 `(x,y,z) → (x,2-z,y)`；人体 body-local 坐标不旋转。PBF 的物理标量沿用 `PBFConfig()` 代码默认值，时间步由引擎提供；没有读取流体参数 JSON，也没有修改空间重排或恢复原始粒子 ID。

池内尺寸 3×2×5 m、壁厚 0.1 m，底面为 Z=0；刚体底部只使用原地面，四壁及高位顶盖为隐藏的静态碰撞形状，池边用线框显示。bunny 原始文件为 `mimickit/sim_ready/Assets/bunny_flat_bottom.obj`，高度 0.6 m、包围盒中心 (2.15,1,0.5)。它和容器分别校验、组合成互不相交的闭合壳体，不进行二者之间的布尔合并，共用一份静态 SDF/VM。bunny 没有 body，其凸分解代理挂在世界上。

三自由刚体的模型质量为 13.5、32.4、10.602877 kg，对应指定尺寸和密度。人形保留 50.05 kg、28 DoF 和 PD 设置。人体 15 个 body 与三个自由刚体各绑定一份 body-local 边界；同 body 复合碰撞几何先合并，质心由 Newton `body_com` 转到世界。初始水块为 (0.1,0.75,0.5) 到 (1.5,1.9,2.5)，共有 195,525 粒子。每次 GSI 后整体平移人形，最低几何点放在 Z=2.6，检查与池壁、顶盖间隙。

临时产物位于 `output/sim_ready/pbf_play/`，运行时按需生产并缓存。静态 SDF 体素为 0.0075 m，body-local SDF 为 0.005 m；SDF 支持半径与 VM 均为 0.05 m，VM 使用原 16 点/轴积分规则。缓存核对 OBJ 和落盘 SDF 内容摘要，VM 追加保存不会改变 SDF 身份。`scene_manifest.json` 记录 body 绑定、局部包围盒、质量、质心、初始间隙和静态符号探针。

验收记录保存在 `output/batch2_integration/`，验收脚本为本次定向运行工具，未迁入原测试套件：

- `eager.json`：普通推进 60 控制步，加载原策略及 4096 个 GSI 样本，146 维观测/28 维动作、全状态有限。
- `headless.json`：CUDA Graph 下 600 步默认输入、300 步真实球输入、手动重置后 300 步；无自动终止，所有 18 个动态 body 均有压力反馈。
- `visual-cubes.json` 和对应 PNG：立方体粒子可视化 450 步；第 150 步切换输入，第 300 步调用原生重置回调，第 420 步注入 R 键事件，关闭窗口后检查 GL context 和引擎释放。
- `interrupt.json`：通过应用主循环注入 KeyboardInterrupt，检查 finally 清理路径；不是外部控制台键盘硬件测试。
- `coupling_probe.json`：固定初态和控制的 eager 重复、eager/graph 对照，以及只去掉力传递的对照。前 10 控制步对照一致；30 步时 graph/eager 最大 body 位置差约 0.030 m，粒子分布总变差约 0.0077；重复 eager 本身约 0.092 m / 0.0107。碰撞后轨迹不保证逐位复现。移除反力传递后，30 步时人形 body 最大位置差约 0.274 m，三个自由体差约 0.274/0.252/0.866 m，确认反力影响真实刚体运动。
- `regression.json`：原 Newton 1.7 SMP demo 1 回合、600 步/20 s，回报约 493.653，4096 个 GSI 样本，正常退出。
- `asset_validation.json`、`integrity.json`、`backend_changes.diff`：资产摘要/支持半径/坐标校验、已有文件保护核对和复制后端的增量差异。

有意保留的边界：当前只支持一个 CUDA 环境、一个 PBF 实例和固定场景。bunny 的刚体代理为既有程序生成的 32 个凸块，体积对称差约 4.41%，与流体 SDF 并非几何完全一致；这是代理近似，不是缩放或坐标错位。策略在流体中可能摔倒，输入切换也可能改变行为；均不触发自动重置。通用资产管线重构、训练和 PBF observables 不属于本批。
