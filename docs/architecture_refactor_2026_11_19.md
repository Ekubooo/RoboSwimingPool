# RSP 架构重构实施记录

本轮完成六个模块的职责整理，保持单仓库、原安装方式与依赖版本。
入口仍为 `python tools/run_rsp.py`；当前架构和资源所有权见
[RSP/README.md](../RSP/README.md)。

## 文件迁移和依赖

| 原位置 | 当前实现位置 |
| --- | --- |
| `RSP/launcher` | `RSP/app/application.py`、`app/ui`；策略配置解析归 `integrations/mimickit/profiles.py` |
| `RSP/runtime/playback*.py`、`interaction.py` | `RSP/app/playback*.py`、`app/ui/interaction.py` |
| `RSP/runtime/inference.py`、`observations.py`、`observation_math.py`、`g1_playback.py` | `RSP/integrations/mimickit` |
| `mimickit/engines/newton_pbf_{engine,api,runtime,viewer}.py` | 环境适配归 `integrations/mimickit`；Newton 状态和步进归 `simulation`；显示归 `rendering/presentation.py` |
| `mimickit/envs/pbf_play_env.py` | `RSP/integrations/mimickit/environment.py` |
| `mimickit/solvers/pbf` | `RSP/physics/pbf` |
| `RSP/runtime/coupling.py`、`fluid_clock.py` | `RSP/physics/clocked_coupling.py`、`fluid_clock.py` |
| `sim_ready/pipelines/boundary/{schema,load}.py` | `sim_ready/boundary` |
| `sim_ready/pipelines/boundary/prepare.py` | `sim_ready/pipelines/boundary.py` |
| `sim_ready/pipelines/scenes/{spec,assembly}.py` | `simulation/scenes/{spec,resolve}.py` |
| 场景资产制作配方 | `sim_ready/pipelines/recipes`；共用参考坐标归 `sim_ready/reference.py` |
| `launcher/preview_assets.py`、`rendering/asset_preview.py` | `rendering/preview/{geometry,textures}.py` |
| `rendering/playback_ui.py` | 应用面板归 `app/ui/playback.py`；Newton GUI 适配归 `rendering/newton_gui.py` |
| 失效的历史资产 preset | `docs/archive/pbf_preset` |

旧 Engine、环境和 `RSP/runtime/playback.py` 入口保留薄转发；Solver 没有旧路径副本。
历史 `newton_pbf_engine_before_refactor.py` 仅供已有检查工具使用。

资产 API 显式接收支持半径。体积图预计算和 NVDB 写入仍属资产管线，运行时边界
查询和力计算仍属物理模块。生产摘要使用稳定角色名和源码内容；七个旧版本通过
精确摘要兼容表复用，并继续校验全部产物，不改写旧清单。

应用统一绑定 Viewer、会话和面板；MimicKit 会话负责环境私有字段和策略操作。
`NewtonStatePair` 保持主状态存储，`StepHooks` 保留现有执行次序。正常步进仍由
`env.step()` 绘制；暂停或图更新失败时才调用 `render_only()`。

共有 CUDA–OpenGL 生命周期归 `RegisteredBuffer`，粒子实例资源由组合复用。
数据布局、着色器和绘制顺序保留，继续使用 `fallback_to_copy=False`。

## 必要检查结果

- `python tools/check_rsp_architecture.py --solver-baseline 3ae4d6d`：89 个模块通过
  依赖检查；11 个 Solver 文件除批准的导入和模块路径替换外与重构前一致。
- 资产：禁用烘焙及发布写入后，真实准备 CLI 复用全部七个版本；支持半径变化仍报
  版本冲突。发布目录 254 个文件的内容和修改时间均未变化。
- 场景和 G1 资产：11 项针对性用例通过；应用 CPU 用例覆盖选择、取消、加载失败、
  暂停、单步、重置、图更新失败、借用窗口和旧入口绑定。
- 后端：五种控制模式、两类关节布局通过。CPU Warp 数组检查确认奇偶子步都写回
  固定输出地址，并保持反馈和回调次序。
- `python tools/check_rsp_same_window.py`：Humanoid 选择后在同一窗口/context
  运行 11 帧，策略周期、重置和借用窗口生命周期通过。
- `python tests/check_rsp_g1_playback.py`：G1 无窗口 11 帧，策略、观测、GSI 和
  重置通过。首次退出时第三方 wandb 临时目录清理遇到沙箱权限提示；检查脚本现已
  将后续临时目录定位到工作区。此调整未重复进行 GPU 运行。
- 四项实际 GL 用例通过：失败上传后的解除映射与复用、扩容和模型切换、
  Original/Transparent/SSFR 模式切换、CUDA/GL 资源先于窗口释放。
- 补充真实窗口按键检查通过：GUI 重复安装复用原对象，会话按键只转发一次，
  旧会话解绑不会影响新会话；另有 CPU 检查确认旧会话对象可释放。
- 141 个活动 Python 文件语法检查通过。没有运行全套回归、长期性能或录像测试。

## 数值比较的实际结论

基线来自本轮开始前的当前实现，不使用历史 `before_refactor` 替代。
每份短快照含 630 个数组，均通过存储地址稳定和有限值检查；初始与重置状态完全相同。
原比较器的 `rtol=1e-5, atol=1e-6` 没有修改，严格逐数组比较仍存在失败：

| 比较 | 完全相同数组 | 超容差数组 | 第 6 步匹配粒子的最大位置差 |
| --- | ---: | ---: | ---: |
| 原代码两次运行 | 399 | 60 | 3.37e-5 m |
| 重构前与重构后 | 397 | 49 | 1.12e-5 m |
| 重构后两次运行 | 400 | 50 | 9.70e-6 m |

该步的粒子匹配均为双向一一对应，原始数组差异包括粒子排序变化和小量浮点差异。
本次采样的重构差异没有超过原代码自身重复运行的对应最大位置差，但这些结果不能
证明逐位等价，也没有据此放宽容差或修改 Solver。详细原始记录保留在本地
`output/newton_pbf_refactor_checks/architecture_*.{json,npz}` 和 `compare_*.json`。

未调整用户的 `AGENTS.md` 或 `imgui.ini`，未提交 Git commit。
