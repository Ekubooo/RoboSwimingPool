当前仓库是克隆了最新mimickit，在Newton引擎层初步接入自定流体求解器、完善了资产管线能力。

已经有适配最新稳定版本的 Newton引擎/warp dsl/mujoco warp。当前使用conda环境 warpsim。

接入的实验性实现可以参考另其他的两个项目: 
1. 流体求解器。 D:\ComputerScience\PROJECT\WarpLab 。流体项目主要参考其根目录下的PBF与simready。忽略demo文件夹，是归档过往代码。
2. 适配新newton的mimickit及其工具、mimickit newton修复。D:/ComputerScience/PROJECT/MimicKit
3. 过往的项目结构描述文档，以及粗略的约定。D:\ComputerScience\PROJECT\MimicKit\NEWTON_PBF_SIMREADY_INTEGRATION_PLAN.md

当前状态与任务：
1. PBF流体程序当前以自定义solver接入，在volume map方法下和刚体进行弱流固双向耦合。
2. 当前Newton引擎的自定义solver这一后端已经基本完备，资产管线能力与管线定义也基本完备，上层由临时胶水层进行组合组装。接下来准备逐步拆除胶水层，以及渲染器，完善程序整体流程。
3. 当前运行目标，主要是运行mimickit的SMP路径下的躲避球task相关，加载已有RL策略进入流体环境，以观察人形机器人行为表现。主要以视觉呈现、程序运行为主，暂不考虑机器学习训练相关，暂不考虑多环境运行。 


重要信息：
1. 不破坏/减少破坏/非必要不破坏：当前Mimickit newton后端的架构、PBF算法求解器以及调用到的下属相关的主要求解器方法与架构、以及已完成的资产管线架构安排。
2. 文字描述可能会失准，所有项目文档、记录仅供参考，必须分析核对实际代码。
3. 不参考已有代码，仅从newton引擎规范、newton相关开源库、其他开源机器人仿真软件等角度考虑架构设计与代码实现。也即以各大开源库的做法与要求为标准，分析给出建议。