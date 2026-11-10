当前仓库是克隆了最新原版的mimickit。

已经有适配最新稳定版本的：Newton引擎/warp dsl/mujoco warp。现在将要进行自定义流体的流固耦合求解器的接入。
具体可以参考另一个项目: 
1. 流体求解器。 D:\ComputerScience\PROJECT\WarpLab
2. 适配新newton的mimickit以及一些工具，一些修复等。 /D:/ComputerScience/PROJECT/MimicKit
3. 描述文档以及约定记录。D:\ComputerScience\PROJECT\MimicKit\NEWTON_PBF_SIMREADY_INTEGRATION_PLAN.md

主要参考其根目录下的PBF与simready，忽略demo文件夹，那个是归档过往代码。
1. 首先PBF流体程序将以自定义solver接入，在体积图方法下和刚体进行弱流固耦合，当前PBF已经实现此功能。
2. 其次SimRead资产管线需要接入，且需要扩展，不仅负责PBF相关，还需要适配承担人形导入的资产校验功能。
3. 最后会需要将所有上述资产管线与PBF的参数部分接入到Mimickit当前使用的Newton渲染器与GUI，让GUI可以调整流体参数，以及资产的加载选取，资产场景的组装等。
项目整体需要保持良好的工程结构、良好的代码可读性。

不破坏/减少破坏：当前Mimickit newton后端的架构、PBF py以及调用到的下属相关的主要求解器架构。

渲染器，资产管线可能需要小幅度架构结构重构与整理。

分析项目情况，给出整合建议，给出合理的分步骤实现计划。

注：文字描述可能会失准，项目文档/记录仅供参考，必须分析、核对实际代码。

详叙简述，暂不修改代码。