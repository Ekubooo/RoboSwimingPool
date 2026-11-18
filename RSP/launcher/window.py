"""ImGui selection panel hosted by the existing Newton Viewer."""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from RSP.sim_ready.catalog import AssetRef
from .assembly import asset_label
from .profiles import read_yaml


@dataclass
class PropChoice:
    enabled: bool
    count: int
    version: str


class SelectionPanel:
    def __init__(self, choices, presets, default_preset):
        self.choices, self.presets = choices, presets
        self.preset_scenes = {name: read_yaml(path) for name, path in presets.items()}
        self.mode = "preset"
        self.preset = next(name for name, path in presets.items() if path == default_preset)
        self.container = asset_label(AssetRef(**choices.template["container"]))
        defaults = {x["asset"]["name"]: asset_label(AssetRef(**x["asset"])) for x in choices.template["instances"]}
        self.props = {name: PropChoice(name in defaults, 1, defaults.get(name, next(iter(versions))))
                      for name, versions in choices.props.items()}
        self.placement, self.seed = "regular", 1234
        self.use_sphere = False
        self.layout = self.result = None
        self.cancelled = False
        self.status = "预设直接运行；自定义仅使用已发布资产。"
        self.font = None
        self.scale = 1.

    def attach(self, viewer):
        if viewer.gui is None or not viewer.ui.is_available:
            raise RuntimeError("场景选择需要 Newton 的 ImGui 界面")
        self.scale = viewer.ui.dpi_scale
        font_path = Path("C:/Windows/Fonts/msyh.ttc")
        if not font_path.is_file():
            raise RuntimeError("找不到中文界面字体：C:/Windows/Fonts/msyh.ttc")
        self.font = viewer.ui.io.fonts.add_font_from_file_ttf(str(font_path), 18. * self.scale)

    def detach(self):
        self.font = None  # The Viewer owns its font atlas.

    def invalidate(self):
        self.layout = None
        self.status = "选择已更新，启动前将重新生成布局。"
        sphere = self.props.get("scene_sphere")
        if sphere is None or not sphere.enabled:
            self.use_sphere = False

    def generate(self):
        try:
            counts = {self.choices.props[name][p.version]: p.count if p.enabled else 0
                      for name, p in self.props.items()}
            self.layout = self.choices.build(self.choices.containers[self.container], counts,
                mode=self.placement, seed=self.seed, use_sphere=self.use_sphere)
            self.status = f"已生成 {len(self.layout['instances'])} 个刚体，随机种子 {self.seed}"
            return True
        except (OSError, ValueError, KeyError) as error:
            self.layout = None
            self.status = str(error)
            return False

    def request_start(self):
        if self.mode == "preset":
            self.result = self.presets[self.preset]
        elif self.layout is not None or self.generate():
            self.result = deepcopy(self.layout)

    @staticmethod
    def _combo(imgui, label, selected, options):
        names = list(options)
        changed, index = imgui.combo(label, names.index(selected), names)
        return changed, names[index]

    def _custom(self, imgui):
        changed, self.container = self._combo(imgui, "容器 × 1", self.container, self.choices.containers)
        if changed:
            self.invalidate()
        imgui.text("人形 × 1：默认 humanoid / 当前 SMP 策略")
        imgui.text_disabled("出生状态、水域及物理参数沿用当前 demo")
        imgui.separator()
        for name, p in self.props.items():
            imgui.push_id(name)
            changed, p.enabled = imgui.checkbox(name, p.enabled)
            if changed:
                self.invalidate()
            imgui.same_line()
            imgui.begin_disabled(not p.enabled)
            imgui.set_next_item_width(100 * self.scale)
            changed, index = imgui.combo("数量", p.count-1, ["1", "2", "3"])
            if changed:
                p.count = index+1
                self.invalidate()
            changed, p.version = self._combo(imgui, "资产版本", p.version, self.choices.props[name])
            if changed:
                self.invalidate()
            imgui.end_disabled()
            imgui.pop_id()
        modes = ["regular", "random"]
        changed, index = imgui.combo("摆放方式", modes.index(self.placement), ["规则摆放", "随机摆放"])
        if changed:
            self.placement = modes[index]
            self.invalidate()
        imgui.same_line()
        if imgui.button("生成布局"):
            self.generate()
        imgui.same_line()
        if imgui.button("重新随机"):
            self.placement = "random"
            self.seed += 1
            self.generate()
        sphere = self.props.get("scene_sphere")
        imgui.begin_disabled(sphere is None or not sphere.enabled)
        changed, self.use_sphere = imgui.checkbox("策略使用场景观测球", self.use_sphere)
        if changed:
            self.invalidate()
        imgui.end_disabled()

    def draw(self, imgui):
        if self.font is not None:
            imgui.push_font(self.font, 18. * self.scale)
        viewport = imgui.get_main_viewport()
        imgui.set_next_window_pos(viewport.work_pos)
        imgui.set_next_window_size(viewport.work_size)
        flags = (imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_resize
                 | imgui.WindowFlags_.no_collapse | imgui.WindowFlags_.no_saved_settings)
        visible, _ = imgui.begin("场景选择", flags=flags)
        try:
            if not visible:
                return
            if imgui.radio_button("使用预设", self.mode == "preset"):
                self.mode = "preset"
            imgui.same_line()
            if imgui.radio_button("自定义装配", self.mode == "custom"):
                self.mode = "custom"
            if self.mode == "preset":
                _, self.preset = self._combo(imgui, "预设", self.preset, self.presets)
                summary = self.preset_scenes[self.preset]
            else:
                self._custom(imgui)
                summary = self.layout
            imgui.separator()
            if imgui.button("开始仿真"):
                self.request_start()
            imgui.same_line()
            if imgui.button("退出"):
                self.cancelled = True
            imgui.text_wrapped(self.status)
            imgui.separator()
            imgui.text("实例 / 资产版本 / XYZ（米）")
            if summary is not None:
                for item in summary["instances"]:
                    position = ", ".join(f"{x:.3f}" for x in item["position"])
                    imgui.text(f"{item['name']}   {asset_label(AssetRef(**item['asset']))}   {position}")
        finally:
            imgui.end()
            if self.font is not None:
                imgui.pop_font()
