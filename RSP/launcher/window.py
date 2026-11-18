"""ImGui selection panel hosted by the existing Newton Viewer."""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from RSP.sim_ready.catalog import AssetRef
from .assembly import asset_label
from .profiles import read_yaml, load_characters, character_for_scene, LaunchRequest


@dataclass
class PropChoice:
    enabled: bool
    count: int
    version: str


class SelectionPanel:
    def __init__(self, choices, presets, default_preset, *, characters=None):
        self.choices, self.presets = choices, presets
        self.preset_scenes = {name: read_yaml(path) for name, path in presets.items()}
        self.mode = "preset"
        self.preset = next(name for name, path in presets.items() if path == default_preset)
        self.characters = load_characters() if characters is None else characters
        self.character = character_for_scene(choices.template, self.characters).profile_id
        self.container = asset_label(AssetRef(**choices.template["container"]))
        defaults = {x["asset"]["name"]: asset_label(AssetRef(**x["asset"])) for x in choices.template["instances"]}
        self.props = {name: PropChoice(name in defaults, 1, defaults.get(name, next(iter(versions))))
                      for name, versions in choices.props.items()}
        self.placement, self.seed = "regular", 1234
        self.use_sphere = False
        self.layout = self.result = None
        self.cancelled = False
        self.status = "选择预设或自定义资产，然后点击开始仿真。"
        self.font = None
        self.scale = 1.
        self.previews = None

    def attach(self, viewer):
        if viewer.gui is None or not viewer.ui.is_available:
            raise RuntimeError("场景选择需要 Newton 的 ImGui 界面")
        from RSP.rendering.asset_preview import AssetPreviews
        self.previews = AssetPreviews(self.choices.assets)
        self.scale = viewer.ui.dpi_scale
        font_path = Path("C:/Windows/Fonts/msyh.ttc")
        if not font_path.is_file():
            raise RuntimeError("找不到中文界面字体：C:/Windows/Fonts/msyh.ttc")
        self.font = viewer.ui.io.fonts.add_font_from_file_ttf(str(font_path), 18. * self.scale)

    def detach(self):
        if self.previews is not None:
            self.previews.close()
            self.previews = None
        self.font = None  # The Viewer owns its font atlas.

    def cancel(self):
        self.cancelled = True

    def prepare_previews(self):
        self.previews.process_one()
        # Only cards visible in the upcoming UI frame renew queued requests.
        self.previews.pending.clear()

    def _color(self, ref):
        if ref.category == 'articulations':
            return (1., 0., 0.) if ref.name == 'humanoid' else None
        return next((tuple(x['color']) for x in self.choices.template['instances']
                     if x['asset']['name'] == ref.name), (.7, .7, .7))

    def _presets(self, imgui):
        from collections import Counter
        from .cards import card_grid, card, picture
        with card_grid(imgui, 'presets', self.scale) as opened:
            if not opened:
                return
            for name, scene in self.preset_scenes.items():
                counts = Counter(AssetRef(**x['asset']) for x in scene['instances'])
                entries = [(AssetRef(**scene['container']), 1), (AssetRef(**scene['character']), 1), *counts.items()]
                with card(imgui, name, 430, self.scale) as visible:
                    if not visible:
                        continue
                    if imgui.checkbox(name, self.preset == name)[0]:
                        self.preset = name
                    with card_grid(imgui, 'composition', self.scale*.4) as tiles:
                        if tiles:
                            for ref, count in entries:
                                imgui.table_next_column()
                                picture(imgui, self.previews, ref, self._color(ref),
                                        min(90*self.scale, imgui.get_content_region_avail().x))
                                imgui.text_wrapped(f'{ref.name} × {count}')
                    try:
                        character_for_scene(scene, self.characters)
                    except ValueError as error:
                        imgui.text_wrapped(str(error))

    def invalidate(self):
        self.layout = None
        self.status = "选择已更新，启动前将重新生成布局。"
        sphere = self.props.get("scene_sphere")
        if sphere is None or not sphere.enabled:
            self.use_sphere = False

    def generate(self):
        try:
            character = self.characters[self.character]
            character.require_available()
            counts = {self.choices.props[name][p.version]: p.count if p.enabled else 0
                      for name, p in self.props.items()}
            self.layout = self.choices.build(self.choices.containers[self.container], counts,
                mode=self.placement, seed=self.seed, use_sphere=self.use_sphere, character=character.asset)
            self.status = f"已生成 {len(self.layout['instances'])} 个刚体，随机种子 {self.seed}"
            return True
        except (OSError, ValueError, KeyError) as error:
            self.layout = None
            self.status = str(error)
            return False

    def request_start(self):
        try:
            if self.mode == "preset":
                choice = character_for_scene(self.preset_scenes[self.preset], self.characters)
                self.result = LaunchRequest(choice.profile_id, self.presets[self.preset])
            elif self.layout is not None or self.generate():
                self.characters[self.character].require_available()
                self.result = LaunchRequest(self.character, deepcopy(self.layout))
        except (ValueError, KeyError) as error:
            self.status = str(error)

    def select_character(self, profile_id):
        self.character = profile_id
        self.invalidate()

    @staticmethod
    def _combo(imgui, label, selected, options):
        names = list(options)
        changed, index = imgui.combo(label, names.index(selected), names)
        return changed, names[index]

    def _custom(self, imgui):
        from .cards import card_grid, card, asset_card
        for group, options in (('容器 × 1', self.choices.containers), ('人形 × 1', self.characters)):
            imgui.separator_text(group)
            with card_grid(imgui, group, self.scale) as opened:
                if not opened:
                    continue
                for key, option in options.items():
                    is_character = group == '人形 × 1'
                    ref = option.asset if is_character else option
                    with card(imgui, key, 245, self.scale) as visible:
                        if not visible:
                            continue
                        if ref is None:
                            imgui.text_wrapped(option.error or '资产不可用')
                            continue
                        selected = self.character == key if is_character else self.container == key
                        changed, _ = asset_card(imgui, self.previews, ref, self._color(ref),
                            option.label if is_character else key, selected, self.scale,
                            disabled=bool(option.error) if is_character else False)
                        if changed:
                            if is_character:
                                self.select_character(key)
                            else:
                                self.container = key
                                self.invalidate()
                        if is_character and option.error:
                            imgui.text_wrapped(option.error)
        imgui.separator_text('自由刚体')
        with card_grid(imgui, 'props', self.scale) as opened:
            if opened:
                for name, p in self.props.items():
                    with card(imgui, name, 300, self.scale) as visible:
                        if not visible:
                            continue
                        ref = self.choices.props[name][p.version]
                        changed, p.enabled = asset_card(imgui, self.previews, ref, self._color(ref),
                                                       name, p.enabled, self.scale)
                        if changed:
                            self.invalidate()
                        imgui.begin_disabled(not p.enabled)
                        changed, index = imgui.combo('数量', p.count-1, ['1', '2', '3'])
                        if changed:
                            p.count = index+1
                            self.invalidate()
                        changed, p.version = self._combo(imgui, '版本', p.version, self.choices.props[name])
                        if changed:
                            self.invalidate()
                        imgui.end_disabled()
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
            imgui.text_disabled('正交预览为模型默认姿态，并非策略出生姿态；勾选后点击开始仿真。')
            available = imgui.get_content_region_avail().y
            imgui.begin_child('selection_content', imgui.ImVec2(0, max(80*self.scale, available-105*self.scale)))
            try:
                if self.mode == 'preset':
                    self._presets(imgui)
                    summary = self.preset_scenes[self.preset]
                else:
                    self._custom(imgui)
                    summary = self.layout
                if summary is not None:
                    imgui.separator_text('实例 / 资产版本 / XYZ（米）')
                    for item in summary['instances']:
                        position = ', '.join(f'{x:.3f}' for x in item['position'])
                        imgui.text(f"{item['name']}   {asset_label(AssetRef(**item['asset']))}   {position}")
            finally:
                imgui.end_child()
            imgui.separator()
            if imgui.button("开始仿真"):
                self.request_start()
            imgui.same_line()
            if imgui.button("退出"):
                self.cancelled = True
            imgui.text_wrapped(self.status)
        finally:
            imgui.end()
            if self.font is not None:
                imgui.pop_font()
