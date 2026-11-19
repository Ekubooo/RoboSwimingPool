"""G1 publication and playback integration contracts; no training or recording."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "mimickit")]

from RSP.sim_ready.catalog import AssetRef, digest, resolve_asset
from RSP.sim_ready.pipelines.recipes import g1 as g1_recipe
from RSP.sim_ready.pipelines.publication import prepare_sources


class G1PublicationChecks(unittest.TestCase):
    def test_copy_preserves_model_and_dependencies(self):
        checks = ROOT / "output/g1_checks"
        checks.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=checks) as folder:
            model = g1_recipe.copy_model(Path(folder))
            self.assertEqual(digest(model), digest(g1_recipe.SOURCE_ROOT / "g1.xml"))
            for name, source in g1_recipe.source_files().items():
                target = model if name == "g1.xml" else Path(folder) / name
                self.assertEqual(digest(target), digest(source))

    def test_collision_scope_rejects_missing_or_extra_geometry(self):
        empty = [SimpleNamespace(name=n, shapes=()) for n in g1_recipe.EXCLUDED_LINKS]
        solid = [SimpleNamespace(name=str(i), shapes=[SimpleNamespace(kind="CAPSULE")] * (2 if i < 6 else 1))
                 for i in range(21)]
        asset = SimpleNamespace(bodies=empty + solid)
        g1_recipe.check_collision_scope(asset)
        empty[0].shapes = [SimpleNamespace(kind="SPHERE")]
        with self.assertRaisesRegex(ValueError, "31 bodies"):
            g1_recipe.check_collision_scope(asset)

    def test_published_scope_and_reuse(self):
        ref = AssetRef("articulations", "g1", "1")
        asset = resolve_asset(ref)
        self.assertEqual(len(asset.record["links"]), 21)
        self.assertEqual(set(asset.record["excluded_links"]), set(g1_recipe.EXCLUDED_LINKS))
        self.assertFalse(set(asset.record["links"]) & set(asset.record["excluded_links"]))
        before = digest(asset.root / "asset.json")
        with patch('RSP.sim_ready.pipelines.articulation.prepare_articulated_asset',
                   side_effect=AssertionError("Must reuse publication")):
            repeated, = prepare_sources(support_radius=.04, only=ref)
        self.assertEqual(repeated.root, asset.root)
        self.assertEqual(before, digest(asset.root / "asset.json"))
        for path in asset.record["links"].values():
            self.assertEqual(json.loads(asset.file(path).read_text())["kind"], "boundary")


class G1BindingChecks(unittest.TestCase):
    def test_coverage_and_exclusions(self):
        from RSP.simulation.scenes.build import PBFScene
        from RSP.sim_ready.lib.asset_validation import AssetValidationError
        physical = SimpleNamespace(name="test", bodies=[
            SimpleNamespace(name="solid", shapes=(object(),)), SimpleNamespace(name="empty", shapes=())])
        valid = dict(links={"solid": "boundary.json"}, excluded_links=["empty"])
        PBFScene.validate_published_links(physical, valid)
        for record in (dict(valid, excluded_links=[]), dict(valid, excluded_links=["empty", "empty"]),
                       dict(valid, excluded_links=["unknown"]), dict(valid, links={"unknown": "b"}),
                       dict(links={"empty": "b"}, excluded_links=["solid"])):
            with self.assertRaises((ValueError, AssetValidationError)):
                PBFScene.validate_published_links(physical, record)
        PBFScene.validate_published_links(physical, dict(links={"solid": "a", "empty": "b"}))

    def test_registration_uses_package_exclusions(self):
        from RSP.integrations.mimickit.scene import MimicKitScene
        scene = MimicKitScene.__new__(MimicKitScene)
        ref = AssetRef("articulations", "g1", "1")
        scene.spec = SimpleNamespace(character=ref)
        scene.resolved_assets = {ref: SimpleNamespace(record={"excluded_links": ["empty"]})}
        scene._object_assets, scene._skeletons, scene._excluded_links = {}, {}, {}
        scene.register_character(0, "skeleton")
        self.assertEqual(scene._excluded_links[0], ("empty",))
        with self.assertRaisesRegex(ValueError, "differ"):
            scene.register_character(1, "skeleton", ["wrong"])


class G1ProfileChecks(unittest.TestCase):
    def test_selector_switch_keeps_props_and_invalidates_layout(self):
        from RSP.integrations.mimickit.profiles import load_profile, load_presets, read_yaml, load_characters
        from RSP.app.ui.scene_choices import SceneChoices
        from RSP.app.ui.selection import SelectionPanel
        from dataclasses import replace
        profile = load_profile()
        panel = SelectionPanel(SceneChoices(read_yaml(profile.scene_preset)), load_presets(), profile.scene_preset)
        panel.mode = 'custom'
        for prop in panel.props.values():
            prop.enabled = False
        self.assertTrue(panel.generate())
        container = panel.container
        panel.select_character('g1')
        self.assertIsNone(panel.layout)
        self.assertEqual(panel.container, container)
        panel.request_start()
        self.assertEqual(panel.result.profile_id, 'g1')
        self.assertEqual(panel.result.scene['character']['name'], 'g1')
        self.assertEqual(panel.result.scene['instances'], [])
        panel.result = None
        panel.characters['g1'] = replace(panel.characters['g1'], error='missing model')
        panel.request_start()
        self.assertIsNone(panel.result)
        self.assertEqual(panel.status, 'missing model')
        panel.select_character('humanoid')
        panel.request_start()
        self.assertEqual(panel.result.profile_id, 'humanoid')

    def test_missing_g1_does_not_disable_default(self):
        from RSP.integrations.mimickit import profiles
        actual = profiles.project_file
        def missing(value):
            if 'g1_lafan_full' in str(value) and str(value).endswith('model.pt'):
                raise ValueError('missing G1 checkpoint')
            return actual(value)
        with patch.object(profiles, 'project_file', side_effect=missing):
            choices = profiles.load_characters()
        self.assertIsNone(choices['humanoid'].error)
        self.assertEqual(choices['g1'].error, 'missing G1 checkpoint')

    def test_profiles_keep_original_files(self):
        from RSP.integrations.mimickit.profiles import load_profile, read_yaml
        default, g1 = load_profile(), load_profile('g1')
        self.assertEqual(default.profile_id, 'humanoid')
        self.assertEqual(g1.model_file, ROOT / 'output/g1_lafan_full_20261107_r01/model.pt')
        self.assertEqual(default.engine_config, g1.engine_config)
        self.assertEqual(read_yaml(g1.scene_preset)['character']['name'], 'g1')
        self.assertEqual(read_yaml(default.scene_preset)['character']['name'], 'humanoid')
        original = read_yaml(g1.env_config)
        self.assertEqual(original['env_name'], 'task_dodgeball_g1_lafan')
        self.assertTrue(original['zero_center_action'])
        self.assertTrue(original['clip_position_targets_to_physical_limits'])


if __name__ == "__main__":
    unittest.main()
