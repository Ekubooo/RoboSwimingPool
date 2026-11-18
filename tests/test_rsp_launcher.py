"""Targeted CPU checks for launcher configuration, placement and lifecycle."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'mimickit')]
from RSP.launcher.profiles import load_profile, load_presets, read_yaml
from RSP.launcher.assembly import SceneChoices
from RSP.launcher.application import scene_path
from dataclasses import replace
from RSP.sim_ready.catalog import AssetRef
from RSP.sim_ready.pipelines.scenes.spec import load_scene, parse_scene


class LauncherChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / 'output/launcher_checks').mkdir(parents=True, exist_ok=True)
        cls.profile = load_profile()
        cls.template = read_yaml(cls.profile.scene_preset)
        cls.choices = SceneChoices(cls.template)
        cls.container = AssetRef(**cls.template['container'])
        cls.counts = {AssetRef(**entry['asset']): 3 for entry in cls.template['instances']}

    def test_presets_parse_identically(self):
        for path in load_presets().values():
            self.assertEqual(load_scene(path), parse_scene(read_yaml(path)))

    def test_layout_modes_and_roles(self):
        import numpy as np
        for mode in ('regular', 'random'):
            result = self.choices.build(self.container, self.counts, mode=mode)
            self.assertEqual(result, self.choices.build(self.container, self.counts, mode=mode))
            self.assertEqual(len(result['instances']), 9)
            self.assertEqual(len({x['name'] for x in result['instances']}), 9)
            self.assertEqual(sum(x.get('role') == 'policy_sphere' for x in result['instances']), 1)
            boxes = [self.choices.obstacles(self.container)]
            for item in result['instances']:
                center, half = self.choices.bounds(AssetRef(**item['asset']))
                center = center + item['position']
                bounds = np.array([center-half, center+half])
                self.assertTrue(all(np.any((bounds[1] < b[0]) | (b[1] < bounds[0])) for b in boxes))
                boxes.append(bounds)
        self.assertNotEqual(self.choices.build(self.container, self.counts, mode='random', seed=1),
                            self.choices.build(self.container, self.counts, mode='random', seed=2))

    def test_empty_invalid_and_no_space(self):
        self.assertEqual(self.choices.build(self.container, {})['instances'], [])
        self.assertEqual(self.choices.build(self.container, {r: 0 for r in self.counts})['instances'], [])
        with self.assertRaises(ValueError):
            self.choices.build(self.container, {next(iter(self.counts)): 4})
        with self.assertRaises(ValueError):
            self.choices.build(self.container, {}, use_sphere=True)
        template = deepcopy(self.template)
        template['fluid'] = dict(start=[1, 1, 1], end=[1.1, 1.1, 1.1])
        with self.assertRaises(ValueError):
            SceneChoices(template).build(self.container, self.counts)

    def test_scene_snapshot_and_original_settings(self):
        self.assertEqual(self.profile.device, 'cuda:0')
        self.assertEqual(self.profile.seed, 1234)
        self.assertEqual(scene_path(self.profile.scene_preset, self.profile), self.profile.scene_preset)
        original = self.profile.env_config.read_bytes()
        with tempfile.TemporaryDirectory(dir=ROOT / 'output/launcher_checks') as tmp:
            path = scene_path(self.template, replace(self.profile, out_dir=Path(tmp)))
            self.assertEqual(load_scene(path), parse_scene(self.template))
            self.assertFalse(list(Path(tmp).rglob('env.yaml')))
        self.assertEqual(original, self.profile.env_config.read_bytes())


if __name__ == '__main__':
    unittest.main()
