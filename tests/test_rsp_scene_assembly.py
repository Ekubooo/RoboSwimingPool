"""Scene authoring and CPU preflight, using the prepared reference packages."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "mimickit")]
from RSP.sim_ready.pipelines.scenes.spec import load_scene
from RSP.sim_ready.pipelines.scenes.assembly import resolve_scene
from RSP.sim_ready.catalog import scan_published, AssetCatalog


class SceneAssemblyChecks(unittest.TestCase):
    def setUp(self):
        self.record = yaml.safe_load((ROOT / "data/scenes/pbf_play.yaml").read_text())
        root = ROOT / "output/scene_checks"
        root.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.directory.cleanup)

    def parse(self, record):
        path = Path(self.directory.name) / "scene.yaml"
        path.write_text(yaml.safe_dump(record))
        return load_scene(path)

    def test_invalid_roles_quaternions_and_state(self):
        for change in (lambda r: r["instances"][0].update(rotation=[0, 0, 0, 0]),
                       lambda r: r["instances"][0].update(role="policy_sphere"),
                       lambda r: r["instances"][0].update(linear_velocity=[float("nan"), 0, 0]),
                       lambda r: r.update(instances=[], policy_use_scene_sphere=True)):
            record = copy.deepcopy(self.record)
            change(record)
            with self.assertRaises(ValueError):
                self.parse(record)

    def test_four_presets_and_shared_assets(self):
        for name, count, size in (("pbf_play", 3, (3, 2, 5)), ("pbf_empty", 0, (3, 2, 5)),
                                  ("pbf_multi", 4, (3, 2, 5)), ("pbf_wide", 3, (4, 2, 5))):
            spec = load_scene(ROOT / f"data/scenes/{name}.yaml")
            resolved = resolve_scene(spec)
            self.assertEqual(len(spec.instances), count)
            self.assertEqual(tuple(resolved.layout["size"]), size)
            self.assertEqual(len(resolved.dynamic), len(set((spec.character, *spec.props))))

    def test_overlap_outside_and_fluid_preflight(self):
        for change in (lambda r: r["instances"][0].update(position=[-.1, .5, 1]),
                       lambda r: r["instances"][0].update(position=[.17, .5, 1], rotation=[0, 0, .3826834323650898, .9238795325112867]),
                       lambda r: r["instances"][0].update(position=r["instances"][1]["position"]),
                       lambda r: r.update(spawn_xy=[4, 1]),
                       lambda r: r["fluid"].update(end=[4, 2, 4])):
            record = copy.deepcopy(self.record)
            change(record)
            with self.assertRaises(ValueError):
                resolve_scene(self.parse(record))

    def test_catalog_browsing_and_resolution_cache(self):
        with patch("RSP.sim_ready.catalog.digest", side_effect=AssertionError("Browsing read payload")):
            self.assertGreaterEqual(len(scan_published()), 6)
        ref = load_scene().character
        catalog = AssetCatalog()
        asset = catalog.resolve(ref)
        with patch("RSP.sim_ready.catalog.digest", side_effect=AssertionError("Repeated payload verification")):
            self.assertIs(catalog.resolve(ref), asset)

    def test_container_metadata_matches_reference(self):
        import trimesh
        from RSP.sim_ready.pipelines.scenes.container_recipe import description
        from RSP.sim_ready.pipelines.scenes import pbf_play
        from unittest.mock import Mock
        resolved = resolve_scene(load_scene())
        bunny = trimesh.load(resolved.static.file("bunny.obj"), process=False)
        self.assertEqual(description([3, 2, 5], bunny), resolved.layout)
        # Compare authored collision calls without constructing GPU resources.
        from RSP.sim_ready.pipelines.scenes.container_recipe import add_collisions
        from RSP.sim_ready.lib.storage.hull_io import load_hulls
        import newton
        import numpy as np
        original, current = newton.ModelBuilder(), newton.ModelBuilder()
        hulls = load_hulls(resolved.static.file("hulls.npz"))
        pbf_play.add_static_collisions(original, bunny, hulls)
        add_collisions(current, resolved.layout, bunny, hulls)
        np.testing.assert_equal(original.shape_type, current.shape_type)
        np.testing.assert_equal(original.shape_scale, current.shape_scale)
        np.testing.assert_equal(original.shape_transform, current.shape_transform)

    def test_packages_contain_only_selected_boundary_dependencies(self):
        resolved = resolve_scene(load_scene())
        for asset in (resolved.static, *resolved.dynamic.values()):
            roots = [asset.record["boundary"]] if "boundary" in asset.record else list(asset.record["links"].values())
            expected = set()
            for relative in roots:
                expected.add(relative)
                path = asset.file(relative)
                record = json.loads(path.read_text())
                for dependency in record["files"].values():
                    expected.add((path.parent / dependency).resolve().relative_to(asset.root).as_posix())
            actual = {name for name in asset.record["files"] if name.startswith("boundaries/")}
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
