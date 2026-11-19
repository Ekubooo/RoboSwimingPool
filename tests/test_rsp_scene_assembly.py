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
from RSP.simulation.scenes.spec import load_scene
from RSP.simulation.scenes.resolve import resolve_scene
from RSP.sim_ready.catalog import scan_published, AssetCatalog

SUPPORT_RADIUS = .04


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
            resolved = resolve_scene(spec, support_radius=SUPPORT_RADIUS)
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
                resolve_scene(self.parse(record), support_radius=SUPPORT_RADIUS)
        with self.assertRaisesRegex(ValueError, "boundary parameters"):
            resolve_scene(load_scene(), support_radius=2 * SUPPORT_RADIUS)

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
        from RSP.sim_ready.pipelines.recipes.container import description
        resolved = resolve_scene(load_scene(), support_radius=SUPPORT_RADIUS)
        bunny = trimesh.load(resolved.static.file("bunny.obj"), process=False)
        self.assertEqual(description([3, 2, 5], bunny), resolved.layout)
        # Check runtime collision assembly without constructing GPU resources.
        from RSP.simulation.scenes.build import add_collisions
        from RSP.sim_ready.lib.storage.hull_io import load_hulls
        import newton
        import numpy as np
        builder = newton.ModelBuilder()
        hulls = load_hulls(resolved.static.file("hulls.npz"))
        add_collisions(builder, resolved.layout, bunny, hulls)
        boxes = resolved.layout["boxes"]
        self.assertEqual(builder.shape_label, [*(f"pool_wall_{i}" for i in range(len(boxes))),
            *(f"pool_bunny_collision_{i}" for i in range(len(hulls))), "pool_bunny"])
        self.assertEqual(builder.shape_type, [newton.GeoType.BOX] * len(boxes)
            + [newton.GeoType.CONVEX_MESH] * len(hulls) + [newton.GeoType.MESH])
        np.testing.assert_allclose(builder.shape_scale[:len(boxes)], [box["half"] for box in boxes])
        np.testing.assert_allclose([pose.p for pose in builder.shape_transform[:len(boxes)]],
                                   [box["center"] for box in boxes])

    def test_packages_contain_only_selected_boundary_dependencies(self):
        resolved = resolve_scene(load_scene(), support_radius=SUPPORT_RADIUS)
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
