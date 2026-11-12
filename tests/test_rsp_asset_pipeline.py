"""Output protection and the split convex pipeline's persisted behavior."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]

from RSP.sim_ready.paths import ASSET_ROOT, ORIGIN_ROOT, TEST_ROOT
from RSP.sim_ready.lib.mesh import geometry, convex
from RSP.sim_ready.lib.storage.fingerprints import mesh_digest
from RSP.sim_ready.lib.storage.store import AssetStore
from RSP.sim_ready.lib.storage import publish
from RSP.sim_ready.pipelines import collision


class AssetPipeline(unittest.TestCase):
    def setUp(self):
        directory = TEST_ROOT / "pipeline"
        directory.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="case-", dir=directory)
        self.addCleanup(self.temporary.cleanup)
        self.store = AssetStore(self.temporary.name)

    def test_repository_output_roots_protect_origin_and_other_inputs(self):
        invalid = (ASSET_ROOT, ORIGIN_ROOT, ORIGIN_ROOT / "new-output", ROOT / "output/new-assets",
                   ROOT / "data/assets", ASSET_ROOT / "pbf_play/../origin/nested")
        before = sorted(p.relative_to(ORIGIN_ROOT) for p in ORIGIN_ROOT.rglob("*"))
        for path in invalid:
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "outside origin"):
                AssetStore(path)
        self.assertEqual(before, sorted(p.relative_to(ORIGIN_ROOT) for p in ORIGIN_ROOT.rglob("*")))
        self.assertEqual(AssetStore(ASSET_ROOT / "pbf_play").root, ASSET_ROOT / "pbf_play")

    def test_explicit_external_roots_remain_supported(self):
        external = Path(tempfile.gettempdir()).resolve() / "rsp-store-contract-external"
        self.assertFalse(external.is_relative_to(ROOT))
        existed = external.exists()
        store = AssetStore(external)
        self.assertEqual(store.path("collision/hulls.npz"), external / "collision/hulls.npz")
        self.assertEqual(existed, external.exists())

    def test_escape_and_publication_targets_are_checked_before_any_write(self):
        for parts in (("../../../../origin/overwrite.obj",), (str(ORIGIN_ROOT / "overwrite.obj"),),
                      ("..", "another-store")):
            with self.subTest(parts=parts), self.assertRaises(ValueError):
                self.store.path(*parts)
        source = self.store.path("staged.txt")
        source.write_text("new", encoding="utf-8")
        safe = self.store.path("published.txt")
        with self.assertRaises(ValueError):
            self.store.publish([(source, safe), (source, ORIGIN_ROOT / "overwrite.obj")])
        self.assertFalse(safe.exists())
        with self.assertRaises(ValueError):
            self.store.recipe_directory("collision", {}, name="../escape")
        with self.assertRaises(ValueError):
            with self.store.staging(ORIGIN_ROOT / "escape"):
                self.fail("must not enter source staging")
        with self.assertRaises(ValueError):
            with self.store.staging(self.store.path("artifact"), prefix="../../../escape-"):
                self.fail("must not accept a staging prefix containing a path")

    def test_group_publication_rolls_back_and_removes_staging_files(self):
        destination = self.store.path("artifact")
        destination.mkdir()
        for name in ("data", "manifest"):
            (destination / name).write_text("old", encoding="utf-8")
        replace = publish.os.replace

        def fail_manifest(source, target):
            if Path(target) == destination / "manifest":
                raise OSError("injected final publication failure")
            return replace(source, target)

        with self.assertRaisesRegex(OSError, "injected"):
            with self.store.staging(destination) as stage:
                for name in ("data", "manifest"):
                    (stage / name).write_text("new", encoding="utf-8")
                with patch.object(publish.os, "replace", side_effect=fail_manifest):
                    self.store.publish([(stage / name, destination / name) for name in ("data", "manifest")])
        self.assertEqual({p.name: p.read_text() for p in destination.iterdir()}, {"data": "old", "manifest": "old"})
        self.assertEqual(list(self.store.root.iterdir()), [destination])

    def test_read_only_loader_does_not_import_build_algorithms(self):
        code = """
import sys
from RSP.sim_ready.pipelines.boundary.load import load_boundary_asset
for name in ('RSP.sim_ready.lib.volume.sdf', 'RSP.sim_ready.lib.volume.pbf_volume_map',
             'RSP.sim_ready.lib.mesh.convex', 'RSP.sim_ready.pipelines.boundary.prepare', 'coacd'):
    assert name not in sys.modules, name
"""
        result = subprocess.run([sys.executable, "-I", "-c", code], cwd=ROOT.parent,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_editable_package_imports_outside_checkout_with_unchanged_paths(self):
        code = """
from importlib.metadata import distribution
import json
from pathlib import Path
import RSP.sim_ready
from RSP.sim_ready.paths import PROJECT_ROOT, ASSET_ROOT, ORIGIN_ROOT, PBF_PLAY_ROOT, TEST_ROOT

dist = distribution('roboswimmingpool-rsp')
print(json.dumps(dict(source=str(Path(RSP.sim_ready.__file__).resolve()),
                     project=str(PROJECT_ROOT), assets=str(ASSET_ROOT), origin=str(ORIGIN_ROOT),
                     play=str(PBF_PLAY_ROOT), tests=str(TEST_ROOT),
                     packages=dist.read_text('top_level.txt').split(), requires=dist.requires)))
"""
        result = subprocess.run([sys.executable, "-I", "-c", code], cwd=ROOT.parent,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        installed = json.loads(result.stdout)
        self.assertEqual(Path(installed["source"]), ROOT / "RSP/sim_ready/__init__.py")
        self.assertEqual(Path(installed["project"]), ROOT)
        self.assertEqual(Path(installed["assets"]), ASSET_ROOT)
        self.assertEqual(Path(installed["origin"]), ORIGIN_ROOT)
        self.assertEqual(Path(installed["play"]), ASSET_ROOT / "pbf_play")
        self.assertEqual(Path(installed["tests"]), TEST_ROOT)
        self.assertEqual(installed["packages"], ["RSP"])
        requirements = (ROOT / "RSP/sim_ready/requirements.txt").read_text(encoding="utf-8")
        expected = {line.strip() for line in requirements.splitlines() if line.strip() and not line.startswith("#")}
        self.assertEqual(set(installed["requires"]), expected)

    def test_convex_compute_cache_format_reuse_and_corruption_behavior(self):
        first = trimesh.creation.box(extents=[.3, .1, .12])
        second = trimesh.creation.box(extents=[.1, .3, .12])
        second.apply_translation([-.1, .1, 0.])
        mesh = geometry.union([first, second])
        vertices, faces = mesh.vertices.copy(), mesh.faces.copy()
        expected = convex.decompose_convex(mesh)
        self.assertEqual(len(expected), 2)
        expected_errors = convex.measure_hull_error(mesh, expected)
        generated = collision.prepare_hulls(mesh, "concave", self.store)
        for actual, reference in zip(generated, expected, strict=True):
            np.testing.assert_array_equal(actual.vertices, reference.vertices)
            np.testing.assert_array_equal(actual.faces, reference.faces)
        np.testing.assert_array_equal(mesh.vertices, vertices)
        np.testing.assert_array_equal(mesh.faces, faces)

        recipe = dict(version=1, geometry=mesh_digest(mesh), backend=convex.COACD_VERSION, options=convex.COACD_OPTIONS)
        legacy_key = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:20]
        directory = self.store.path("collision", f"concave-{legacy_key}")
        metrics = json.loads((directory / "metrics.json").read_text())
        self.assertEqual({key: metrics[key] for key in expected_errors}, expected_errors)
        path = directory / "hulls.npz"
        with np.load(path, allow_pickle=False) as saved:
            self.assertEqual(saved["count"].dtype, np.int32)
            self.assertEqual(saved["vertices_0"].dtype, np.float64)
            self.assertEqual(saved["faces_0"].dtype, np.int32)
        before = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in directory.iterdir()}
        with patch.object(collision, "decompose_convex", side_effect=AssertionError("cache must be reused")):
            reused = collision.prepare_hulls(mesh, "concave", self.store)
            np.testing.assert_array_equal(reused[0].vertices, expected[0].vertices)
            self.assertEqual(before, {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in directory.iterdir()})
            # Diagnostics are not a new integrity gate for this existing format.
            (directory / "metrics.json").unlink()
            collision.prepare_hulls(mesh, "concave", self.store)
            path.write_bytes(b"damaged cache")
            with self.assertRaises(ValueError):
                collision.prepare_hulls(mesh, "concave", self.store)
            self.assertEqual(path.read_bytes(), b"damaged cache")


if __name__ == "__main__":
    unittest.main()
