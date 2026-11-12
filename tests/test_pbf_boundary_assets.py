"""Boundary artifact contracts; all generated fixtures stay under data/assets_RSP/tests."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import trimesh
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mimickit"))
OUTPUT = ROOT / "data/assets_RSP/tests/boundary"
wp.config.kernel_cache_dir = str(OUTPUT / "warp_cache")

from RSP.sim_ready.lib.storage import fingerprints as asset_io, nvdb_io
from RSP.sim_ready.lib.storage.store import AssetStore
from RSP.sim_ready.lib.volume import sdf
from RSP.sim_ready.lib.mesh import validation
from RSP.sim_ready.pipelines.boundary import prepare as assets
from RSP.sim_ready.pipelines.boundary.load import load_boundary_asset
from RSP.sim_ready.pipelines.boundary.schema import check_volume_map_layout
from RSP.sim_ready.pipelines import rigid
from RSP.sim_ready.config import BakeOptions
from RSP.sim_ready.lib.volume.pbf_volume_map import precompute_volume_map
from solvers.pbf import boundary_model


class BoundaryAssets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not wp.is_cuda_available():
            raise unittest.SkipTest("Boundary asset tests require CUDA")
        (OUTPUT / "tmp").mkdir(parents=True, exist_ok=True)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="assets-", dir=OUTPUT / "tmp")
        self.root = Path(self.temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(OUTPUT.resolve()))
        self.addCleanup(self.temporary.cleanup)
        self.mesh = trimesh.creation.box(extents=[.2, .16, .12])
        self.checked = validation.validate_solid(self.mesh)
        self.options = BakeOptions(voxel_size=.04, padding_voxels=3, max_support_radius=.06)

    def prepare(self, *, h=.06, q=3, mesh=None, options=None):
        mesh = self.mesh if mesh is None else mesh
        return assets.prepare_boundary_asset(mesh, validation.validate_solid(mesh), store=AssetStore(self.root),
                                             bake_options=options or self.options, support_radius=h,
                                             quadrature_points=q, provenance={"source_kind": "procedural_box"})

    def load(self, artifact, *, h=.06, q=3):
        return load_boundary_asset(artifact, support_radius=h, quadrature_points=q, device="cuda:0")

    def files(self):
        return {p.relative_to(self.root).as_posix(): (p.stat().st_mtime_ns, asset_io.file_digest(p))
                for p in self.root.rglob("*") if p.is_file()}

    def record(self, artifact):
        return json.loads(artifact.manifest_path.read_text(encoding="utf-8"))

    def test_cold_build_warm_reuse_and_read_only_load(self):
        with patch.object(assets, "precompute_volume_map", wraps=precompute_volume_map) as generate:
            first = self.prepare()
            self.assertEqual(generate.call_count, 1)
        before = self.files()
        with patch.object(assets, "precompute_volume_map", side_effect=AssertionError("unexpected generation")), \
                patch.object(sdf, "bake_rigid", side_effect=AssertionError("unexpected bake")), \
                patch.object(nvdb_io, "save_volumes", side_effect=AssertionError("unexpected write")):
            self.assertEqual(first, self.prepare())
            loaded = self.load(first)
        self.assertEqual(before, self.files())
        self.assertEqual(nvdb_io.grid_count(loaded.sdf_path), 2)
        self.assertEqual(self.record(first)["schema"], 3)
        self.assertNotIn("body_id", first.manifest_path.read_text(encoding="utf-8"))

    def test_integration_settings_share_sdf_without_overwriting_variants(self):
        first = self.prepare()
        original = self.files()
        with patch.object(sdf, "bake_rigid", side_effect=AssertionError("SDF should be reused")):
            smaller = self.prepare(h=.04)
            quadrature = self.prepare(q=4)
        self.assertEqual(len({first, smaller, quadrature}), 3)
        self.assertEqual(self.record(first)["sdf_sha256"], self.record(smaller)["sdf_sha256"])
        for path, values in original.items():
            self.assertEqual(values, self.files()[path])
        self.load(smaller, h=.04)
        self.load(quadrature, q=4)

    def test_support_growth_and_smallest_sufficient_selection(self):
        initial = self.prepare()
        larger = self.prepare(h=.10)
        self.assertNotEqual(self.record(initial)["sdf_sha256"], self.record(larger)["sdf_sha256"])
        with patch.object(sdf, "bake_rigid", side_effect=AssertionError("SDF should be reused")):
            small = self.prepare(h=.03)
            middle = self.prepare(h=.08)
        self.assertEqual(self.record(small)["bake_options"]["max_support_radius"], .06)
        self.assertEqual(self.record(middle)["bake_options"]["max_support_radius"], .10)

    def test_geometry_and_bake_settings_invalidate_sdf(self):
        initial = self.prepare()
        mesh = self.mesh.copy()
        mesh.apply_translation([.04, 0., 0.])
        changed = self.prepare(mesh=mesh)
        resolution = self.prepare(options=replace(self.options, voxel_size=.05))
        identities = {self.record(ref)["sdf_sha256"] for ref in (initial, changed, resolution)}
        self.assertEqual(len(identities), 3)

    def test_corrupt_map_is_rejected_by_load_and_rebuilt_by_prepare(self):
        artifact = self.prepare()
        volume_path = self.load(artifact).sdf_path
        volume_path.write_bytes(b"damaged cache")
        before = self.files()
        with self.assertRaisesRegex(ValueError, "Prepare the asset"):
            self.load(artifact)
        self.assertEqual(before, self.files())
        with patch.object(sdf, "bake_rigid", side_effect=AssertionError("valid SDF should be reused")), \
                patch.object(assets, "precompute_volume_map", wraps=precompute_volume_map) as generate:
            rebuilt = self.prepare()
            self.assertEqual(generate.call_count, 1)
        self.assertEqual(artifact, rebuilt)
        self.load(rebuilt)

    def test_corrupt_sdf_and_missing_manifest_are_rebuilt(self):
        first = self.prepare()
        sdf_file = next(self.root.glob("boundaries-v3/*/sdfs/*/sdf.nvdb"))
        sdf_file.write_bytes(b"damaged SDF")
        with patch.object(sdf, "bake_rigid", wraps=sdf.bake_rigid) as bake:
            self.prepare()
            self.assertEqual(bake.call_count, 1)
        first.manifest_path.unlink()
        with self.assertRaisesRegex(ValueError, "Prepare the asset"):
            self.load(first)
        rebuilt = self.prepare()
        self.load(rebuilt)

    def test_bad_manifest_and_request_are_rejected_without_writes(self):
        artifact = self.prepare()
        before = self.files()
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.load(artifact, h=.04)
        self.assertEqual(before, self.files())
        record = self.record(artifact)
        record["bounds"]["radius"] *= 2
        artifact.manifest_path.write_text(json.dumps(record), encoding="utf-8")
        damaged = self.files()
        with self.assertRaisesRegex(ValueError, "manifest"):
            self.load(artifact)
        self.assertEqual(damaged, self.files())
        self.prepare()
        self.load(artifact)

    def test_missing_visual_obj_does_not_block_runtime(self):
        artifact = self.prepare()
        loaded = self.load(artifact)
        expected = (loaded.bounds_center, loaded.bounds_half, loaded.bounds_radius)
        loaded.mesh_path.unlink()
        before = self.files()
        with patch.object(trimesh, "load", side_effect=AssertionError("runtime must not open visual mesh")):
            loaded = self.load(artifact)
            system = boundary_model.load_boundaries(artifact, [artifact], boundary_model.BoundaryLoadOptions(.06, "cuda:0", 3))
        self.assertEqual(expected, (loaded.bounds_center, loaded.bounds_half, loaded.bounds_radius))
        self.assertEqual(len(system.assets), 1)
        self.assertEqual(before, self.files())

    def test_equal_tile_counts_do_not_accept_different_coverage(self):
        artifact = self.prepare()
        distance = self.load(artifact).sdf
        tiles = wp.empty(distance.get_tile_count(), dtype=wp.vec3i, device=distance.device)
        distance.get_tiles(out=tiles)
        shifted = wp.array(tiles.numpy() + np.array([8, 0, 0]), dtype=wp.vec3i, device=distance.device)
        info = distance.get_grid_info()
        wrong = wp.Volume.allocate_by_tiles(shifted, transform=info.transform_matrix,
                                             translation=info.translation, bg_value=0.)
        self.assertEqual(distance.get_tile_count(), wrong.get_tile_count())
        with self.assertRaisesRegex(ValueError, "coverage"):
            check_volume_map_layout(distance, wrong)

    def test_shared_resources_keep_independent_instances_and_graph_buffers(self):
        artifact = self.prepare()
        system = boundary_model.load_boundaries(artifact, [artifact, artifact],
                                                boundary_model.BoundaryLoadOptions(.06, "cuda:0", 3))
        self.assertEqual(len(system.assets), 1)
        self.assertEqual(system.rigid_count, 2)
        model = SimpleNamespace(body_count=2, device=system.device,
                                body_com=wp.zeros(2, dtype=wp.vec3, device=system.device))
        state = SimpleNamespace(body_q=wp.array([[1., 0., 0., 0., 0., 0., 1.],
                                                 [0., 2., 0., 0., 0., 0., 1.]],
                                                dtype=wp.transform, device=system.device))
        system.bind_bodies(model, [1, 0])
        wp.load_module(module="solvers.pbf.boundary_model", device=system.device)
        with wp.ScopedCapture(device=system.device) as capture:
            system.sync_body_poses(state)
        wp.capture_launch(capture.graph)
        positions = system.device_data.transforms.numpy()[:, :3]
        np.testing.assert_array_equal(positions, [[0., 0., 0.], [0., 2., 0.], [1., 0., 0.]])
        pointer = system.device_data.transforms.ptr
        volumes = (system.assets[0].sdf.id, system.assets[0].volume_map.id)
        system.reset()
        np.testing.assert_array_equal(system.device_data.transforms.numpy()[:, :3], np.zeros((3, 3)))
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(system.device_data.transforms.numpy()[:, :3], positions)
        self.assertEqual(pointer, system.device_data.transforms.ptr)
        self.assertEqual(volumes, (system.assets[0].sdf.id, system.assets[0].volume_map.id))

    def test_failed_publication_preserves_existing_variant(self):
        original = self.prepare()
        before = self.files()
        with patch.object(AssetStore, "publish", side_effect=RuntimeError("injected publication failure")):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                self.prepare(q=4)
        self.assertEqual(before, self.files())
        self.load(original)

    def test_generic_rigid_build_respects_store_and_retains_obj_precision(self):
        store = AssetStore(self.root)
        with self.assertRaises(ValueError):
            rigid.build(self.mesh, ".", store=store, bake_options=self.options)
        self.assertEqual(self.files(), {})
        mesh_path, volume_path, hulls = rigid.build(self.mesh, "rigid/box", store=store,
                                                    bake_options=self.options, device="cuda:0")
        self.assertEqual(mesh_path, store.path("rigid/box.obj"))
        self.assertEqual(volume_path, store.path("rigid/box.nvdb"))
        self.assertEqual(hulls, ())
        self.assertEqual(nvdb_io.grid_count(volume_path), 1)
        persisted = trimesh.load(mesh_path, force="mesh", process=False)
        np.testing.assert_array_equal(persisted.vertices, self.mesh.vertices)
        np.testing.assert_array_equal(persisted.faces, self.mesh.faces)

    def test_invalid_inputs_create_no_assets_and_sources_are_protected(self):
        for h, q in ((0., 3), (float("nan"), 3), (.06, 0), (.06, True)):
            with self.subTest(h=h, q=q), self.assertRaises(ValueError):
                self.prepare(h=h, q=q)
        self.assertEqual(self.files(), {})
        with self.assertRaisesRegex(ValueError, "assets_RSP"):
            assets.prepare_boundary_asset(self.mesh, self.checked, store=AssetStore(ROOT / "data/assets_RSP/origin"),
                                          support_radius=.06, bake_options=self.options)
        artifact = self.prepare()
        paths = [str(p.relative_to(ROOT)) for p in self.root.rglob("*") if p.is_file()]
        result = subprocess.run(["git", "check-ignore", "--no-index", "--", *paths], cwd=ROOT,
                                capture_output=True, text=True, check=True)
        self.assertEqual(len(result.stdout.splitlines()), len(paths))
        for name in ("bunny_flat_bottom.obj", "spot_triangulated_good.obj"):
            result = subprocess.run(["git", "check-ignore", "--no-index", "--", f"data/assets_RSP/origin/{name}"],
                                    cwd=ROOT, capture_output=True)
            self.assertEqual(result.returncode, 1)


if __name__ == "__main__":
    unittest.main()
