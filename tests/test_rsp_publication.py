"""Focused CPU checks for scene selection and immutable publication."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "mimickit")]
from RSP.sim_ready.catalog import AssetRef, scan_published, resolve_asset, contained
from RSP.sim_ready.pipelines.publication import publish_package, scan_sources
from RSP.simulation.scenes.spec import load_scene


CHECK_ROOT = ROOT / "output" / "scene_checks"
CHECK_ROOT.mkdir(parents=True, exist_ok=True)


class PublicationChecks(unittest.TestCase):
    def test_reference_scene_and_invalid_instances(self):
        self.assertEqual(load_scene().spawn_xy, (.65, 1.))
        baseline = yaml.safe_load((ROOT / "data/scenes/pbf_play.yaml").read_text())
        with tempfile.TemporaryDirectory(dir=CHECK_ROOT) as folder:
            path = Path(folder) / "scene.yaml"
            for change in (lambda r: r.update(unknown=True),
                           lambda r: r["instances"][1].update(name="light_box"),
                           lambda r: r["instances"][0].update(position=[float("nan"), 0, 0]),
                           lambda r: r.update(policy_use_scene_sphere=True, instances=[])):
                record = json.loads(json.dumps(baseline))
                change(record)
                path.write_text(yaml.safe_dump(record))
                with self.assertRaises(ValueError):
                    load_scene(path)

    def test_publish_reuse_conflict_and_corruption(self):
        ref = AssetRef("props", "box", "1")
        def populate(stage):
            (stage / "model.xml").write_text("model")
            return {"model": "model.xml"}
        with tempfile.TemporaryDirectory(dir=CHECK_ROOT) as folder:
            asset = publish_package(ref, {"recipe_digest": "a"}, populate, root=folder)
            unused = Mock(side_effect=AssertionError("must reuse"))
            publish_package(ref, {"recipe_digest": "a"}, unused, root=folder)
            unused.assert_not_called()
            self.assertEqual(list(scan_published(folder)), [ref])
            with self.assertRaisesRegex(ValueError, "version changed"):
                publish_package(ref, {"recipe_digest": "b"}, unused, root=folder)
            asset.file("model.xml").write_text("changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                resolve_asset(ref, folder)

    def test_failed_publication_and_path_escape(self):
        ref = AssetRef("props", "box", "1")
        def fail(stage):
            (stage / "partial").write_text("incomplete")
            raise RuntimeError("preparation failed")
        with tempfile.TemporaryDirectory(dir=CHECK_ROOT) as folder:
            with self.assertRaises(RuntimeError):
                publish_package(ref, {"recipe_digest": "a"}, fail, root=folder)
            self.assertEqual(scan_published(folder), {})
            self.assertFalse((Path(folder) / ref.relative).exists())
            for relative in ("../outside", "C:/outside"):
                with self.assertRaises(ValueError):
                    contained(folder, relative)
            with self.assertRaisesRegex(ValueError, "not published"):
                resolve_asset(ref, folder)

    def test_source_duplicates_and_missing_dependencies(self):
        with tempfile.TemporaryDirectory(dir=CHECK_ROOT) as folder:
            root = Path(folder)
            source = dict(schema=1, identity=asdict(AssetRef("props", "box", "1")), recipe="reference_prop", inputs=[])
            for name in ("one", "two"):
                (root / name).mkdir()
                (root / name / "source.yaml").write_text(yaml.safe_dump(source))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                scan_sources(root)
            (root / "two/source.yaml").unlink()
            source["inputs"] = ["missing.xml"]
            (root / "one/source.yaml").write_text(yaml.safe_dump(source))
            with self.assertRaisesRegex(ValueError, "Missing"):
                scan_sources(root)

    def test_nested_boundary_references_and_catalog_identity(self):
        ref = AssetRef("props", "box", "1")
        with tempfile.TemporaryDirectory(dir=CHECK_ROOT) as folder:
            def bad_boundary(stage):
                (stage / "boundary.json").write_text(json.dumps(
                    {"kind": "boundary", "files": {"mesh": "../outside.obj"}}))
                return {}
            with self.assertRaisesRegex(ValueError, "escapes"):
                publish_package(ref, {"recipe_digest": "a"}, bad_boundary, root=folder)
            def populate(stage):
                (stage / "model.xml").write_text("model")
                return {"model": "model.xml"}
            asset = publish_package(ref, {"recipe_digest": "a"}, populate, root=folder)
            path = asset.root / "asset.json"
            record = json.loads(path.read_text())
            record["identity"]["name"] = "other"
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "misplaced"):
                scan_published(folder)
            asset.file("model.xml").unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                resolve_asset(ref, folder)


if __name__ == "__main__":
    unittest.main()
