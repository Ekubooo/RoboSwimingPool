# Published reference scene

Run from the repository root in the existing WarpSim environment:

```powershell
python tools/prepare_rsp_assets.py
python mimickit/run.py --arg_file args/rsp_pbf_demo_args.txt
```

Preparation scans `data/assets_RSP/origin/**/source.yaml`. The initial recipes
cover the existing humanoid, pool with its suspended bunny, and three props.
They reuse the existing geometry, boundary and collision pipelines and the
`data/assets_RSP/pbf_play` cache. Original MimicKit assets remain untouched.

Complete packages are published under
`data/assets_RSP/published/<category>/<name>/<version>/asset.json`. Categories
are `articulations`, `props`, `containers`, and `environments`. The last category
is reserved; there is no new playground importer in this version. Packages
contain their physical models, visual/collision resources and boundary families,
including relative dependencies. Published versions are immutable: unchanged
recipes reuse verified packages; changed recipes require a new version.

`data/scenes/pbf_play.yaml` references explicit asset versions. The environment
configuration selects it through `scene_preset`; `pbf_asset_dir` now selects the
published root. Scene v1 validates the current container, humanoid and three-prop
layout. Changing rigid counts, object order, container geometry or robot type
belongs to the subsequent assembly work. The existing agent/checkpoint launch
arguments and fluid defaults remain authoritative.

Playback resolves published resources, imports the physical models, validates
the skeleton and local physical descriptions, then binds published boundaries
to the new runtime body IDs. It never prepares missing assets. Run records go to
`output/pbf_play/runs/<session>/scene_manifest.json`, outside published packages.

Targeted checks:

```powershell
python -m unittest discover -s tests -p test_rsp_publication.py
python tools/check_rsp_published_scene.py
```

The second check uses the existing checkpoint, executes 11 headless frames and
one reset, forbids preparation calls, and verifies published files are unchanged.
Neither command tests recording.
