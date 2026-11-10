# Migration provenance

Source repository commit: `d37c96150f41941283028f76d71d06d213f82daf`.

Project-owned geometry, validation, baking and NVDB helpers were migrated from
`demos/sim_Ready`; aligned collision extraction, CoACD generation/cache and grouped
publication came from `demos/PBF_Rigid`. They are independently maintained here;
none of these packages is imported by this tool.

The copied Newton terrain generator is **not** included. Terrain uses the installed
Newton 1.7.0 public `newton.Mesh.create_terrain` API. Newton's original generator
is marked Copyright (c) 2025 The Newton Developers, Apache-2.0; its implementation
remains in the third-party package under its own license.

The bunny source is copied byte-for-byte from the existing demo asset. Its OBJ
header attribution, `https://github.com/mikedh/trimesh`, is retained. The local
asset has no additional license declaration; this migration does not assign a
new license or establish redistribution rights beyond the source information.

| Source / migrated model | SHA-256 at migration |
| --- | --- |
| `demos/sim_Ready/geometry.py` | `7c5a2d6ac3df84120c0f7d0e6bbe901ab5a8e19ddda186ded808e10c56e48372` |
| `demos/sim_Ready/validation.py` | `2bb8062cb7994260baf3770f0980c0819090c0dcde655a8a5bed6f36e7c42d77` |
| `demos/sim_Ready/sdf.py` | `e1079ffdd02e56f000a474ae6487e97efd15265eddfddf731551397d7fd3476e` |
| `demos/sim_Ready/nvdb_io.py` | `f6e463377270f4ef8e73b980577cf3c5899a1e2cc66167347508e9ba3334495d` |
| `demos/PBF_Rigid/sim_ready/geometry.py` | `72b7a50aa9b0c3e164c41bf3a28c9c070d13b1c679f71ebf1187478305f2df9f` |
| `demos/PBF_Rigid/sim_ready/rigid_assets.py` | `bafbeb02fe868227d865f6bd5b99dec71147c5c90c5823babb3388cff0f5ff8e` |
| `demos/PBF_Rigid/sim_ready/pipeline.py` | `f0743fe2089a30eb79fa50b00a82ae62d58d6134634c92f07cbe74cce3ed9efd` |
| `demos/PBF_Rigid/asset_io.py` | `c9e24809acd36f6351a571f44a84cb75b43bd76bc2f22f3c29be69b7a4f33023` |
| `Sim_Ready/Assets/bunny_flat_bottom.obj` | `5c2fc17c5e851b02dd5503ac50bd6456dbb09959d90492a784c3481cef22d529` |

Dependency pins: Newton 1.7.0, Warp 1.18.0, NumPy 2.4.6, trimesh 5.1.1,
manifold3d 3.5.4 and CoACD 1.0.14. Generated recipe manifests record installed
versions. Original published SDF/VM and collision data are preserved by verified
migration; a fresh bake is evaluated separately from bytewise migration parity.
