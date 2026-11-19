# Application and playback

`application.py` owns the main window and the selection-to-session transition.
`playback.py` binds session callbacks and advances the policy/environment through
the session interface; `ui/` owns panels, shortcuts and queued CPU commands.
The same playback binding serves the legacy MimicKit entry point. Unbinding
removes its callbacks without closing the borrowed environment or window.

## Asset selection previews

`python tools/run_rsp.py` opens the existing Newton window. Cards edit the
selection only; **Start simulation** submits it. Presets reuse the asset images
as a composition, while custom mode retains version, quantity and layout rules.

- `RSP/rendering/preview/geometry.py`: reads published display geometry with Newton's MJCF
  importer on CPU; no finalized simulation model, strategy or asset baking.
- `ui/cards.py`: draws cards and requests images; no geometry loading in UI callbacks.
- `RSP/rendering/preview/textures.py`: renders fixed orthographic thumbnails into GL
  textures owned by this selection session. No PNG or preview package is written.
- `application.select_scene`: processes one visible-card request per frame and
  owns cleanup. Window close and Escape are deferred until the frame ends.

The camera looks from +X/-Y/+Z (right-front azimuth 45 degrees, elevation 45
degrees). Thumbnails show the model's authored default pose, not a sampled GSI
spawn. G1 retains authored display colors; humanoid and props use demo colors.
Containers display their frame and the published bunny mesh.

Temporary geometry, buffers, shader and FBO objects are released after each
render. Cached textures live only until selection ends. Start, Cancel, Escape,
window close and setup failure all remove callbacks and release preview resources
before simulation construction or context destruction. A failed thumbnail shows
a placeholder without altering runtime asset checks.

Focused checks (active conda environment, project root):

```powershell
python -m unittest discover -s tests -p "test_rsp_launcher*.py"
python -m unittest discover -s tests -p "test_rsp_previews.py"
```

The latter includes a real hidden GL window and in-memory pixel checks; it writes
no preview images. Interactive layout acceptance is a separate check.
