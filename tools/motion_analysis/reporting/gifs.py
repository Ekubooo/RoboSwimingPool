"""Record saved motions through Newton at exact frame times, with one fixed crop.

This is an export adapter, not another motion player or diffusion implementation.
It uses one GL scene for all clips and never samples or fits a PCA model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..paths import DEFAULT_CASE, ROOT, ensure_animation_imports

from .metrics import body_positions, load_character
from ..core.collection import load_collection
from ..player.environment import _environment_types
from ..player.controls import PlaybackControls
from ..player.view import FixedCamera, ViewLayout
from ..data.storage import atomic_write


@dataclass(frozen=True)
class Recording:
    name: str
    session: object
    group: str
    clip: int
    k: int | None

    @property
    def frames(self) -> np.ndarray:
        return (self.session.source.frames if self.k is None else
                self.session.reconstruct(tuple(range(self.k))))


def recording_plan(manifest: Path) -> list[Recording]:
    """Use independent PCA for compression GIFs, native frames for stage GIFs."""
    recordings = []
    for group in ("reference", "generated"):
        analysis = load_collection(manifest, group=group, basis="independent")
        for clip in range(1 if group == "reference" else 4):
            session = analysis.for_clip(clip)
            recordings.append(Recording(f"{group}_{clip:02d}_full", session, group, clip, None))
            if clip == 0:
                for k in (1, 2, 3):
                    recordings.append(Recording(f"{group}_{clip:02d}_K{k}", session, group, clip, k))
    for step in (49, 39, 29, 19, 9, 0):
        group = f"stage_{step}"
        session = load_collection(manifest, group=group).for_clip(0)
        recordings.append(Recording(f"stage_t{step:02d}_sample00", session, group, 0, None))
    return recordings


def fixed_crop(viewer, points: np.ndarray) -> tuple[int, int, int, int]:
    """Project a single union of all FK body bounds, including geometry margins."""
    view = viewer.camera.get_view_matrix().reshape(4, 4).T
    projection = viewer.camera.get_projection_matrix().reshape(4, 4).T
    corners = np.array([[x, y, z] for x in (-.18, .18) for y in (-.18, .18) for z in (-.18, .18)])
    expanded = (points[:, None, :] + corners).reshape(-1, 3)
    homogeneous = np.column_stack([expanded, np.ones(len(expanded))])
    projected = homogeneous @ view.T @ projection.T
    ndc = projected[:, :2] / projected[:, 3:4]
    width, height = viewer.renderer._screen_width, viewer.renderer._screen_height
    pixels = (ndc * [.5, -.5] + .5) * [width, height]
    low, high = pixels.min(axis=0), pixels.max(axis=0)
    center = (low + high) / 2
    half = np.ceil(max(high - low) / 2 + 10)
    bounds = np.r_[np.floor(center - half), np.ceil(center + half)].astype(int)
    if bounds[0] < 0 or bounds[1] < 0 or bounds[2] > width or bounds[3] > height:
        raise ValueError(f"Common actor crop {bounds.tolist()} exceeds the {width}x{height} viewport.")
    return tuple(int(value) for value in bounds)


def _encode(images: list[Image.Image], path: Path, fps: float) -> dict:
    # GIF has centisecond timing: alternate 30/40 ms instead of silently running at 33.3 FPS.
    boundaries = np.rint(np.arange(len(images) + 1) * 100 / fps).astype(int)
    durations = (np.diff(boundaries) * 10).tolist()
    # One palette per clip avoids frame-to-frame quantization flicker.
    atlas = Image.new("RGB", (images[0].width * len(images), images[0].height))
    for index, frame in enumerate(images):
        atlas.paste(frame, (index * frame.width, 0))
    palette = atlas.quantize(colors=128)
    encoded = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in images]
    stream = io.BytesIO()
    encoded[0].save(stream, format="GIF", save_all=True, append_images=encoded[1:],
                    duration=durations, loop=0, disposal=2, optimize=False)
    payload = stream.getvalue()
    atomic_write(path, payload)
    images[len(images) // 2].save(path.with_suffix(".png"))
    return {"sha256": hashlib.sha256(payload).hexdigest(), "frame_times_s": (np.arange(len(images)) / fps).tolist(),
            "requested_frames": len(images), "durations_ms": durations,
            "encoded_duration_s": sum(durations) / 1000, "size": list(images[0].size)}


def record_case(case: Path, device: str = "cuda:0") -> None:
    ensure_animation_imports()
    import torch
    from envs.env_builder import load_configs

    case = case.resolve()
    data = case / "assets/data"
    output = case / "assets/gifs"
    output.mkdir(parents=True, exist_ok=True)
    recordings = recording_plan(data / "analysis.json")
    root_height = float(recordings[0].session.source.frames[0, 2])
    local_frames = np.concatenate([r.frames for r in recordings]).copy()
    local_frames[:, :6] = 0
    local_frames[:, 2] = root_height
    points = body_positions(load_character(), local_frames).reshape(-1, 3)
    environment, engine = load_configs(str(ROOT / "data/envs/view_motion_humanoid_env.yaml"),
                                       str(ROOT / "data/engines/newton_engine.yaml"))
    session = recordings[0].session
    controls = PlaybackControls(session, (), "cumulative", view="local")
    _, environment_type = _environment_types()
    env = environment_type.__new__(environment_type)
    metadata = {"fps": 30, "root_height_m": root_height, "root_rotation": "identity",
                "ui": False, "source_loop_mode_preserved_in_data": True,
                "endpoint_sampling": "CLAMP during capture only, so the last stored frame is included",
                "timing": "frame timestamp span (T-1)/30; final frame held once, loop period T/30 rounded to 10 ms",
                "recordings": []}
    posters = []
    try:
        env.__init__(session, controls, environment, engine, device)
        env.reset()
        viewer = env._engine.get_viewer()
        viewer.show_ui = False
        viewer.renderer.draw_sky = False
        viewer.renderer.sky_upper = (.96, .96, .96)
        target = np.array([0., 0., root_height])
        layout = ViewLayout(np.zeros((4, 3), dtype=np.float32), (True, False, False, False),
                            target + [3.6, -4.8, 2.8], target)
        env._view_layout = layout
        env._camera = FixedCamera(env._engine, layout)
        env._display_renderer.configure(layout, "classroom")
        env._display_renderer.log_state(env._engine.get_sim_state().raw_state)
        env._engine.render()
        crop = fixed_crop(viewer, points)
        metadata.update({"crop_xyxy": list(crop), "camera_position": layout.camera_position.tolist(),
                         "camera_target": target.tolist(), "scale": "same projected crop and 520x520 output for every recording"})
        for recording in recordings:
            env.session = recording.session
            env._reconstructed = recording.frames
            env._classroom_frames = (recording.frames,) * 3
            env._load_motions(str(recording.session.source.path))
            env._motion_lib.local_root_height = root_height
            env._motion_lib._motion_loop_modes.zero_()
            env._display_motion_ids = torch.ones(4, dtype=torch.long, device=device)
            images = []
            for frame in range(len(recording.frames)):
                with torch.no_grad():
                    env._time_buf.fill_(frame / 30)
                    env._sync_motion()
                    env._camera.update()
                    env._display_renderer.log_state(env._engine.get_sim_state().raw_state)
                    env._engine.render()
                image = Image.fromarray(viewer.get_frame(render_ui=False).numpy())
                images.append(image.crop(crop).resize((520, 520), Image.Resampling.LANCZOS))
            path = output / f"{recording.name}.gif"
            entry = _encode(images, path, 30)
            entry.update({"file": path.name, "group": recording.group, "clip": recording.clip,
                          "mode": "pose", "basis": "independent" if recording.group in ("reference", "generated") else "shared",
                          "k": recording.k, "source_loop_mode": recording.session.source.loop_mode})
            metadata["recordings"].append(entry)
            posters.append((recording.name, images[len(images) // 2]))
            print(f"GIF: {path.name}, {len(images)} frames", flush=True)
    finally:
        env.close()
    sheet = Image.new("RGB", (4 * 260, int(np.ceil(len(posters) / 4)) * 290), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (name, poster) in enumerate(posters):
        x, y = index % 4 * 260, index // 4 * 290
        sheet.paste(poster.resize((260, 260)), (x, y))
        draw.text((x + 8, y + 264), name, fill="black")
    sheet.save(output / "contact_sheet.png")
    atomic_write(data / "gifs.json", (json.dumps(metadata, indent=2) + "\n").encode())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    record_case(args.case, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
