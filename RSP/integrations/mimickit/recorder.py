"""Keep MimicKit's recorder and presentation interfaces on the Engine adapter."""

from engines.newton_recorder import NewtonVideoRecorder
from util import resource_util
from util.logger import Logger

from RSP.rendering.newton_gl import NewtonBillboardViewer as PBFViewer
from RSP.rendering.presentation import Presentation


class PBFVideoRecorder(NewtonVideoRecorder):
    def _build_viewer(self):
        self._viewer = resource_util.construct(
            PBFViewer, width=self._resolution[0], height=self._resolution[1], headless=True)
        self._engine.init_viewer(self._viewer)


class EnginePresentation(Presentation):
    def __init__(self, num_envs, env_spacing, visualize, record_video, *, viewer=None):
        self._video_recorder = None
        self._record_video = record_video and not visualize
        self._recording = False
        super().__init__(
            num_envs, env_spacing, visualize, viewer=viewer,
            create_viewer=lambda: resource_util.construct(PBFViewer, headless=False),
        )

    def initialize_recording(self, engine_api):
        if self._record_video:
            Logger.print("Video recording enabled")
            self._video_recorder = resource_util.construct(PBFVideoRecorder, engine_api)

    def capture_before_step(self):
        if self._record_video and self._recording:
            self._video_recorder.capture_frame()

    def enabled_record_video(self):
        return self._record_video

    def get_video_recording(self):
        return self._video_recorder.get_video()

    def start_video_recording(self):
        self._video_recorder.clear()
        self._recording = True

    def stop_video_recording(self):
        self._recording = False

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._recording = False
        recorder = getattr(self, "_video_recorder", None)
        try:
            resource_util.cleanup(recorder.close if recorder is not None else None, super().close)
        finally:
            self._video_recorder = None
