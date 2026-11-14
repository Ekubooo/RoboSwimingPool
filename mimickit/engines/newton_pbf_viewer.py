"""MimicKit entry points for RSP's Newton billboard viewer and recorder."""

from RSP.rendering.newton_gl import NewtonBillboardViewer as PBFViewer

from engines.newton_recorder import NewtonVideoRecorder
from util import resource_util


class PBFVideoRecorder(NewtonVideoRecorder):
    def _build_viewer(self):
        self._viewer = resource_util.construct(
            PBFViewer, width=self._resolution[0], height=self._resolution[1], headless=True)
        self._engine.init_viewer(self._viewer)
