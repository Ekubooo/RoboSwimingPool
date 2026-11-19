"""CPU wall-time estimates of continuous playback, without GPU synchronization."""


class PlaybackStats:
    """Publish count / total duration once per non-overlapping half-second window."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.fps = None
        self._frames = 0
        self._elapsed = 0.0

    def record(self, elapsed, *, continuous):
        if not continuous:
            self.clear()
        elif elapsed > 0.0:
            self._frames += 1
            self._elapsed += elapsed
            if self._elapsed >= 0.5:
                self.fps = self._frames / self._elapsed
                self._frames = 0
                self._elapsed = 0.0
        return self.fps
