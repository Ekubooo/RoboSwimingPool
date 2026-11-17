"""CPU-owned demo settings shared by the panel and keyboard commands.

Only apply_pending() submits settings, on the playback stream at a frame
boundary. UI readout uses these values, never simulator array readbacks.
"""
import math


class FluidInteraction:
    def __init__(self, apply_settings):
        self._apply_settings = apply_settings
        self._dirty = False
        self.error = None
        self._defaults()

    def _defaults(self):
        self.gravity_strength = 9.81
        self.direction = (0., 0., -1.)
        self.feedback_scale = 1.0
        self.slow_motion = False
        self._horizontal = None

    def set_gravity_strength(self, value, *, commit=True):
        self.gravity_strength = self._bounded(value, 20., "gravity strength")
        if commit:
            self.commit_gravity()

    def commit_gravity(self):
        self._dirty = True

    def reverse_gravity(self):
        self.direction = tuple(-value for value in self.direction)
        self._horizontal = None
        self._dirty = True

    def select_horizontal(self, direction):
        if direction not in (-1, 1):
            raise ValueError("Horizontal direction must be -1 or +1")
        if self._horizontal == direction:
            self.direction, self._horizontal = (0., 0., -1.), None
        else:
            self.direction, self._horizontal = (float(direction), 0., 0.), direction
        self._dirty = True

    def set_feedback_scale(self, value):
        self.feedback_scale = self._bounded(value, 2., "feedback scale")
        self._dirty = True

    def set_slow_motion(self, enabled):
        self.slow_motion = bool(enabled)
        self._dirty = True

    def restore_defaults(self):
        self._defaults()
        self._dirty = True
        self.error = None

    def apply_pending(self):
        """Submit changed settings once, on the playback stream."""
        if self._dirty:
            self._apply_settings(
                gravity=tuple(value * self.gravity_strength for value in self.direction),
                feedback_scale=self.feedback_scale, slow_motion=self.slow_motion)
            self._dirty = False

    @staticmethod
    def _bounded(value, maximum, name):
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= maximum:
            raise ValueError(f"{name} must be between 0 and {maximum}")
        return value
