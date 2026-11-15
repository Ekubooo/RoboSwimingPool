"""A selection-only view of Newton geometry; physics keeps the full model."""
import numpy as np
import warp as wp
import newton


class VisiblePickModel:
    """Hide collision-only pool walls from rays, retaining visible occluders.

    Newton 1.7's picker raycasts every shape without consulting shape_flags.
    A closed, invisible collision box therefore intercepts rays from outside.
    Cache only the picker's six shape arrays; body IDs and live state still
    refer to the original model. Rebuild on model/topology changes, not frames.
    """

    def __init__(self, model):
        self._source = model
        flags = model.shape_flags.numpy()
        self.shape_ids = np.flatnonzero(flags & int(newton.ShapeFlags.VISIBLE))
        self.shape_count = len(self.shape_ids)
        for name in ("shape_body", "shape_transform", "shape_type", "shape_scale",
                     "shape_source_ptr", "shape_world"):
            source = getattr(model, name)
            setattr(self, name, wp.array(source.numpy()[self.shape_ids],
                                         dtype=source.dtype, device=model.device))

    def __getattr__(self, name):
        return getattr(self._source, name)
