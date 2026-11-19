"""Apply MimicKit's articulation and skeleton requirements to a Newton scene."""

from RSP.sim_ready.lib.asset_validation import validate_skeleton
from RSP.simulation.scenes.build import PBFScene

from .validation import validate_model_for_mimickit


class MimicKitScene(PBFScene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._skeletons = {}

    def register_character(self, obj_id, skeleton, excluded_links=()):
        super().register_character(obj_id, excluded_links)
        self._skeletons[obj_id] = skeleton

    def validate_consumer_layout(self, model, bindings):
        self.validation.extend(validate_model_for_mimickit(model, bindings))

    def validate_consumer_object(self, obj_id, binding):
        if obj_id in self._skeletons:
            self.validation.extend(validate_skeleton(binding.asset, self._skeletons[obj_id]))

    def validate_consumer_objects(self, bindings):
        if set(self._skeletons) - set(range(len(bindings))):
            self.validation.add("error", "skeleton.object", "scene", "Registered character has no physical articulation")
