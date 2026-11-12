"""A small output store: paths, recipe names, staging and group publication.

Asset-specific validity, support selection and cache-hit decisions belong to
pipelines. This class does not import geometry, baking or runtime modules.
"""

from contextlib import contextmanager
from pathlib import Path
import tempfile

from ...paths import output_root as resolve_output_root
from .fingerprints import recipe_digest
from .publish import publish_files


class AssetStore:
    def __init__(self, output_root):
        self._root = resolve_output_root(output_root)

    @property
    def root(self):
        return self._root

    def _checked(self, path):
        path = resolve_output_root(path)
        if not path.is_relative_to(self.root):
            raise ValueError(f"Asset output escapes its store: {path}")
        return path

    def path(self, *parts):
        """Resolve a relative output path without creating it."""
        if any(Path(part).is_absolute() or Path(part).drive or Path(part).root for part in parts):
            raise ValueError("Asset paths must be relative to their store")
        return self._checked(self.root.joinpath(*parts))

    def directory(self, *parts):
        path = self.path(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def recipe_key(description, *, allow_nan=False):
        # Preserve the previous 20-character key and JSON encoding.
        return recipe_digest(description, allow_nan=allow_nan)[:20]

    def recipe_directory(self, category, description, *, name=None, allow_nan=False):
        key = self.recipe_key(description, allow_nan=allow_nan)
        if name is not None:
            if not name or any(c in name for c in '/\\:') or name in ('.', '..'):
                raise ValueError("An asset recipe name must be one filename component")
            key = f"{name}-{key}"
        return self.directory(category, key)

    @contextmanager
    def staging(self, destination, *, prefix=".asset-"):
        """Stage beside a destination so relative artifact references stay valid."""
        if any(c in prefix for c in '/\\:'):
            raise ValueError("A staging prefix must be one filename component")
        destination = self._checked(destination)
        parent = self._checked(destination.parent)
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=prefix, dir=parent) as temporary:
            yield Path(temporary)

    def publish(self, destinations):
        """Validate every target before publishing a rollback-protected group."""
        checked = [(Path(source), self._checked(destination)) for source, destination in destinations]
        publish_files(checked)
