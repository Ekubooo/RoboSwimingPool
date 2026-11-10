"""Resource cleanup at application boundaries, without altering computation."""
import sys
import warnings


def cleanup(*callbacks):
    """Try every callback; retain an existing exception instead of replacing it."""
    original_error = sys.exc_info()[1]
    errors = []
    for callback in callbacks:
        if callback is None:
            continue
        try:
            callback()
        except Exception as error:
            errors.append(error)
    if errors:
        if original_error is None:
            raise errors[0]
        for error in errors:
            warnings.warn("Resource cleanup failed: {}".format(error), RuntimeWarning)


def close_resources(*resources):
    cleanup(*(resource.close for resource in resources if resource is not None))


def construct(resource_class, *args, **kwargs):
    """Close partially constructed resources if their initializer fails."""
    resource = resource_class.__new__(resource_class)
    try:
        resource_class.__init__(resource, *args, **kwargs)
    except BaseException:
        close_resources(resource)
        raise
    return resource


def close_viewer(viewer):
    """Close a Newton viewer while callbacks live; skip an already closed context."""
    if viewer is None:
        return
    window = getattr(getattr(viewer, "renderer", None), "window", None)
    if window is not None and window.context is None:
        return
    viewer.close()
