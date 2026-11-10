"""Small NanoVDB file helpers; no simulation or geometry-pipeline dependencies."""

import os
from pathlib import Path
import struct
import tempfile

import numpy as np
import warp as wp


_FILE_MAGIC = (0x304244566F6E614E, 0x324244566F6E614E)
_GRID_MAGIC = (0x304244566F6E614E, 0x314244566F6E614E)


def grid_count(path):
    """Inspect one file segment, rejecting truncation and additional segments."""
    path = Path(path)
    with path.open("rb") as stream:
        header = stream.read(16)
        if len(header) != 16:
            raise ValueError("Truncated NanoVDB file header")
        magic, version, count, codec = struct.unpack("<QIHH", header)
        if magic not in _FILE_MAGIC or version >> 21 != 32:
            raise ValueError("Unsupported NanoVDB file signature or version")
        if count not in (1, 2) or codec not in (0, 1, 2):
            raise ValueError("Expected one segment with one or two NanoVDB grids")
        payload_size = 0
        for _ in range(count):
            metadata = stream.read(176)
            if len(metadata) != 176:
                raise ValueError("Truncated NanoVDB grid metadata")
            size = struct.unpack_from("<Q", metadata, 8)[0]
            name_size = struct.unpack_from("<I", metadata, 136)[0]
            if size < 1 or not 1 <= name_size <= 256:
                raise ValueError("Invalid NanoVDB grid size or name")
            name = stream.read(name_size)
            if len(name) != name_size or name[-1:] != b"\0":
                raise ValueError("Invalid NanoVDB grid name")
            payload_size += size
        remaining = path.stat().st_size - stream.tell()
        # Warp excludes compressed chunk prefixes from fileSize; native writers
        # may include them. Both conventions are accepted by Warp's reader.
        valid_sizes = (payload_size,) if codec == 0 else (payload_size, payload_size + 8 * count)
        if remaining not in valid_sizes:
            raise ValueError("Truncated NanoVDB payload or unsupported additional segment")
    return count


def save_volumes(volumes, path, *, grid_names=None):
    """Atomically save one or two independent grids as one NanoVDB segment.

    Only raw grid bytes are copied: values, topology, transforms and background
    remain unchanged. Inputs may alias a multi-grid buffer; only their own grid
    spans are copied. No source Volume is modified.
    """
    volumes = tuple(volumes)
    if len(volumes) not in (1, 2) or any(not isinstance(v, wp.Volume) for v in volumes):
        raise ValueError("Expected one or two wp.Volume objects")
    infos = [volume.get_grid_info() for volume in volumes]
    names = [info.name for info in infos] if grid_names is None else list(grid_names)
    if len(names) != len(volumes):
        raise ValueError("Each Volume must have one grid name")
    encoded_names = []
    for name in names:
        if not isinstance(name, str) or "\0" in name:
            raise ValueError("Grid names must be strings without null characters")
        encoded = name.encode("utf-8")
        if len(encoded) >= 256:
            raise ValueError("Grid names must fit in 255 UTF-8 bytes")
        encoded_names.append(encoded)

    packed = np.empty(sum(info.size_in_bytes for info in infos), dtype=np.uint8)
    offset = 0
    for index, (volume, info, name) in enumerate(zip(volumes, infos, encoded_names)):
        size = info.size_in_bytes
        if size < 672 or size % 32:
            raise ValueError("Unsupported NanoVDB grid layout")
        raw = wp.array(ptr=volume.array().ptr, dtype=wp.uint8, shape=size, device=volume.device)
        target = packed[offset:offset + size]
        target[:] = raw.numpy()
        if (struct.unpack_from("<Q", target)[0] not in _GRID_MAGIC
                or struct.unpack_from("<Q", target, 32)[0] != size):
            raise ValueError("Invalid NanoVDB grid header")
        struct.pack_into("<Q", target, 8, 0xFFFFFFFFFFFFFFFF)  # Undefined/disabled CRC.
        flags = struct.unpack_from("<I", target, 20)[0]
        struct.pack_into("<III", target, 20, flags & ~1, index, len(volumes))
        target[40:296] = 0
        target[40:40 + len(name)] = np.frombuffer(name, dtype=np.uint8)
        offset += size

    # CPU-only packing avoids allocating another full grid pair on the GPU.
    packed_volume = wp.Volume(wp.array(packed, dtype=wp.uint8, device="cpu"))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.stem + ".", suffix=".nvdb.tmp", dir=path.parent)
    os.close(descriptor)
    try:
        packed_volume.save_to_nvdb(temporary, codec="none")
        if grid_count(temporary) != len(volumes):
            raise RuntimeError("Saved NanoVDB grid count does not match inputs")
        with open(temporary, "rb") as stream:
            checked = wp.Volume.load_from_nvdb(stream, device="cpu")
        for index, info in enumerate(infos):
            if (checked is None or checked.get_grid_info().name != names[index]
                    or checked.get_grid_info().size_in_bytes != info.size_in_bytes):
                raise RuntimeError("Saved NanoVDB grid could not be read back")
            checked = checked.load_next_grid()
        if checked is not None:
            raise RuntimeError("Unexpected additional NanoVDB grid")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path
