"""PBF configuration, initialization, and device mathematics.

Particle state is initialized by a Warp kernel on the active device.
Solver coefficients are precomputed here before the first solver compilation,
for one fixed configuration per process.
"""

from dataclasses import dataclass
from numbers import Real

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False})

# //////////////////////////////////////////////////////////////////////////////
# para initiative

@dataclass(frozen=True)
class PBFConfig:
    """Inputs in meters, kilograms and seconds; initialization derives state and constants."""

    particle_radius: float = 0.0125
    rest_density: float = 1000.0
    frame_dt: float = 1.0 / 150.0
    substeps: int = 3
    pressure_iterations: int = 3
    lambda_regularization: float = 2.0
    gravity: tuple = (0.0, -9.81, 0.0)
    viscosity: float = 0.5
    vorticity_confinement: float = 0.5
    max_speed: float = 6.5
    artificial_pressure_strength: float = 0.0005
    artificial_pressure_q: float = 0.1
    clamp_negative_pressure: bool = True

    # Reference container/block translated by (1.55, 0, 0.8), so the existing
    # zero-origin position constraint can remain unchanged.
    container_size: tuple = (3.0, 5.0, 2.0)
    block_start: tuple = (0.1, 0.5, 0.1)
    block_end: tuple = (1.5, 2.5, 1.25)
    initial_velocity: tuple = (0.0, 0.0, 0.0)

    def __post_init__(self):
        if not isinstance(self.clamp_negative_pressure, bool):
            raise ValueError("clamp_negative_pressure must be a bool")
        for name in (
            "particle_radius", "rest_density", "frame_dt", "lambda_regularization", "max_speed"
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("substeps", "pressure_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("viscosity", "vorticity_confinement", "artificial_pressure_strength"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        q = self.artificial_pressure_q
        if isinstance(q, bool) or not isinstance(q, Real) or not np.isfinite(q) or not 0 < q < 1:
            raise ValueError("artificial_pressure_q must be finite and in (0, 1)")
        for name in ("container_size", "block_start", "block_end", "initial_velocity", "gravity"):
            components = getattr(self, name)
            value = np.asarray(components)
            if (value.shape != (3,) or value.dtype.kind not in "iuf" or not np.isfinite(value).all()
                    or any(isinstance(item, (bool, np.bool_)) for item in components)):
                raise ValueError(f"{name} must contain three finite components")
        size = np.asarray(self.container_size)
        start = np.asarray(self.block_start)
        end = np.asarray(self.block_end)
        if np.any(size <= 2.0 * self.particle_radius):
            raise ValueError("Container dimensions must exceed the particle diameter")
        if np.any(start < 0.0) or np.any(end > size) or np.any(end <= start):
            raise ValueError("Fluid block must have positive size and lie inside the container")

@wp.struct
class NeighborCache:
    NeighborCount: wp.array[int]
    CacheBuffer: wp.array2d[int]

    def allocateNCache(self, pNum, nCap): 
        self.NeighborCount = wp.zeros(pNum, dtype=int)
        self.CacheBuffer = wp.zeros((nCap, pNum), dtype=int)
        # (nb_cache, p_num) shape is good for mem access.

        return self

@wp.kernel
def init_particles(
    block_start: wp.vec3,
    spacing: float,
    block_counts: wp.vec3i,
    positions: wp.array[wp.vec3]
):
    tid = wp.tid()
    # One thread per particle; Z varies fastest, then Y, then X.
    z = tid % block_counts[2]
    y = (tid // block_counts[2]) % block_counts[1]
    x = tid // (block_counts[1] * block_counts[2])
    offset = spacing * wp.vec3(float(x + 1), float(y + 1), float(z + 1))
    positions[tid] = block_start + offset

@wp.kernel
def by_cache_reordering(
    pos: wp.array(dtype=wp.vec3),
    pre_Pos: wp.array(dtype=wp.vec3),
    pos_temp: wp.array(dtype=wp.vec3),
    pPos_temp: wp.array(dtype=wp.vec3)
):
    id = wp.tid()
    ori_index = wp.hash_grid_point_id(GRID_id, id)
    pos_temp[id] = pos[ori_index]
    pPos_temp[id] = pre_Pos[ori_index]


def compute_hash_grid_dims(container_min, container_max, cell_width):
    """Use the reference's support padding and strictly larger powers of two."""
    # Float32 matches Warp's cell mapping at integer cell boundaries.
    lower = np.asarray(container_min, dtype=np.float32)
    upper = np.asarray(container_max, dtype=np.float32)
    width = np.float32(cell_width)
    # PBFConfig checks dimensions/ranges; float32 conversion can still overflow
    # or underflow, unlike the original Python values.
    if not np.isfinite(upper).all() or not np.isfinite(width) or width == 0:
        raise ValueError("Hash grid dimensions must be representable in float32")

    query_min = np.floor((lower - width) / width).astype(np.int64)
    query_max = np.floor((upper + width) / width).astype(np.int64)
    query_span = query_max - query_min + 1

    if np.any(query_span < 1):
        raise ValueError(f"Hash grid query span must be positive, got {tuple(query_span)}")

    dims = tuple(1 << int(span).bit_length() for span in query_span)
    cell_count = int(np.prod(dims, dtype=object))

    if cell_count > np.iinfo(np.int32).max:
        raise ValueError(
            "Automatic hash grid exceeds Warp's 32-bit cell index range: "
            f"span={tuple(query_span)}, dims={dims}, cells={cell_count}"
        )
    return dims


def init_parameters(sim):
    """Derive all particle scales and grid dimensions from the configuration."""
    global S_corr_K, S_corr_Q, Rho_0, Inv_Rho0, Lamb_Eps, visStrength, vorConfirm, MaxVel, Gravity
    global K_SPow3, K_DSPow3, K_SPoly6, Fluid_Mass, Smoothing_Radius, Particle_Radius
    global Boundary_Restitution, Boundary_Tangential_Retention, Boundary_Contact_Eps, Boundary_Disturbance
    global NP_clamping

    config = sim.config
    sim.particle_radius = config.particle_radius
    sim.particle_spacing = 2.0 * sim.particle_radius
    sim.support_radius = 4.0 * sim.particle_radius
    # Compatibility name used by the existing kernel launch arguments.
    sim.smoothing_length = sim.support_radius

    rest_density = config.rest_density
    fluid_volume = 0.8 * sim.particle_spacing**3
    fluid_mass = fluid_volume * rest_density
    lambda_epsilon = config.lambda_regularization / sim.support_radius**2

    # Keep the existing box collision settings independent of fluid parameters.
    boundaryRestitution = 0.98
    boundaryTangentialRetention = 1.0
    boundaryContactEpsScale = 1.0e-4
    boundaryDisturbance = 0.001

    # Preserve the original expressions, using the initialized support radius.
    paraPoly6 = 315.0 / (
        64.0 * wp.pi * wp.pow(wp.abs(sim.smoothing_length), 9.0)
    )
    paraPow3 = 15.0 / (wp.pi * wp.pow(sim.smoothing_length, 6.0))

    # Module globals are read directly by wp.func and by the solver via fn.
    Fluid_Mass = wp.constant(fluid_mass)
    Particle_Radius =  wp.constant(float(sim.particle_radius))
    Smoothing_Radius = wp.constant(float(sim.smoothing_length))

    # Lambda has units of length squared; scale tensile strength accordingly.
    S_corr_K = wp.constant(config.artificial_pressure_strength * sim.support_radius**2)
    S_corr_Q = wp.constant(config.artificial_pressure_q)
    Rho_0 = wp.constant(rest_density)
    Inv_Rho0 = wp.constant(1.0 / rest_density)
    Lamb_Eps = wp.constant(lambda_epsilon)
    visStrength = wp.constant(config.viscosity)
    vorConfirm = wp.constant(config.vorticity_confinement)
    MaxVel = wp.constant(config.max_speed)
    Gravity = wp.constant(wp.vec3(*config.gravity))
    K_SPow3 = wp.constant(paraPow3)
    K_DSPow3 = wp.constant(3.0 * paraPow3)
    K_SPoly6 = wp.constant(paraPoly6)
    Boundary_Restitution = wp.constant(boundaryRestitution)
    Boundary_Tangential_Retention = wp.constant(boundaryTangentialRetention)
    Boundary_Contact_Eps = wp.constant(
        boundaryContactEpsScale * sim.smoothing_length
    )
    Boundary_Disturbance = wp.constant(boundaryDisturbance)
    NP_clamping = wp.constant(int(config.clamp_negative_pressure))

    sim.iterations = config.pressure_iterations
    sim.neighborCapacity = int(192)
    sim.hash_grid_dims = compute_hash_grid_dims(
        (0.0, 0.0, 0.0), config.container_size, sim.support_radius
    )


def init_scene_parameters(sim):
    """Frame settings and the original unfiltered particle lattice."""
    config = sim.config
    sim.frame_dt = config.frame_dt
    sim.substep = config.substeps
    sim.particle_radius = config.particle_radius
    sim.particle_spacing = 2.0 * sim.particle_radius
    sim.smoothing_length = 4.0 * sim.particle_radius
    sim.width, sim.height, sim.length = config.container_size
    # Only three axis counts are computed on the host; no host particle arrays.
    start = np.asarray(config.block_start, dtype=float)
    end = np.asarray(config.block_end, dtype=float)
    counts = np.floor((end - start) / sim.particle_spacing + 0.5).astype(int) - 1
    if np.any(counts < 1):
        raise ValueError("Fluid block must fit at least one particle on every axis")
    sim.block_counts = wp.vec3i(*(int(count) for count in counts))
    sim.n = int(np.prod(counts))



def init_device_buffers(sim):
    """Allocate only solver work arrays; public particle storage belongs to State."""
    sim.pre_Pos     = wp.zeros(sim.n, dtype=wp.vec3)
    sim.delta_Pos   = wp.zeros(sim.n, dtype=wp.vec3)
    sim.lambda_Opt  = wp.zeros(sim.n, dtype=float)
    sim.volume_Rho  = wp.zeros(sim.n, dtype=float)
    sim.volume_Grad = wp.zeros(sim.n, dtype=wp.vec3)
    sim.delta_Vel   = wp.zeros(sim.n, dtype=wp.vec3)
    sim.curl        = wp.zeros(sim.n, dtype=wp.vec4)

    # reorder
    sim.temp_pPos   = wp.zeros(sim.n, dtype=wp.vec3)
    

    # Neighbor Cache
    sim.n_Cache = NeighborCache().allocateNCache(
        sim.n, 
        sim.neighborCapacity, 
    )

    # Special Hash Grid
    global GRID_id 
    sim.grid = wp.HashGrid(*sim.hash_grid_dims)
    GRID_id = wp.constant(wp.uint64(sim.grid.id))


def init_model(sim, builder=None):
    """Newton allocates particle storage; the original GPU kernel sets positions."""
    import newton
    if builder is None:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    builder.gravity = tuple(sim.config.gravity)
    mass = 0.8 * sim.particle_spacing**3 * sim.config.rest_density
    builder.add_particles(pos=[(0.0, 0.0, 0.0)] * sim.n,
                          vel=[(0.0, 0.0, 0.0)] * sim.n,
                          mass=[mass] * sim.n, radius=[sim.particle_radius] * sim.n)
    return builder.finalize(device=wp.get_device())


def init_particle_state(sim):
    # The helper module also contains reorder, so initialize GRID_id first.
    wp.launch(init_particles, sim.n,
              inputs=[wp.vec3(*sim.config.block_start), sim.particle_spacing, sim.block_counts],
              outputs=[sim.model.particle_q], device=sim.model.device)


def clear_work_buffers(sim):
    for buffer in (sim.pre_Pos, sim.temp_pPos, sim.delta_Pos, sim.lambda_Opt,
                   sim.volume_Rho, sim.volume_Grad, sim.delta_Vel, sim.curl,
                   sim.n_Cache.NeighborCount):
        buffer.zero_()


def buffer_reorder(sim, pos_in, pos_out):
    wp.launch(
        kernel=by_cache_reordering,
        dim=sim.n,
        inputs=[
            pos_in,
            sim.pre_Pos,
        ],
        outputs=[            
            pos_out,
            sim.temp_pPos,
        ]
    )

    sim.pre_Pos, sim.temp_pPos = sim.temp_pPos, sim.pre_Pos


# //////////////////////////////////////////////////////////////////////////////
# calculate function

@wp.func_native("return query.cell_index - 1;")
def sorted_index(query: wp.HashGridQuery) -> int:
    """Read the sorted slot immediately after a successful hash_grid_query_next.

    Warp 1.15 increments its internal cell_index when yielding the input point
    index. Use the slot for particle state gathered in the same build's hash
    order. Recheck this internal field when upgrading Warp.
    """
    ...


@wp.func
def square(x: float):
    return x * x


@wp.func
def cube(x: float):
    return x * x * x


@wp.func
def fifth(x: float):
    return x * x * x * x * x


@wp.func
def density_kernel(xyz: wp.vec3, smoothing_length: float):
    # calculate distance
    distance = wp.dot(xyz, xyz)

    return wp.max(cube(square(smoothing_length) - distance), 0.0)


@wp.func
def diff_pressure_kernel(
    xyz: wp.vec3, pressure: float, neighbor_pressure: float, neighbor_rho: float, smoothing_length: float
):
    # calculate distance
    distance = wp.sqrt(wp.dot(xyz, xyz))

    if distance < smoothing_length:
        # calculate terms of kernel
        term_1 = -xyz / distance
        term_2 = (neighbor_pressure + pressure) / (2.0 * neighbor_rho)
        term_3 = square(smoothing_length - distance)
        return term_1 * term_2 * term_3
    else:
        return wp.vec3()


@wp.func
def diff_viscous_kernel(
    xyz: wp.vec3, v: wp.vec3, neighbor_v: wp.vec3, neighbor_rho: float, smoothing_length: float
):
    # calculate distance
    distance = wp.sqrt(wp.dot(xyz, xyz))

    # calculate terms of kernel
    if distance < smoothing_length:
        term_1 = (neighbor_v - v) / neighbor_rho
        term_2 = smoothing_length - distance
        return term_1 * term_2
    else:
        return wp.vec3()

@wp.func
def Poly6(pos: wp.vec3, neighbor_pos: wp.vec3):
    O2N = neighbor_pos - pos
    sqrD2N = wp.dot(O2N, O2N)
    if sqrD2N < square(Smoothing_Radius):
        return cube(square(Smoothing_Radius) - sqrD2N) * K_SPoly6
    return 0.0

@wp.func
def GradPow3(pos: wp.vec3, neighbor_pos: wp.vec3):
    O2N = neighbor_pos - pos
    sqrD2N = wp.dot(O2N, O2N)
    if sqrD2N > 0.0 and sqrD2N < square(Smoothing_Radius):
        dst2N = wp.sqrt(sqrD2N)
        grad = -1.0 * square(Smoothing_Radius - dst2N) * K_DSPow3
        return (O2N/dst2N) * grad
    return wp.vec3(0,0,0)

@wp.func
def Pow3(dst: float):
    if dst < Smoothing_Radius:
        # SPow3 = 15.0 / (wp.pi * wp.pow(Smoothing_Radius, 6.0))
        return cube(Smoothing_Radius - dst) * K_SPow3
    return 0.0


@wp.func
def apply_boundary(pos: wp.vec3, boundary: wp.vec3, tid: int):

    width = boundary[0]
    high = boundary[1]
    length = boundary[2]

    state = wp.rand_init(123, tid)
    disturbance = Boundary_Disturbance * wp.vec3(
        wp.abs(wp.randf(state)),
        wp.abs(wp.randf(state)),
        wp.abs(wp.randf(state)),
    )

    # clamping
    # clamp pos left
    if pos[0] < 0.0:
        pos = wp.vec3(disturbance[0], pos[1], pos[2])

    # clamp x right
    if pos[0] > width:
        pos = wp.vec3(width - disturbance[0], pos[1], pos[2])

    # clamp y bot
    if pos[1] < 0.0:
        pos = wp.vec3(pos[0], disturbance[1], pos[2])

    # clamp y up
    if pos[1] > high:
        pos = wp.vec3(pos[0], high - disturbance[1], pos[2])

    # clamp z left
    if pos[2] < 0.0:
        pos = wp.vec3(pos[0], pos[1], disturbance[2])

    # clamp z right
    if pos[2] > length:
        pos = wp.vec3(pos[0], pos[1], length - disturbance[2])

    return pos

 
@wp.func
def query_sdf_normal(sdf_id: wp.uint64, position: wp.vec3):
    """Return validity, signed distance and a safe outward world-space normal.

    Validate before normalizing; accept negative distances for collision
    projection. Assumes axis-aligned, uniformly spaced voxels, as in the
    baked SDF. Invalid samples return zero distance and normal.
    """
    index = wp.volume_world_to_index(sdf_id, position)
    gradient = wp.vec3(0.0)
    distance = wp.volume_sample_grad_f(sdf_id, index, wp.Volume.LINEAR, gradient)
    if not wp.isfinite(distance):
        return False, 0.0, wp.vec3(0.0)

    # For this grid, the direction transform is simply gradient / voxel_size.
    world_gradient = wp.volume_world_to_index_dir(sdf_id, gradient)
    gradient_squared = wp.dot(world_gradient, world_gradient)
    if not wp.isfinite(gradient_squared) or gradient_squared <= 1.0e-12:
        return False, 0.0, wp.vec3(0.0)

    return True, distance, world_gradient / wp.sqrt(gradient_squared)


@wp.func
def apply_SDF(volume_id: wp.uint64, pos: wp.vec3, radius: wp.float32):
    valid, distance, normal = query_sdf_normal(volume_id, pos)
    if valid and distance < radius:
        pos += normal * (radius - distance)
    return pos


@wp.func
def query_volume_boundary(sdf_id: wp.uint64, position: wp.vec3, support_radius: float):
    """Return validity, protected pressure distance and direction toward the boundary."""
    valid, distance, normal = query_sdf_normal(sdf_id, position)
    if not valid or distance <= 0.0 or distance >= support_radius:
        return False, 0.0, wp.vec3(0.0)
    
    # PBF uses support_radius = 4 * particle_radius. Keep the equivalent
    # boundary point at least one particle diameter away to limit pressure.
    pressure_distance = wp.max(distance + 0.125 * support_radius, 0.5 * support_radius)
    return True, pressure_distance, -normal

