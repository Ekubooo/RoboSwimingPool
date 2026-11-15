"""transparent fluid pass, sharing pipeline-owned GPU resources."""
from pyglet import gl
from .resources import texture


def render(context):
    context.targets['oit'].bind([(0.,)*4, (1.,)*4, (0.,)*4])
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendEquation(gl.GL_FUNC_ADD)
    gl.glBlendFunci(0, gl.GL_ONE, gl.GL_ONE)
    gl.glBlendFunci(1, gl.GL_ZERO, gl.GL_ONE_MINUS_SRC_COLOR)
    gl.glBlendFunci(2, gl.GL_ONE, gl.GL_ONE)
    context.particle_pass('transparent', 1.)
    gl.glDisable(gl.GL_BLEND)
    enabled = context.config.transparent_depth_enabled and context.config.transparent_depth_strength > 0
    depth = context.targets['oit'].textures[2]
    range_texture = depth
    if enabled:
        shader = context.programs['depth_range']
        with shader:
            texture(shader, 'accumulation', context.targets['oit'].textures[0], 1)
            first = True
            for name, width, height in context.depth_range_levels():
                target = context.targets[name]
                target.bind()
                gl.glViewport(0, 0, width, height)
                texture(shader, 'source', range_texture, 0)
                shader['first_level'] = first
                context.full_screen()
                range_texture = target.textures[0]
                first = False
        gl.glViewport(0, 0, *context.size)
    context.targets['output'].bind()
    shader = context.programs['resolve']
    with shader:
        texture(shader, 'background', context.viewer.renderer._frame_texture, 0)
        texture(shader, 'accumulation', context.targets['oit'].textures[0], 1)
        texture(shader, 'reveal', context.targets['oit'].textures[1], 2)
        texture(shader, 'weighted_depth', depth, 3)
        texture(shader, 'depth_range', range_texture, 4)
        shader['depth_enabled'] = enabled
        shader['depth_strength'] = context.config.transparent_depth_strength
        context.full_screen()

