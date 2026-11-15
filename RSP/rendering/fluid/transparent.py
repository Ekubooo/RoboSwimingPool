"""transparent fluid pass, sharing pipeline-owned GPU resources."""
from pyglet import gl
from .resources import texture


def render(context):
    context.targets['oit'].bind([(0.,)*4, (1.,)*4])
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendEquation(gl.GL_FUNC_ADD)
    gl.glBlendFunci(0, gl.GL_ONE, gl.GL_ONE)
    gl.glBlendFunci(1, gl.GL_ZERO, gl.GL_ONE_MINUS_SRC_COLOR)
    context.particle_pass('transparent', 1.)
    gl.glDisable(gl.GL_BLEND)
    context.targets['output'].bind()
    shader = context.programs['resolve']
    with shader:
        texture(shader, 'background', context.viewer.renderer._frame_texture, 0)
        texture(shader, 'accumulation', context.targets['oit'].textures[0], 1)
        texture(shader, 'reveal', context.targets['oit'].textures[1], 2)
        context.full_screen()

