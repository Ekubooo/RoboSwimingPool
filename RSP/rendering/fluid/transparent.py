"""transparent fluid pass, sharing pipeline-owned GPU resources."""
from pyglet import gl
from .resources import texture


def render(context, scene):
    layered = context.config.transparent_layer_blending
    target = context.targets['oit' if layered else 'nearest']
    target.bind([(0.,)*4, (1.,)*4, (0.,)*4])
    if layered:
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendEquation(gl.GL_FUNC_ADD)
        gl.glBlendFunci(0, gl.GL_ONE, gl.GL_ONE)
        gl.glBlendFunci(1, gl.GL_ZERO, gl.GL_ONE_MINUS_SRC_COLOR)
        gl.glBlendFunci(2, gl.GL_ONE, gl.GL_ONE)
    else:
        # Private depth buffer: never write to the opaque scene's depth.
        gl.glDisable(gl.GL_BLEND)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LESS)
        gl.glDepthMask(True)
        gl.glClearBufferfv(gl.GL_DEPTH, 0, (gl.GLfloat*1)(1.))
    try:
        context.particle_pass(scene, 'transparent', 1.)
    finally:
        gl.glDisable(gl.GL_BLEND)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDepthMask(False)
    enabled = context.config.transparent_depth_enabled and context.config.transparent_depth_strength > 0
    depth = target.textures[2]
    range_texture = depth
    if enabled:
        shader = context.programs['depth_range']
        with shader:
            texture(shader, 'accumulation', target.textures[0], 1)
            first = True
            for name, width, height in context.depth_range_levels():
                range_target = context.targets[name]
                range_target.bind()
                gl.glViewport(0, 0, width, height)
                texture(shader, 'source', range_texture, 0)
                shader['first_level'] = first
                context.full_screen()
                range_texture = range_target.textures[0]
                first = False
        gl.glViewport(0, 0, *context.size)
    context.targets['output'].bind()
    shader = context.programs['resolve']
    with shader:
        texture(shader, 'background', scene.color, 0)
        texture(shader, 'accumulation', target.textures[0], 1)
        texture(shader, 'reveal', target.textures[1], 2)
        texture(shader, 'weighted_depth', depth, 3)
        texture(shader, 'depth_range', range_texture, 4)
        shader['depth_enabled'] = enabled
        shader['depth_strength'] = context.config.transparent_depth_strength
        context.full_screen()

