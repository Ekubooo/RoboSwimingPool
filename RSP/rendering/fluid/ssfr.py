"""ssfr fluid pass, sharing pipeline-owned GPU resources."""
from pyglet import gl
from .resources import matrix, texture
import numpy as np


def render(context):
    r = context.viewer.renderer
    velocity = context.ssfr_preset == 'Velocity'
    thickness_name = 'thickness_velocity' if velocity else 'thickness'
    context.targets['depth'].bind([(1e6, 1e6, 0., 0.)])
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendEquation(gl.GL_MIN)
    context.particle_pass('depth', context.config.surface_radius)
    gl.glBlendEquation(gl.GL_FUNC_ADD)
    gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE)
    thickness = context.targets[thickness_name]
    thickness.bind([(0.,)*4] * len(thickness.textures))
    context.particle_pass(thickness_name, context.config.surface_radius)
    gl.glDisable(gl.GL_BLEND)
    p = np.asarray(r._projection_matrix).reshape(4,4,order='F')
    shader = context.programs['blur']
    for source, target, axis in (('depth','horizontal',(1.,0.)),('horizontal','smooth',(0.,1.))):
        context.targets[target].bind()
        with shader:
            texture(shader,'source',context.targets[source].textures[0],0)
            shader['size'], shader['axis'] = context.size, axis
            shader['focal'] = float(p[1,1]*context.size[1]*.5)
            shader['max_radius'] = context.config.filter_radius
            context.full_screen()
    context.targets['output'].bind()
    shader = context.programs['surface_velocity' if velocity else 'surface']
    with shader:
        for unit,(name,tex) in enumerate((('background',r._frame_texture),('scene_depth',r._frame_depth_texture),
                     ('surface',context.targets['smooth'].textures[0]),('thickness',thickness.textures[0]))):
            texture(shader,name,tex,unit)
        matrix(shader,'projection',r._projection_matrix)
        matrix(shader,'inv_projection',np.linalg.inv(p).T)
        matrix(shader,'inv_view',np.linalg.inv(np.asarray(r._view_matrix).reshape(4,4,order='F')).T)
        shader['size'] = context.size
        shader['ior'], shader['absorption'] = context.config.ior, context.config.absorption
        shader['base_color'] = context.config.base_color
        if velocity:
            texture(shader, 'material_color', thickness.textures[1], 5)
            texture(shader, 'material_coverage', thickness.textures[2], 6)
        shader['sky_up'], shader['sky_down'] = tuple(r.sky_upper), tuple(r.sky_lower)
        shader['sun'] = tuple(r._sun_direction)
        environment = getattr(r, '_env_texture', None)
        shader['has_environment'] = environment is not None
        texture(shader, 'environment', environment if environment is not None else r._frame_texture, 4)
        context.full_screen()

