"""ssfr fluid pass, sharing pipeline-owned GPU resources."""
from pyglet import gl
from .resources import matrix, texture
import numpy as np


def render(context, scene):
    velocity = context.ssfr_preset == 'Velocity'
    thickness_name = 'thickness_velocity' if velocity else 'thickness'
    context.targets['depth'].bind([(1e6, 1e6, 0., 0.)])
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendEquation(gl.GL_MIN)
    context.particle_pass(scene, 'depth', context.config.surface_radius)
    gl.glBlendEquation(gl.GL_FUNC_ADD)
    gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE)
    thickness = context.targets[thickness_name]
    thickness.bind([(0.,)*4] * len(thickness.textures))
    context.particle_pass(scene, thickness_name, context.config.surface_radius)
    gl.glDisable(gl.GL_BLEND)
    p = np.asarray(scene.projection).reshape(4,4,order='F')
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
        for unit,(name,tex) in enumerate((('background',scene.color),('scene_depth',scene.depth),
                     ('surface',context.targets['smooth'].textures[0]),('thickness',thickness.textures[0]))):
            texture(shader,name,tex,unit)
        matrix(shader,'projection',scene.projection)
        matrix(shader,'inv_projection',np.linalg.inv(p).T)
        matrix(shader,'inv_view',np.linalg.inv(np.asarray(scene.view).reshape(4,4,order='F')).T)
        shader['size'] = context.size
        shader['ior'], shader['absorption'] = context.config.ior, context.config.absorption
        use_billboard_color = velocity or context.water_color == 'Billboard blue'
        shader['base_color'] = context.config.billboard_color if use_billboard_color else context.config.base_color
        if velocity:
            shader['white_opacity'] = context.config.alpha_high
            texture(shader, 'material_color', thickness.textures[1], 5)
            texture(shader, 'material_coverage', thickness.textures[2], 6)
        else:
            shader['billboard_blue'] = context.water_color == 'Billboard blue'
        shader['sky_up'], shader['sky_down'] = tuple(scene.sky_upper), tuple(scene.sky_lower)
        shader['sun'] = tuple(scene.sun)
        environment = scene.environment
        shader['has_environment'] = environment is not None
        texture(shader, 'environment', environment if environment is not None else scene.color, 4)
        context.full_screen()

