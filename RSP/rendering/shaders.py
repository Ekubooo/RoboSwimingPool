"""Opaque square billboards, matching WarpLab's view-space expansion."""

BILLBOARD_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec2 corner;
layout (location = 1) in vec4 center_radius;
layout (location = 2) in vec4 color;

uniform mat4 view;
uniform mat4 projection;
out vec4 particle_color;

void main() {
    // Orientation comes from the current camera, even if simulation is paused.
    vec4 center = view * vec4(center_radius.xyz, 1.0);
    gl_Position = projection * (center + vec4(corner * center_radius.w, 0.0, 0.0));
    particle_color = color;
}
"""

BILLBOARD_FRAGMENT_SHADER = """
#version 330 core
in vec4 particle_color;
out vec4 fragment_color;

void main() {
    fragment_color = particle_color;
}
"""
