#version 120
uniform mat4 u_view_mat;
uniform mat4 u_perspective_mat;
attribute vec3 a_position;
attribute vec2 a_uv;
varying vec3 v_pos;
varying vec2 v_uv;
uniform float object_rotation;
uniform vec2 object_position;


void main() {
    vec4 pos4 = vec4(a_position, 1.0);

    float or = object_rotation;
    mat2 rotation_mat = mat2(cos(or), -sin(or), sin(or), cos(or));
    vec4 pos4_2 = pos4;
    pos4_2.xy = rotation_mat * pos4_2.xy;
    pos4_2.xy += object_position;

    gl_Position = u_perspective_mat * u_view_mat * pos4_2;

    v_pos = pos4.xyz;
    v_uv  = a_uv;
}
