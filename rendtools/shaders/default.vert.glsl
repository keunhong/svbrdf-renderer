#version 120
uniform mat4 u_view_mat;
uniform mat4 u_model_mat;
uniform mat4 u_perspective_mat;
attribute vec3 a_position;
attribute vec2 a_uv;
attribute vec3 a_normal;
attribute vec3 a_tangent;
attribute vec3 a_bitangent;
varying vec3 v_position;
varying vec3 v_normal;
varying vec3 v_tangent;
varying vec3 v_bitangent;
varying vec2 v_uv;

void main() {
    gl_Position = u_perspective_mat * u_view_mat * u_model_mat * vec4(a_position, 1.0);

    v_position = a_position.xyz;
    v_normal = a_normal;
    v_tangent = a_tangent;
    v_bitangent = a_bitangent;
    v_uv  = a_uv;
}
