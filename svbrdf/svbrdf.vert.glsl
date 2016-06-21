#version 120
uniform mat4 u_view_mat;
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

    v_position = pos4.xyz;
//    v_normal = mat3(u_view_mat) * a_normal;
//    v_tangent = mat3(u_view_mat) * a_tangent;
//    v_bitangent = mat3(u_view_mat) * a_bitangent;
    v_normal = a_normal;
    v_tangent = a_tangent;
    v_bitangent = a_bitangent;
    v_uv  = a_uv;
}
