#version 120

varying vec3 v_position;
varying vec3 v_normal;

uniform vec3 cam_pos;
uniform vec3 u_diff;
uniform vec3 u_spec;
uniform float u_shininess;
uniform float light_intensity[$num_lights];
uniform vec3 light_position[$num_lights];
uniform vec3 light_color[$num_lights];

const float NUM_LIGHTS = $num_lights;

void main() {
	vec3 color = vec3(0.0);
	vec3 view_dir = normalize(cam_pos - v_position);
	for (int i = 0; i < $num_lights; i++) {
		vec3 light_dir = normalize(light_position[i] - v_position);
		float ndotl = dot(v_normal, light_dir);
		vec3 refl_dir = normalize(2.0 * ndotl * v_normal - light_dir);
		float rdotv = dot(refl_dir, view_dir);

		vec3 Id = u_diff * ndotl;
		vec3 Is = u_spec * pow(rdotv, u_shininess);
		color += light_color[i] * light_intensity[i]/2000 * (Id + Is);
	}
	gl_FragColor = vec4(color, 1.0);
}
