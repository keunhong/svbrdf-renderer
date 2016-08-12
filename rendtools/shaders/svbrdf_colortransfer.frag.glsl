#version 120

uniform sampler2D diff_map;
uniform sampler2D spec_map;
uniform sampler2D spec_shape_map;
uniform sampler2D normal_map;
uniform vec3 cam_pos;
varying vec3 v_position;
varying vec3 v_normal;
varying vec3 v_tangent;
varying vec3 v_bitangent;
varying vec2 v_uv;

uniform float alpha;
uniform float light_intensity[$num_lights];
uniform vec3 light_position[$num_lights];
uniform vec3 light_color[$num_lights];

uniform vec3 source_mean;
uniform vec3 source_std;
uniform float spec_scale;
uniform float spec_shape_scale;

const float NUM_LIGHTS = $num_lights;
const float F0 = 0.04;

vec3 lab2xyz( vec3 c ) {
    float fy = ( c.x + 16.0 ) / 116.0;
    float fx = c.y / 500.0 + fy;
    float fz = fy - c.z / 200.0;
    return vec3(
         95.047 * (( fx > 0.206897 ) ? fx * fx * fx : ( fx - 16.0 / 116.0 ) / 7.787),
        100.000 * (( fy > 0.206897 ) ? fy * fy * fy : ( fy - 16.0 / 116.0 ) / 7.787),
        108.883 * (( fz > 0.206897 ) ? fz * fz * fz : ( fz - 16.0 / 116.0 ) / 7.787)
    );
}

vec3 xyz2rgb( vec3 c ) {
    vec3 v =  c / 100.0 * mat3(
        3.2406, -1.5372, -0.4986,
        -0.9689, 1.8758, 0.0415,
        0.0557, -0.2040, 1.0570
    );
    vec3 r;
    r.x = ( v.r > 0.0031308 ) ? (( 1.055 * pow( v.r, ( 1.0 / 2.4 ))) - 0.055 ) : 12.92 * v.r;
    r.y = ( v.g > 0.0031308 ) ? (( 1.055 * pow( v.g, ( 1.0 / 2.4 ))) - 0.055 ) : 12.92 * v.g;
    r.z = ( v.b > 0.0031308 ) ? (( 1.055 * pow( v.b, ( 1.0 / 2.4 ))) - 0.055 ) : 12.92 * v.b;
    return r;
}

vec3 lab2rgb(vec3 c) {
    return xyz2rgb( lab2xyz( vec3(c.x, c.y, c.z) ) );
}

void main() {
    vec3 alb_d = lab2rgb(texture2D(diff_map, v_uv).xyz * source_std + source_mean);
    vec3 alb_s = texture2D(spec_map, v_uv).rgb * spec_scale;
    vec3 specv = texture2D(spec_shape_map, v_uv).rgb * spec_shape_scale;

    vec3 E = normalize(cam_pos - v_position);

    mat3 TBN = mat3(v_tangent, v_bitangent, v_normal);
    vec3 N = normalize(TBN * texture2D(normal_map, v_uv).rgb);
    mat3 R = mat3(0, 0, N.x,
                  0, 0, N.y,
                  -N.x, -N.y, 0);

    mat2 M = mat2(specv.x, specv.z,
                  specv.z, specv.y);

	vec3 total_radiance = vec3(0.0);
	for (int i = 0; i < NUM_LIGHTS; i++) {
		vec3 L = light_position[i] - v_position;
		float D2 = dot(L, L);
		L = L / sqrt(D2);
		vec3 H = normalize(L+E);

		// Halfway vector in normal-oriented coordinates (so normal is [0,0,1])
		vec3 Hn = H + R * H + 1.0 / (N.z + 1.0) * (R * (R * H));
		vec3 Hnp = Hn / vec3(Hn.z);

		vec3 HnpW = vec3(M * Hnp.xy, 1.0);

		float spec = exp(-pow(dot(HnpW.xy, Hnp.xy), alpha * 0.5));
		float ndotl = max(.0, dot(N, L));

		float fres = F0 + (1 - F0) * pow(1.0 - max(0, dot(H,E)), 5.0); // Schlick
		spec = spec * fres / F0;
		spec = spec / dot(H, L); // From Brady et al. model A

		vec3 radiance = (vec3(spec) * alb_s + alb_d)
				* vec3(ndotl) / D2 * light_intensity[i] * light_color[i];
		total_radiance += radiance;
	}
    gl_FragColor = vec4(sqrt(total_radiance), 1.0);	// rough gamma

}
