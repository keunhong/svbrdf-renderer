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
uniform float light_intensity;
uniform vec3 light_position;
uniform vec3 light_color;

void main() {
    vec3 alb_d = texture2D(diff_map, v_uv).rgb;
    vec3 alb_s = texture2D(spec_map, v_uv).rgb;
    vec3 specv = texture2D(spec_shape_map, v_uv).rgb;

    vec3 E = normalize(cam_pos - v_position);

    vec3 L = light_position - v_position;
    float D2 = dot(L, L);

    L = L / sqrt(D2);

    vec3 H = normalize(L+E);
    mat3 TBN = mat3(v_tangent, v_bitangent, v_normal);
    vec3 N = normalize(TBN * texture2D(normal_map, v_uv).rgb);

    mat3 R = mat3(0, 0, N.x,
                  0, 0, N.y,
                  -N.x, -N.y, 0);

    // Halfway vector in normal-oriented coordinates (so normal is [0,0,1])
    vec3 Hn = H + R * H + 1.0 / (N.z + 1.0) * (R * (R * H));
    vec3 Hnp = Hn / vec3(Hn.z);

    mat2 M = mat2(specv.x, specv.z,
                  specv.z, specv.y);

    vec3 HnpW = vec3(M * Hnp.xy, 1.0);

    float spec = exp(-pow(dot(HnpW.xy, Hnp.xy), alpha * 0.5));
    float cosine = max(.0, dot(N, L));
    float F0 = 0.04;

    float fres = F0 + (1 - F0) * pow(1.0 - max(0, dot(H,E)), 5.0); // Schlick
    spec = spec * fres / F0;
    spec = spec / dot(H, L); // From Brady et al. model A

    vec3 radiance = (vec3(spec) * alb_s + alb_d)
            * vec3(cosine) / D2 * light_intensity * light_color;
    gl_FragColor = vec4(sqrt(radiance), 1.0);	// rough gamma

}
