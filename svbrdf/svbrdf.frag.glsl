#version 120
uniform sampler2D diff_map_tex;
uniform sampler2D spec_map_tex;
uniform sampler2D spec_shape_map_tex;
uniform sampler2D normal_map_tex;
uniform vec3 cam_pos;
varying vec3 v_pos;
varying vec2 v_uv;

uniform float alpha;

uniform float light_intensity;
uniform float light_azimuth;
uniform float light_elevation;
uniform float light_distance;
uniform vec3 light_color;

uniform float object_rotation;
uniform vec2 object_position;

void main() {
    vec3 alb_d = texture2D(diff_map_tex, v_uv).rgb;
    vec3 alb_s = texture2D(spec_map_tex, v_uv).rgb;
//    alb_d = vec3(0.0);
    vec3 specv = texture2D(spec_shape_map_tex, v_uv).rgb;

    float or = -object_rotation;
    mat2 OR = mat2(cos(or), -sin(or), sin(or), cos(or));

    vec3 pos2 = v_pos;
    vec3 cameraPos2 = cam_pos;
    cameraPos2.xy = OR * (cameraPos2.xy-object_position);
    vec3 E = normalize(cameraPos2-pos2);
    vec3 L;

    L.x = sin(light_elevation)*cos(light_azimuth);
    L.y = sin(light_elevation)*sin(light_azimuth);
    L.z = cos(light_elevation);
    L = L * light_distance;
    L.xy = OR * (L.xy - object_position);
    L = L - pos2;

    float D2 = dot(L, L);
    L = L / sqrt(D2);

    float h = light_distance * cos(light_elevation);
    vec3 H = normalize(L + E);
    vec3 N = texture2D(normal_map_tex, v_uv).rgb;
    N = N / length(N);

    mat3 R = mat3(
        0, 0, N.x,
        0, 0, N.y,
        -N.x, -N.y, 0);

    // Halfway vector in normal-oriented coordinates (so normal is [0,0,1])
    vec3 Hn = H + R*H + 1.0/(N.z+1.0) * (R*(R*H));
    vec3 Hnp = Hn / vec3(Hn.z);

    mat2 M = mat2(
        specv.x, specv.z,
        specv.z, specv.y);

    vec3 HnpW = vec3(M * Hnp.xy, 1.0);

    float spec = exp(-pow(dot(HnpW.xy,Hnp.xy), alpha * 0.5));
    float cosine = max(.0, dot(N,L));
    float F0 = 0.04;

    float fres = F0 + (1-F0)*pow(1.0-max(0,(dot(H,E))), 5.0); // Schlick
    spec = spec * fres / F0;

    vec3 spec3 = vec3(spec / dot(H, L)); // From Brady et al. model A

    gl_FragColor.rgb = (vec3(spec3) * alb_s + alb_d) * vec3(cosine) / D2
                       * light_intensity * light_color;
    gl_FragColor.rgb = sqrt(gl_FragColor.rgb);    // rough gamma
    gl_FragColor.a = 1.0;
}
