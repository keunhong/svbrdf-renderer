#version 120
uniform sampler2D tex0;
uniform sampler2D tex1;
uniform sampler2D tex2;
uniform sampler2D tex3;
uniform sampler2D tex4;
uniform sampler2D tex5;
uniform sampler2D tex6;
uniform vec3 cameraPos;
uniform mat3 dirToEnv;
uniform vec3 normal;
varying vec3 v_pos;
varying vec2 v_uv;

uniform float alpha;

uniform float intensity;
uniform float light_azimuth;
uniform float light_elevation;
uniform float light_distance;
uniform vec3 light_color;

uniform float obj_rot;
uniform vec2 obj_pos;

void main() {
    vec3 alb_d = texture2D(tex1, v_uv).rgb;
    vec3 alb_s = texture2D(tex2, v_uv).rgb;
    vec3 specv = texture2D(tex3, v_uv).rgb;

    float or = -obj_rot;
    mat2 OR = mat2(cos(or), -sin(or), sin(or), cos(or));

    vec3 pos2 = v_pos;
    vec3 cameraPos2 = cameraPos;
    cameraPos2.xy = OR * (cameraPos2.xy-obj_pos);
    vec3 E = normalize(cameraPos2-pos2);
    vec3 L;

    L.x = sin(light_elevation)*cos(light_azimuth);
    L.y = sin(light_elevation)*sin(light_azimuth);
    L.z = cos(light_elevation);
    L = L * light_distance;
    L.xy = OR * (L.xy - obj_pos);
    L = L - pos2;

    float D2 = dot(L, L);
    L = L / sqrt(D2);

    float h = light_distance * cos(light_elevation);
    vec3 H = normalize(L + E);
    vec3 N = texture2D(tex4, v_uv).rgb;
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

    float spec = exp(-pow(dot(HnpW.xy,Hnp.xy), alpha*0.5));
    float cosine = max(.0, dot(N,L));
    float F0 = 0.04;

    float fres = F0 + (1-F0)*pow(1.0-max(0,(dot(H,E))), 5.0); // Schlick
    spec = spec * fres / F0;

    vec3 spec3 = spec / vec3(dot(H, L)); // From Brady et al. model A

    gl_FragColor.rgb = (vec3(spec3) * alb_s + alb_d) * vec3(cosine) / D2 * intensity * light_color;
    gl_FragColor.rgb = sqrt(gl_FragColor.rgb);    // rough gamma
    gl_FragColor.a = 1.0;
}
