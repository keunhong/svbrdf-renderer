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

uniform float object_rotation;
uniform vec2 object_position;

const float PI = 3.141592;
const float F0 = 0.04;


float fresnelSchlick(float F0, vec3 V, vec3 H) {
    float VdotH = max(0, dot(V, H));
    float F = F0 + (1 - F0) * pow(1 - VdotH, 5);
    return F;
}

// Compute the shortest rotation from one unit vector another.
struct RotationBetweenVectors {
    vec3 axis;
    float  cosTheta;
    float  sinTheta;
};

RotationBetweenVectors computeRotationFromTo(
        vec3 fromNormalized, vec3 toNormalized) {
    const float Epsilon = 1e-2;

    RotationBetweenVectors r;
    r.axis       = cross(fromNormalized, toNormalized);

    float len    = length(r.axis);
    float rcpLen = (len < Epsilon) ? 0 : (1 / len);

    r.axis    *= rcpLen;
    r.sinTheta = len;
    r.cosTheta = dot(fromNormalized, toNormalized);
    return r;
}

// https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
vec3 rotateWith(RotationBetweenVectors rotation, vec3 v) {
    vec3 k = rotation.axis;
    float c = rotation.cosTheta;
    float s = rotation.sinTheta;

    vec3 vRot = v * c + cross(k, v) * s + k * (dot(k, v) * (1 - c));
    return vRot;
}


void main() {
//    gl_FragColor = vec4((v_normal + 1.0) / 2.0, 1.0);

    mat3 TBN = mat3(v_tangent, v_bitangent, v_normal);
    vec3 alb_d = texture2D(diff_map, v_uv).rgb * PI;
    vec3 alb_s = texture2D(spec_map, v_uv).rgb * (4 / F0);
//    alb_d = vec3(0);
    vec3 specv = texture2D(spec_shape_map, v_uv).rgb;

    vec3 V = normalize(cam_pos-v_position);
    vec3 L;

    vec3 N = texture2D(normal_map, v_uv).rgb;
    N = normalize(TBN * N);

    RotationBetweenVectors toNormalOriented = computeRotationFromTo(N, vec3(0, 0, 1));

    L = light_position - v_position;

    float D2 = dot(L, L);
    L = L / sqrt(D2);

    vec3 H = normalize(L + V);
    vec3 H_ = rotateWith(toNormalOriented, H);
    vec2 h = H_.xy / H_.z;
    mat2 S = mat2(specv.x, specv.z,
                  specv.z, specv.y);

    vec2 hT_S = S * h;
    float hT_S_h = dot(hT_S, h);

    float D = exp(-pow(abs(hT_S_h), alpha / 2));
    float F = fresnelSchlick(F0, V, H);

    vec3 diffuse = alb_d;
    vec3 specular = alb_s * D * F / (4 * dot(L, H));

    vec3 reflectance = (diffuse + specular);

    float distance    = length(L);
    float attenuation = 1 / (distance * distance) * 1.0f;
    float cosineTerm  = max(0, dot(N, L));

    // Incident radiance of the light
    vec3 L_i = light_color;

    vec3 radiance = L_i * attenuation * cosineTerm * reflectance;

    gl_FragColor = vec4(radiance, 1.0);
}
