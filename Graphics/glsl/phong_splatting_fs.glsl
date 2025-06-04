#version 330 core

in GS_OUT {
    vec3 fragNormal;    // interpolated from GS
    vec3 fragColor;     // interpolated from GS
    vec3 fragPosition;  // camera‐space position of this fragment
} fs_in;

out vec4 outColor;

uniform vec3 light_position;  // in camera space
uniform vec3 light_color;     // e.g. vec3(1.0)
uniform vec3 ambient_color;   // e.g. vec3(0.1)
uniform vec3 specular_color;  // e.g. vec3(1.0)
uniform float shininess;     // e.g. 32.0
uniform float min_color;     // e.g. 32.0
uniform float max_color;     // e.g. 32.0

void main() {
    vec3 N = normalize(fs_in.fragNormal);
    vec3 Ldir = light_position - fs_in.fragPosition;
    float dist2 = dot(Ldir, Ldir);
    vec3 L = normalize(Ldir);
    vec3 C = (fs_in.fragColor - min_color) / (max_color - min_color); // normalize color

    // Lambertian (diffuse) term
    float lambertian = max(dot(N, L), 0.0);
    lambertian = abs(dot(N, L));

    // Phong specular term
    float specular = 0.0;
    if (lambertian > 0.0) {
        vec3 V = normalize(-fs_in.fragPosition);   // view direction (camera at origin)
        vec3 R = reflect(-L, N);                   // perfect reflection
        float specAngle = max(dot(R, V), 0.0);
        specular = pow(specAngle, shininess);
    }

    vec3 ambient = ambient_color * C;
    vec3 diffuse = light_color * (lambertian * C) / dist2;
    vec3 specCol = light_color * (specular * specular_color) / dist2;

    vec3 finalColor = ambient + diffuse + specCol;
    outColor = vec4(finalColor, 1.0);
}
