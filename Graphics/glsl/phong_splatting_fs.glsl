#version 330 core

in GS_OUT {
    vec3 fragNormal;// interpolated from GS
    vec3 fragColor;// interpolated from GS
    vec3 fragPosition;// camera‐space position of this fragment
    vec2 gsUV;
} fs_in;

out vec4 outColor;

uniform vec3 light_position;// in camera space
uniform vec3 light_color;// e.g. vec3(1.0)
uniform vec3 ambient_color;// e.g. vec3(0.1)
uniform vec3 specular_color;// e.g. vec3(1.0)
uniform float shininess;// e.g. 32.0
uniform float min_color;// e.g. 32.0
uniform float max_color;// e.g. 32.0

float lambertianTerm(vec3 N, vec3 L) {
    //return max(dot(N, L), 0.0);
    return abs(dot(N, L));//somehow my normals are inverted, so I use abs...
}

float specularTerm(vec3 R, vec3 V, float shininess) {
    float specAngle = max(dot(R, V), 0.0);
    return pow(specAngle, shininess);
}

vec3 PhongShadingFinalColor(vec3 base_color, vec3 N, vec3 L, vec3 view_pos, float shininess, float dist2){
    // Lambertian (diffuse) term
    float lambertian = lambertianTerm(N, L);

    // Phong specular term
    float specular = 0.0;
    if (lambertian > 0.0) {
        vec3 V = normalize(-view_pos);// view direction (camera at origin)
        vec3 R = reflect(-L, N);// perfect reflection
        specular = specularTerm(R, V, shininess);
    }

    vec3 ambient = ambient_color * base_color;
    vec3 diffuse = light_color * (lambertian * base_color) / dist2;
    vec3 specCol = light_color * (specular * specular_color) / dist2;

    return ambient + diffuse + specCol;
}

void main() {
    vec2 gsUV = fs_in.gsUV; // UV coordinates from geometry shader, in [-1, 1] range discarding to get spherical splats
    if (gsUV.x * gsUV.x + gsUV.y * gsUV.y > 1.0) {
        discard;
    }

    vec3 N = normalize(fs_in.fragNormal);
    vec3 Ldir = light_position - fs_in.fragPosition;
    float dist2 = dot(Ldir, Ldir);
    vec3 L = normalize(Ldir);
    vec3 C = (fs_in.fragColor - min_color) / (max_color - min_color);// normalize color

    vec3 finalColor = PhongShadingFinalColor(C, N, L, fs_in.fragPosition, shininess, dist2);
    outColor = vec4(finalColor, 1.0);
}
