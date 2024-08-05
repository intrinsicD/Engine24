#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in vec3 aNormal;
layout (location = 3) in float radius;

layout (std140) uniform Camera {
    mat4 view;
    mat4 projection;
};

uniform mat4 model;
uniform uint width;
uniform uint height;

uniform bool use_uniform_radius = true;
uniform float uniform_radius = 1.0;

uniform bool use_uniform_color = true;
uniform vec3 uniform_color = vec3(1.0, 1.0, 1.0);

out vec4 f_view;
out vec4 f_world;
out vec3 f_color;
out vec3 f_normal;
out float f_radius_view_space;

void main()
{
    if (use_uniform_color) {
        f_color = uniform_color;
    } else {
        f_color = aColor;
    }
    f_normal = mat3(transpose(inverse(model))) * aNormal;
    f_world = model * vec4(aPos, 1.0);
    f_view = view * f_world;
    vec4 clipSpacePos = projection * f_view;

    float distance = length(f_view.xyz);

    float adjustedPointSize = uniform_radius / distance;
    if(use_uniform_radius){
        adjustedPointSize = uniform_radius / distance;
    }else{
        adjustedPointSize = radius / distance;
    }

    float radius_ndc_x = adjustedPointSize / width * 2.0;
    float radius_ndc_y = adjustedPointSize / height * 2.0;
    // Use the larger of the two dimensions to ensure the point remains a square
    float radius_ndc_space = max(radius_ndc_x, radius_ndc_y);
    f_radius_view_space = (inverse(projection) * vec4(radius_ndc_space * clipSpacePos.w, 0, 0, 1.0)).x;
    //f_radius_view_space /= distance;
    gl_Position = clipSpacePos; //if the near and far plane are too great, there will be artifacts: TODO tight fit od near and fast plane!
    gl_PointSize = adjustedPointSize;
}