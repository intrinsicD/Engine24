#version 330 core

// The output is a single integer, not a color.
layout (location = 0) out int out_EntityID;

// The entity ID is passed in as a uniform for each draw call.
uniform int u_EntityID;

void main()
{
    // Simply write the entity ID to the framebuffer.
    out_EntityID = u_EntityID;
}