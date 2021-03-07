#include "frustrum.cuh"

#include <stdio.h>

__host__ __device__ Frustrum::Frustrum() 
{
    orig_a = Vec3();
    orig_b = Vec3();
    orig_c = Vec3();
    orig_d = Vec3();

    a = Vec3();
    b = Vec3();
    c = Vec3();
    d = Vec3();
}

__host__ __device__ Frustrum::Frustrum(Vec3 origin, Vec3 forward, Vec3 up, Vec3 side)
{
    this->origin = origin;

    this->orig_a = origin - side - up + forward;
    this->orig_b = origin - side + up + forward;
    this->orig_c = origin + side + up + forward;
    this->orig_d = origin + side - up + forward;

    this->up = up; // For viewport scaling purposes.
    this->side = side; // For viewport scaling purposes.
    this->forward = forward;

    this->a = this->orig_a - this->origin;
    this->b = this->orig_b - this->origin;
    this->c = this->orig_c - this->origin;
    this->d = this->orig_d - this->origin;
}

__host__ __device__ bool Frustrum::intersects(AABB3* aabb)
{
    Vec3 aabb_vertices[8];
    aabb->get_vertices(aabb_vertices);

    Plane3 planes[6];
    planes[0] = Plane3(orig_a, orig_b, orig_c); // near plane
    planes[1] = Plane3(orig_a, orig_b, orig_a + a);
    planes[2] = Plane3(orig_b, orig_c, orig_b + b);
    planes[3] = Plane3(orig_c, orig_d, orig_c + c);
    planes[4] = Plane3(orig_d, orig_a, orig_d + d);
    planes[5] = Plane3(orig_c + c, orig_b + b, orig_a + a);
    // ------IMPORTANT----------
    // p < 5 MEANS THE LAST PLANE (FAR PLANE) WONT BE TAKEN INTO ACCOUNT IN THIS TEST.

    // Test each vertice.
    for (int p = 0; p < 5; p++) {
        int result = 0;
        for(int i = 0; i < 8; i++) {
            result += planes[p].outside(aabb_vertices[i]) < 0.0 ? 1 : -1;
        }

        if (abs(result == 8))
            return false;
    }

    return true;
}

__host__ __device__ Vec3 Frustrum::recalculate_origin() {
    Vec3 g = orig_b - orig_a;
    float h = b.cross(g).length();
    float k = b.cross(a).length();

    Vec3 l = a * (h / k);
    return orig_a - l;
}

__host__ __device__ Vec3 Frustrum::normal() {
    return (orig_b - orig_a).cross(orig_d - orig_a);
}

__host__ __device__ void Frustrum::resize(AABB2 boundaries) {
    Vec3 screen_center = origin + forward;
    // 0.5 for average * 0.5 for requiring halve vectors = 0.25
    Vec3 s = (this->orig_d - this->orig_a);
    Vec3 u = (this->orig_b - this->orig_a);
    
    // This feels wrong but the boundaries itself will be negative if necessary thus everything should be addition.
    this->orig_a = screen_center + (s * boundaries.min.x) + (u * boundaries.min.y);
    this->orig_b = screen_center + (s * boundaries.min.x) + (u * boundaries.max.y);
    this->orig_c = screen_center + (s * boundaries.max.x) + (u * boundaries.max.y);
    this->orig_d = screen_center + (s * boundaries.max.x) + (u * boundaries.min.y);
}