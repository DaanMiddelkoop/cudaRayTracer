#include "aabb.cuh"

#include <math.h>

__host__ __device__ AABB3::AABB3() {
    min = Vec3();
    max = Vec3();
}

__host__ __device__ AABB3::AABB3(Vec3 min, Vec3 max) {
    this->min = min;
    this->max = max;
}

__host__ __device__ void AABB3::get_vertices(Vec3* vertices)
{
    vertices[0] = this->min;
    vertices[1] = Vec3(max.x, min.y, min.z);
    vertices[2] = Vec3(min.x, max.y, min.z);
    vertices[3] = Vec3(min.x, min.y, max.z);
    vertices[4] = Vec3(max.x, max.y, min.z);
    vertices[5] = Vec3(min.x, max.y, max.z);
    vertices[6] = Vec3(max.x, min.y, max.z);
    vertices[7] = this->max;
}

__host__ __device__ AABB3 AABB3::get_union(AABB3* other) {
    return AABB3(min.minimum(other->min), max.maximum(other->max));
}

__host__ __device__ bool AABB3::intersects(Vec3 ray_origin, Vec3 ray_direction, float* distance) {
    #ifndef max
    #define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
    #endif

    #ifndef min
    #define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
    #endif

    ray_direction = ray_direction.normalize();
    Vec3 dirfrac = Vec3(1.0 / ray_direction.x, 1.0 / ray_direction.y, 1.0 / ray_direction.z);
    float t1 = (min.x - ray_origin.x) * dirfrac.x;
    float t2 = (max.x - ray_origin.x) * dirfrac.x;
    float t3 = (min.y - ray_origin.y) * dirfrac.y;
    float t4 = (max.y - ray_origin.y) * dirfrac.y;
    float t5 = (min.z - ray_origin.z) * dirfrac.z;
    float t6 = (max.z - ray_origin.z) * dirfrac.z;
    
    float tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
    float tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));

    if (tmax < 0) // box behind us
        return false;
    
    if (tmin > tmax) // no intersection
        return false;

    *distance = tmin;
    return true;
}

__host__ __device__ AABB2::AABB2() {
    min = Vec2();
    max = Vec2();
}

__host__ __device__ AABB2::AABB2(Vec2 min, Vec2 max) {
    this->min = min;
    this->max = max;
}
