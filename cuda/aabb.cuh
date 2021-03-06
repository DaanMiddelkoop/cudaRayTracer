#ifndef HEADER_AABB
#define HEADER_AABB

#include "vector.cuh"

class AABB3 {
public:
    __host__ __device__ AABB3();
    __host__ __device__ AABB3(Vec3 min, Vec3 max);

    Vec3 min;
    Vec3 max;

    __host__ __device__ void get_vertices(Vec3* vertices);
    __host__ __device__ AABB3 get_union(AABB3* other);
    __host__ __device__ bool intersects(Vec3 ray_origin, Vec3 ray_direction, float* distance);
};

class AABB2 {
public:
    __host__ __device__ AABB2();
    __host__ __device__ AABB2(Vec2 min, Vec2 max);

    Vec2 min;
    Vec2 max;
};


#endif