#ifndef HEADER_FRUSTRUM
#define HEADER_FRUSTRUM

#include "vector.cuh"
#include "aabb.cuh"
#include "plane.cuh"

class Frustrum {
public:
    __host__ __device__ Frustrum();
    __host__ __device__ Frustrum(Vec3 origin, Vec3 foward, Vec3 up, Vec3 side);

    __host__ __device__ bool intersects(AABB3* aabb);
    __host__ __device__ Vec3 recalculate_origin();
    __host__ __device__ Vec3 normal();
    __host__ __device__ void resize(AABB2 boundaries);

    Vec3 orig_a;
    Vec3 orig_b;
    Vec3 orig_c;
    Vec3 orig_d;

    Vec3 up;
    Vec3 side;
    Vec3 forward;

    Vec3 origin;

    Vec3 a;
    Vec3 b;
    Vec3 c;
    Vec3 d;
};

#endif