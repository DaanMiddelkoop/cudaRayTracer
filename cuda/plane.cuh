#ifndef HEADER_PLANE
#define HEADER_PLANE

#include "vector.cuh"

class Plane3 {
public:
    __host__ __device__ Plane3();
    __host__ __device__ Plane3(Vec3 a, Vec3 b, Vec3 c);

    Vec3 a;
    Vec3 b;
    Vec3 c;

    __host__ __device__ float outside(Vec3 point);
    __host__ __device__ Vec3 intersection_point(Vec3 line_origin, Vec3 line_direction);
};

#endif