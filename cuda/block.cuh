#ifndef HEADER_BLOCK
#define HEADER_BLOCK

#include "aabb.cuh"
#include "vector.cuh"

class Block {
public:
    __host__ __device__ Block();
    __host__ __device__ Block(AABB3 aabb, Vec3 color);

    AABB3 aabb;
    Vec3 color;

    __host__ __device__ AABB3* get_bounding_box();
};

#endif