#ifndef HEADER_TRACER
#define HEADER_TRACER

#include "aabb_tree.cuh"
#include "frustrum.cuh"

class Tracer {
public:
    int width;
    int height;

    __host__ __device__ Tracer(int width, int height);

    __host__ __device__ void render_frame(int thread_index, Frustrum view, float* canvas);

    AABBTree* aabb_tree;
    float* depth_map;
};

#endif
