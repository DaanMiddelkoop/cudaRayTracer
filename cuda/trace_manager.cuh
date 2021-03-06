#ifndef HEADER_TRACE_MANAGER
#define HEADER_TRACE_MANAGER

#include "frustrum.cuh"
#include "tracer.cuh"
#include "block.cuh"

class TraceManager {
public:
    TraceManager(int width, int height);

    __host__ float* render_frame(Frustrum view);

    __host__ void upload_world(Block* blocks, int amount);

    int width;
    int height;

    Tracer* tracer;
    float* cpu_canvas;
    float* canvas;

    Block* blocks;
};

#endif