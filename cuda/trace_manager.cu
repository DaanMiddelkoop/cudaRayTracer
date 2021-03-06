#include "trace_manager.cuh"

#include <stdio.h>
#include <assert.h>
#include <new>

__global__ void tracer_init_kernel(int width, int height, Tracer* tracer) {
    tracer = new((void*)tracer) Tracer(width, height);

}
__host__ TraceManager::TraceManager(int width, int height) {
    this->width = width;
    this->height = height;
    cpu_canvas = new float[width * height * 4 * sizeof(float)];

    cudaMalloc(&tracer, sizeof(Tracer));
    cudaMalloc(&canvas, width * height * 4 * sizeof(float));
    cudaMalloc(&blocks, sizeof(Block));


    tracer_init_kernel<<<1, 1>>>(width, height, this->tracer);
    cudaDeviceSynchronize();
    cudaError_t error = cudaPeekAtLastError();
    if (error != cudaSuccess) {
        printf("GPUAssert: %s\n", cudaGetErrorString(error));
        assert(false);
    }
}

__global__ void tracer_render_frame_kernel(Tracer* tracer, Frustrum view, float* canvas) {
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    tracer->render_frame(thread_index, view, canvas);
}

__global__ void tracer_render_clear_frame_kernel(Tracer* tracer, float* canvas) {
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    int px = thread_index % tracer->width;
    int py = thread_index / tracer->width;

    canvas[(py * tracer->width + px) * 4 + 0] = 0.0;
    canvas[(py * tracer->width + px) * 4 + 1] = 0.0;
    canvas[(py * tracer->width + px) * 4 + 2] = 0.0;
    canvas[(py * tracer->width + px) * 4 + 3] = 1.0;
}

__host__ float* TraceManager::render_frame(Frustrum view) {

    int threads = width * height;
    int block_size = 512;
    int blocks = (threads / 512) + 1;
    tracer_render_clear_frame_kernel<<<blocks, block_size>>>(tracer, canvas);

    tracer_render_frame_kernel<<<1, 1>>>(tracer, view, canvas);

    cudaDeviceSynchronize();
    cudaError_t error = cudaPeekAtLastError();
    if (error != cudaSuccess) {
        printf("GPUAssert: %s\n", cudaGetErrorString(error));
        assert(false);
    }
    cudaMemcpy(cpu_canvas, canvas, width * height * 4 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) {
        printf("GPUAssert: %s\n", cudaGetErrorString(error));
        assert(false);
    }
    return cpu_canvas;
}

__global__ void receive_world_kernel(Tracer* tracer, Block* blocks, int amount) {
    tracer->aabb_tree->receive_world(blocks, amount);
}

__host__ void TraceManager::upload_world(Block* blocks, int amount) {
    cudaFree(this->blocks);
    cudaMalloc(&this->blocks, amount * sizeof(Block));
    cudaMemcpy(this->blocks, blocks, amount * sizeof(Block), cudaMemcpyHostToDevice);
    receive_world_kernel<<<1, 1>>>(tracer, this->blocks, amount);
}