#include "trace_manager.cuh"

#include <stdio.h>
#include <assert.h>
#include <new>
#include <stdint.h>
#include <algorithm>

__global__ void tracer_init_kernel(int width, int height, Tracer* tracer) {
    tracer = new((void*)tracer) Tracer(width, height);

}
__host__ TraceManager::TraceManager(int width, int height) {
    this->width = width;
    this->height = height;
    cpu_canvas = new float[width * height * 4 * sizeof(float)];

    cudaDeviceSetLimit(cudaLimitStackSize, 1024);

    cudaMalloc(&tracer, sizeof(Tracer));
    cudaMalloc(&canvas, width * height * 4 * sizeof(float));
    cudaMalloc(&blocks, sizeof(Block));
    cudaMalloc(&nodes, sizeof(AABBTreeNode));


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

__global__ void receive_aabb_tree_kernel(Tracer* tracer, AABBTreeNode* nodes, int64_t node_offset, int64_t block_offset, int root_index, int amount) {
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_index > amount)
        return;

    if (thread_index == 0) {
        tracer->aabb_tree->root = &nodes[root_index];
    }

    if (nodes[thread_index].is_leaf()) {
        nodes[thread_index].c1 = (uint8_t*)nodes[thread_index].c1 + block_offset;
    } else {
        nodes[thread_index].c1 = (uint8_t*)nodes[thread_index].c1 + node_offset;
        nodes[thread_index].c2 = (uint8_t*)nodes[thread_index].c2 + node_offset;
    }
}

uint64_t calculate_z_order(AABB3 elem) {
    Vec3 center = elem.center();
    uint32_t x = center.x;
    uint32_t y = center.y;
    uint32_t z = center.z;

    //printf("space curve: %u, %u\n", x, z);

    uint64_t result = 0;


    for (int i = 0; i < 20; i++) {
        uint64_t x_bit = ((x & (1 << 19)) >> 19);
        uint64_t y_bit = ((y & (1 << 19)) >> 19);
        uint64_t z_bit = ((z & (1 << 19)) >> 19);

        result |= x_bit << (0ULL);
        result |= y_bit << (1ULL);
        result |= z_bit << (2ULL);

        x <<= 1;
        y <<= 1;
        z <<= 1;

        result <<= 3;
    }

    return result;
}

bool compare_z_order(AABBTreeNode t1, AABBTreeNode t2) {
    return calculate_z_order(t1.bounding_box) < calculate_z_order(t2.bounding_box);
}

__host__ void TraceManager::upload_world(Block* blocks, int amount) {
    AABBTreeNode* nodes = new AABBTreeNode[amount * 2];
    AABBTreeNode* node_ptr = nodes;

    int cnt = amount;

    for (int i = 0; i < amount; i++) {
        nodes[i] = AABBTreeNode(&blocks[i]);
    }


    std::sort(node_ptr, node_ptr + cnt, compare_z_order);
    while (cnt >= 1) {
        // Sort for approximate clustering.
        printf("Starting new layer at: %i\n", node_ptr - nodes);

        // Built tree layer.
        for (int i = 0; i < cnt / 2; i++) {
            node_ptr[cnt + i] = AABBTreeNode(&node_ptr[i * 2], &node_ptr[i * 2 + 1]);
            node_ptr[i * 2].parent = &node_ptr[cnt + i];
            node_ptr[i * 2 + 1].parent = &node_ptr[cnt + i];

            printf("Array: %i, Index: %i, cnt: %i, surface: %f\n", node_ptr - nodes, i, cnt, node_ptr[cnt + i].bounding_box.surface());
        }

        if (cnt == 1)
            break;

        if (cnt % 2 == 0) {
            node_ptr = &node_ptr[cnt];
            cnt = cnt / 2;
        } else {
            node_ptr = &node_ptr[cnt - 1];
            cnt = (cnt / 2) + 1;
        }

    }

    // for (int i = 0; i < amount * 2; i++) {
    //     printf("aabb: %i -- surface area: %f -- ZCurve: %llu -- center: ", i, nodes[i].bounding_box.surface(), calculate_z_order(nodes[i].bounding_box));
    //     nodes[i].bounding_box.center().print();
    //     printf(" -- ");
    //     nodes[i].bounding_box.print();
    // }

    // printf("bounding of 19997: ");
    // nodes[8190].bounding_box.print();
    // printf("bounding box of its children:\n");
    // nodes[8190].get_c1()->bounding_box.print();
    // nodes[8190].get_c2()->bounding_box.print();


    cudaError_t error;

    cudaFree(this->blocks);
    cudaFree(this->nodes);
    
    cudaMalloc(&this->blocks, amount * sizeof(Block));
    cudaMalloc(&this->nodes, amount * 2 * sizeof(AABBTreeNode));

    cudaMemcpy(this->blocks, blocks, amount * sizeof(Block), cudaMemcpyHostToDevice);
    cudaMemcpy(this->nodes, nodes, amount * 2 * sizeof(AABBTreeNode), cudaMemcpyHostToDevice);


    int64_t node_offset = (int64_t)this->nodes - (int64_t)nodes;
    int64_t block_offset = (int64_t)this->blocks - (int64_t)blocks;
    
    printf("nodes: %p, gpu_nodes: %p, nodes + offset: %p\n", nodes, this->nodes, (void*)nodes + node_offset);

    int block_size = 512;
    int blcks = ((amount * 2) / block_size) + 1;
    

    receive_aabb_tree_kernel<<<blcks, block_size>>>(tracer, this->nodes, node_offset, block_offset, node_ptr - nodes, amount * 2 - 1);
    cudaDeviceSynchronize();
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) {
        printf("GPUAssert: %s\n", cudaGetErrorString(error));
        assert(false);
    }
    
}