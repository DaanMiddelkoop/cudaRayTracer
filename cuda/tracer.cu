#include "tracer.cuh"

#include <stdio.h>
#include <assert.h>

#include "frustrum.cuh"
#include "aabb.cuh"

struct NodeChilds {
    AABBTreeNode* childs[2];
};

__host__ __device__ Tracer::Tracer(int width, int height) {
    this->width = width;
    this->height = height;

    this->aabb_tree = new AABBTree();
    this->depth_map = new float[width * height];
    this->node_queue = new AABBTreeNode*[10000];
}


__host__ __device__ void draw_pixel(Tracer* tracer, float* canvas, int x, int y, Vec3 color) {
    assert(x >= 0 && x < tracer->width && y >= 0 && y < tracer->height);

    canvas[(y * tracer->width + x) * 4 + 0] = color.x;
    canvas[(y * tracer->width + x) * 4 + 1] = color.y;
    canvas[(y * tracer->width + x) * 4 + 2] = color.z;
    canvas[(y * tracer->width + x) * 4 + 3] = 1.0;
}

__host__ __device__ void trace_leaf(Tracer* tracer, Block* block, float distance, float* canvas) {

    float light = 1.0 / distance;
    draw_pixel(tracer, canvas, 0, 0, block->color * (0.3 + min(0.7, light)));
}

__host__ __device__ void trace_node(Tracer* tracer, AABBTreeNode* current_node, Vec3 ray_origin, Vec3 ray_direction, float* canvas) {

    AABBTreeNode* node_tree[20];
    node_tree[0] = current_node;
    float max_depth = 9999999; // already found a pixel at this depth, no need to search deeper
    int index = 1;

    while (index > 0) {
        index -= 1;
        AABBTreeNode* node = node_tree[index];
        // printf("Inspecting node: %p\n", node);



        float depth;
        // printf("Ray(%f, %f, %f) (%f, %f, %f) does not intersect\n (%f, %f, %f) (%f, %f, %f)\n", ray_origin.x, ray_origin.y, ray_origin.z, ray_direction.z, ray_direction.y, ray_direction.z, node->bounding_box.min.x, node->bounding_box.min.y, node->bounding_box.min.z, node->bounding_box.max.x, node->bounding_box.max.y, node->bounding_box.max.z);
        if ((!node->bounding_box.intersects(ray_origin, ray_direction, &depth)) || depth > max_depth + 0.001) {
            continue;
        }

        // printf("hitting node at depth %f , max_depth: %f with aabb: ", depth, max_depth);
        // node->bounding_box.print();

        if (node->is_leaf()) {
            // printf("\n\nNode is leaf\n\n\n");
            trace_leaf(tracer, node->get_leaf(), depth, canvas);
            max_depth = depth;
        } else {

            if (index <= 18) {
                // Order for better performance
                float depth1 = -100.0f;
                bool hit_c1 = node->get_c1()->bounding_box.intersects(ray_origin, ray_direction, &depth1) && (depth <= max_depth + 0.001);

                float depth2 = -100.0f;
                bool hit_c2 = node->get_c2()->bounding_box.intersects(ray_origin, ray_direction, &depth2) && (depth <= max_depth + 0.001);

                // printf("depth of childs: %f, %f, hit: %i, %i\n", depth1, depth2, hit_c1, hit_c2);
                // printf("Bounding boxes: \n");
                // node->get_c1()->bounding_box.print();
                // node->get_c2()->bounding_box.print();
                
                if (hit_c1 && hit_c2) {
                    if (depth1 < depth2) {
                        node_tree[index++] = node->get_c2();
                        node_tree[index++] = node->get_c1();
                    } else {
                        node_tree[index++] = node->get_c1();
                        node_tree[index++] = node->get_c2();
                    }
                } else {
                    if (hit_c1) {
                        node_tree[index++] = node->get_c1();
                    }

                    if (hit_c2) {
                        node_tree[index++] = node->get_c2();
                    }
                }
            } else {
                printf("BAAAAAD\n");
            }
        }
    }
}

__global__ void trace_node_kernel(Tracer* tracer, Frustrum view, float* canvas) {

    AABBTreeNode* root = tracer->aabb_tree->root;
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_index >= 1920 * 1080) {
        return;
    }

    int px = thread_index % 1920;
    int py = thread_index / 1920;

    // if (!(px == 960 && py == 540))
    //     return;

    float x = (float)px / 1920.0 - 0.5;
    float y = (float)py / 1080.0 - 0.5;

    // ray defined by view
    Vec3 ray_origin = view.origin;
    Vec3 ray_direction = view.forward + (view.side * x) + (view.up * y);

    // printf("Ray: (%f, %f, %f), (%f, %f, %f)\n", ray_origin.x, ray_origin.y, ray_origin.z, ray_direction.x, ray_direction.y, ray_direction.z);

    canvas = &canvas[(py * tracer->width + px) * 4];
    trace_node(tracer, root, ray_origin, ray_direction.normalize(), canvas);
}

__host__ __device__ void Tracer::render_frame(int thread_index, Frustrum view, float* canvas) {

    // // Chance view to halve vectors for render purposes.
    // view = Frustrum(view.origin, view.forward, view.up * 0.5, view.side * 0.5);

    // node_queue[0] = this->aabb_tree->root;
    // queue_index = 1;

    printf("Rooooooooot: %p\n", this->aabb_tree->root);

    int threads = 1920 * 1080;
    int block_size = 1000;
    int blocks = (threads / block_size) + 1;

    trace_node_kernel<<<blocks, block_size>>>(this, view, canvas);
    // cudaDeviceSynchronize();
    cudaError_t error = cudaPeekAtLastError();
    if (error != cudaSuccess) {
        printf("GPUAssert: %s\n", cudaGetErrorString(error));
        assert(false);
    }
}