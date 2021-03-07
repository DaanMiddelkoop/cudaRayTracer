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
}


__host__ __device__ void draw_pixel(Tracer* tracer, float* canvas, int x, int y, Vec3 color) {
    assert(x >= 0 && x < tracer->width && y >= 0 && y < tracer->height);

    canvas[(y * tracer->width + x) * 4 + 0] = color.x;
    canvas[(y * tracer->width + x) * 4 + 1] = color.y;
    canvas[(y * tracer->width + x) * 4 + 2] = color.z;
    canvas[(y * tracer->width + x) * 4 + 3] = 1.0;
}

__global__ void trace_pixel(Tracer* tracer, Block* block, Frustrum view, int min_x, int min_y, int max_x, int max_y, float* canvas) {
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    int px = thread_index % (max_x - min_x);
    int py = thread_index / (max_x - min_x);

    if (px < 0 || px >= tracer->width || py < 0 || py >= tracer->height)
        return;
    
    float x_factor = (float)px / (float)(max_x - min_x);
    float y_factor = (float)py / (float)(max_y - min_y);
    Vec3 screen_pos = ((view.orig_b - view.orig_a) * y_factor) + ((view.orig_d - view.orig_a) * x_factor) + view.orig_a;

    Vec3 ray_direction = screen_pos - view.origin;

    float distance;
    if (block->aabb.intersects(view.origin, ray_direction, &distance)) {
        float light = 1000.0 / (distance * distance);
        draw_pixel(tracer, canvas, px + min_x, py + min_y, block->color * light);
    }
}

__host__ __device__ void trace_leaf(Tracer* tracer, Block* block, Frustrum view, float* canvas) {
    Vec3 side_normalized = view.side.normalize();
    Vec3 up_normalized = view.up.normalize();;
    Vec3 screen_center = view.origin + view.forward;
    float side_length = view.side.length() * 2.0;
    float up_length = view.up.length() * 2.0;

    AABB2 screen_bounding_box = AABB2(
        Vec2(
            (view.orig_a - screen_center).dot(side_normalized) / side_length,
            (view.orig_a - screen_center).dot(up_normalized) / up_length
        ),
        Vec2(
            (view.orig_c - screen_center).dot(side_normalized) / side_length,
            (view.orig_c - screen_center).dot(up_normalized) / up_length
        )
    );

    screen_bounding_box.max = screen_bounding_box.max.minimum(Vec2(0.5, 0.5));
    screen_bounding_box.min = screen_bounding_box.min.maximum(Vec2(-0.5, -0.5));
    
    int min_x = (screen_bounding_box.min.x + 0.5) * tracer->width;
    int min_y = (screen_bounding_box.min.y + 0.5) * tracer->height;
    int max_x = (screen_bounding_box.max.x + 0.5) * tracer->width;
    int max_y = (screen_bounding_box.max.y + 0.5) * tracer->height;

    int threads = (max_x - min_x) * (max_y - min_y);
    int block_size = 512;
    int blocks = (threads / 512) + 1;

    trace_pixel<<<blocks, block_size>>>(tracer, block, view, min_x, min_y, max_x, max_y, canvas);
}


__global__ void trace_node(Tracer* tracer, NodeChilds childs, Frustrum view, float* canvas) {
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    AABBTreeNode* current_node = childs.childs[thread_index];

    AABB2 bounding_box = current_node->trace_box(view);
    view.resize(bounding_box);

    if (current_node->is_leaf()) {
        trace_leaf(tracer, current_node->get_leaf(), view, canvas);
    } else {
        NodeChilds childs;
        childs.childs[0] = current_node->get_c1();
        childs.childs[1] = current_node->get_c2();
        trace_node<<<1, 2>>>(tracer, childs, view, canvas);
    }
}

__host__ __device__ void Tracer::render_frame(int thread_index, Frustrum view, float* canvas) {
    NodeChilds childs;
    childs.childs[0] = this->aabb_tree->root;
    
    // Chance view to halve vectors for render purposes.
    view = Frustrum(view.origin, view.forward, view.up * 0.5, view.side * 0.5);
    
    trace_node<<<1, 1>>>(this, childs, view, canvas);
}