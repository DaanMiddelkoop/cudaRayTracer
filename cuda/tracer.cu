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

__global__ void trace_pixel(Tracer* tracer, Block* block, Frustrum view, int offset_x, int offset_y, int width, int height, float* canvas) {
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    int px = thread_index % width;
    int py = thread_index / width;

    if (px < 0 || px >= width || py < 0 || py >= height)
        return;
    
    float x_factor = (float)px / (float)width;
    float y_factor = (float)py / (float)height;
    Vec3 screen_pos = ((view.orig_b - view.orig_a) * y_factor) + ((view.orig_d - view.orig_a) * x_factor) + view.orig_a;
    // printf("Screen pos: (%f, %f, %f)\n", screen_pos.x, screen_pos.y, screen_pos.z);
    // printf("View orig a: %f, %f, %f\n", view.orig_a.x, view.orig_a.y, view.orig_a.z);

    Vec3 ray_direction = screen_pos - view.origin;

    if (px == 0 && py == 540) {
        printf("x_factor: %f\n", x_factor);
        
        Vec3 a_d = view.orig_d - view.orig_a;
        printf("AD vector: %f, %f, %f\n", a_d.x, a_d.y, a_d.z);
        printf("Orig a: %f, %f, %f\n", view.orig_a.x, view.orig_a.y, view.orig_a.z);
        printf("Orig d: %f, %f, %f\n", view.orig_d.x, view.orig_d.y, view.orig_d.z);

        printf("Ray direction: %f, %f, %f\n", ray_direction.x, ray_direction.y, ray_direction.z);
    }
    // printf("Ray: (%f, %f, %f) (%f, %f, %f)\n", view.origin.x, view.origin.y, view.origin.z, ray_direction.x, ray_direction.y, ray_direction.z);

    float distance;
    if (block->aabb.intersects(view.origin, ray_direction, &distance)) {
        float light = 300.0 / (distance * distance);
        draw_pixel(tracer, canvas, px + offset_x, py + offset_y, block->color * light);
    }
}

__host__ __device__ void trace_leaf(Tracer* tracer, Block* block, Frustrum view, float* canvas) {
    Vec3 side_normalized = view.side.normalize();
    Vec3 up_normalized = view.up.normalize();;
    Vec3 screen_center = view.forward + view.origin;
    float side_length = view.side.length() * 0.5;
    float up_length = view.up.length() * 0.5;

    printf("Orig_a: %f, %f, %f\n", view.orig_a.x, view.orig_a.y, view.orig_a.z);

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
    
    int offset_x = (screen_bounding_box.min.x + 0.5) * tracer->width;
    int offset_y = (screen_bounding_box.min.y + 0.5) * tracer->height;
    int width = (screen_bounding_box.max.x + 0.5) * tracer->width - offset_x - 1;
    int height = (screen_bounding_box.max.y + 0.5) * tracer->height - offset_y - 1;

    int threads = width * height;
    int block_size = 512;
    int blocks = (threads / 512) + 1;

    draw_pixel(tracer, canvas, offset_x, offset_y, Vec3(1.0, 0.0, 0.0));
    draw_pixel(tracer, canvas, offset_x, offset_y + height, Vec3(1.0, 0.0, 0.0));
    draw_pixel(tracer, canvas, offset_x + width, offset_y, Vec3(1.0, 0.0, 0.0));
    draw_pixel(tracer, canvas, offset_x + width, offset_y + height, Vec3(1.0, 0.0, 0.0));

    printf("AABB hit block (%f, %f, %f, %f)\n", screen_bounding_box.min.x, screen_bounding_box.min.y, screen_bounding_box.max.x, screen_bounding_box.max.y);

    trace_pixel<<<blocks, block_size>>>(tracer, block, view, offset_x, offset_y, width, height, canvas);
}


__global__ void trace_node(Tracer* tracer, NodeChilds childs, Frustrum view, float* canvas) {
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    AABBTreeNode* current_node = childs.childs[thread_index];

    AABB2 bounding_box = current_node->trace_box(view);
    printf("Resulting bounding box: (%f, %f), (%f, %f)\n", bounding_box.min.x, bounding_box.min.y, bounding_box.max.x, bounding_box.max.y);
    view.resize(bounding_box);

    if (current_node->is_leaf()) {
        trace_leaf(tracer, current_node->get_leaf(), view, canvas);
    } else {
        NodeChilds childs;
        childs.childs[0] = current_node->get_c1();
        childs.childs[1] = current_node->get_c2();
        trace_node<<<1, 2>>>(tracer, childs, view, canvas);
    }

    int offset_x = (bounding_box.min.x + 0.5) * tracer->width;
    int offset_y = (bounding_box.min.y + 0.5) * tracer->height;
    int width = (bounding_box.max.x + 0.5) * tracer->width;
    int height = (bounding_box.max.y + 0.5) * tracer->height;
    printf("SCREEN BB: %i, %i, %i, %i\n", offset_x, offset_y, width, height);
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            // draw_pixel(tracer, canvas, x + offset_x, y + offset_y, Vec3(0.0, 0.0, 1.0));
        }
    }
}

__host__ __device__ void Tracer::render_frame(int thread_index, Frustrum view, float* canvas) {
    NodeChilds childs;
    childs.childs[0] = this->aabb_tree->root;
    
    // Chance view to halve vectors for render purposes.
    view = Frustrum(view.origin, view.forward, view.up * 0.5, view.side * 0.5);
    
    trace_node<<<1, 1>>>(this, childs, view, canvas);
}