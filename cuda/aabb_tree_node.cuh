#ifndef HEADER_AABB_TREE_NODE
#define HEADER_AABB_TREE_NODE

#include "frustrum.cuh"
#include "block.cuh"
#include "aabb.cuh"
#include "vector.cuh"

class AABBTreeNode {
public:
    bool leaf;
    AABB3 bounding_box;
    void* c1;
    void* c2;
    void* parent = 0;

    __host__ __device__ AABBTreeNode();
    __host__ __device__ AABBTreeNode(AABBTreeNode* c1, AABBTreeNode* c2);
    __host__ __device__ AABBTreeNode(Block* child);

    __host__ __device__ bool is_leaf();
    __host__ __device__ AABBTreeNode* get_c1();
    __host__ __device__ AABBTreeNode* get_c2();
    __host__ __device__ Block* get_leaf();
    __host__ __device__ AABB3* get_bounding_box();
    __host__ __device__ void insert_block(Block* child);

    __host__ __device__ AABB2 trace_box(Frustrum view);
};

#endif