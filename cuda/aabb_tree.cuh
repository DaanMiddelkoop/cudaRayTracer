#ifndef HEADER_AABB_TREE
#define HEADER_AABB_TREE

#include "block.cuh"
#include "aabb_tree_node.cuh"

class AABBTree {
public:
    __host__ __device__ AABBTree();

    AABBTreeNode* root;

    __host__ __device__ void receive_world(Block* blocks, int amount);
    __host__ __device__ void construct_tree(Block* blocks, int amount);
};

#endif