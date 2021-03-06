#include "aabb_tree.cuh"

#include <stdio.h>

AABBTree::AABBTree() {}

void AABBTree::receive_world(Block* blocks, int amount) {
    construct_tree(blocks, amount);
}


void AABBTree::construct_tree(Block* blocks, int amount) {
    this->root = new AABBTreeNode(&blocks[0]);
    for (int i = 1; i < amount; i++) {
        root->insert_block(&blocks[i]);
    }
}