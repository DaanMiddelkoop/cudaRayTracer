#include "aabb_tree.cuh"

#include <stdio.h>

AABBTree::AABBTree() {}

void AABBTree::receive_world(Block* blocks, int amount) {
    construct_tree(blocks, amount);
}


void AABBTree::construct_tree(Block* blocks, int amount) {
    this->root = new AABBTreeNode(&blocks[0]);
    for (int i = 1; i < amount; i++) {

        int depth = 0;
        AABBTreeNode* current_node = root;
        while (!current_node->is_leaf()) {
            depth += 1;
            current_node->bounding_box = current_node->bounding_box.get_union(&blocks[i].aabb);

            float surface1_growth = current_node->get_c1()->bounding_box.get_union(&blocks[i].aabb).surface() - current_node->get_c1()->bounding_box.surface();
            float surface2_growth = current_node->get_c2()->bounding_box.get_union(&blocks[i].aabb).surface() - current_node->get_c2()->bounding_box.surface();
    
            if (surface1_growth < surface2_growth) {
                // insert into c1
                current_node = current_node->get_c1();
            } else {
                current_node = current_node->get_c2();
            }
        }
        
        printf("adding block: %i at depth: %i\n", i, depth);
        current_node->insert_block(&blocks[i]);
    }
}