#include "block.cuh"

Block::Block() {
    
}

Block::Block(AABB3 aabb, Vec3 color) {
    this->aabb = aabb;
    this->color = color;
}

AABB3* Block::get_bounding_box() {
    return &this->aabb;
}

