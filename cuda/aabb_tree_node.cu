#include "aabb_tree_node.cuh"

#include <assert.h>
#include <stdio.h>
#include <new>

AABBTreeNode::AABBTreeNode() {
    this->leaf = true;
    this->bounding_box = AABB3();
}

AABBTreeNode::AABBTreeNode(AABBTreeNode* c1, AABBTreeNode* c2) {
    this->leaf = false;
    this->c1 = c1;
    this->c2 = c2;
    this->bounding_box = get_c1()->get_bounding_box()->get_union(get_c2()->get_bounding_box());
}

AABBTreeNode::AABBTreeNode(Block* child) {
    this->leaf = true;
    this->bounding_box = *child->get_bounding_box();
    this->c1 = child;
}

bool AABBTreeNode::is_leaf() {
    return this->leaf;
}

AABBTreeNode* AABBTreeNode::get_c1() {
    assert(!this->leaf);
    return (AABBTreeNode*) this->c1;
}

AABBTreeNode* AABBTreeNode::get_c2() {
    assert(!this->leaf);
    return (AABBTreeNode*) this->c2;
}

Block* AABBTreeNode::get_leaf() {
    assert(this->leaf);
    return (Block*) this->c1;
}

AABB3* AABBTreeNode::get_bounding_box() {
    return &this->bounding_box;
}

void AABBTreeNode::insert_block(Block* child) {
    if (is_leaf()) {
        AABBTreeNode* a = new AABBTreeNode(this->get_leaf());
        AABBTreeNode* b = new AABBTreeNode(child);
        new (this) AABBTreeNode(a, b);
    } else {
        assert(false);
    }
}

AABB2 AABBTreeNode::trace_box(Frustrum view) {
    Vec3 vertices[8];
    Vec2 projections[8];
    get_bounding_box()->get_vertices(vertices);

    Plane3 view_plane = Plane3(view.orig_a, view.orig_b, view.orig_c);
    Vec3 view_plane_center = (view.orig_a + view.orig_c) * 0.5;
    Vec3 up = (view.orig_b - view.orig_a);
    Vec3 side = (view.orig_d - view.orig_a);

    for (int i = 0; i < 8; i++) {
        Vec3 intersection_point = view_plane.intersection_point(view.origin, vertices[i] - view.origin);
        Vec3 relative_screen_point = intersection_point - view_plane_center;
        // Project up vector onto intersection line.
        float y = relative_screen_point.dot(up.normalize()) / up.length();
        float x = relative_screen_point.dot(side.normalize()) / side.length();
        projections[i] = Vec2(min(0.5, max(-0.5, x)), min(0.5, max(-0.5, y))); // Limit vertices outside screen to be on the edge, otherwise weird stuff happens.
    }

    Vec2 minimum = projections[0];
    Vec2 maximum = projections[0];

    for (int i = 1; i < 8; i++) {
        minimum = minimum.minimum(projections[i]);
        maximum = maximum.maximum(projections[i]);
    }

    return AABB2(minimum, maximum);
}