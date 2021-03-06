#include "aabb_tree_node.cuh"

#include <assert.h>
#include <stdio.h>

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
    // This code is unimplemented oopsie.
    assert(false);
}

AABB2 AABBTreeNode::trace_box(Frustrum view) {
    Vec3 vertices[8];
    Vec2 projections[8];
    get_bounding_box()->get_vertices(vertices);

    Plane3 view_plane = Plane3(view.orig_a, view.orig_b, view.orig_c);
    Vec3 view_plane_center = (view.orig_a + view.orig_c) * 0.5;
    Vec3 up = (view.orig_b + view.orig_c) * 0.5 - view_plane_center;
    Vec3 side = (view.orig_c + view.orig_d) * 0.5 - view_plane_center;
    printf("Side vector size: %f\n", side.length());
    printf("view origins:");
    view.orig_a.print();
    view.orig_b.print();
    view.orig_c.print();
    view.orig_d.print();
    printf("\n");

    for (int i = 0; i < 8; i++) {
        Vec3 intersection_point = view_plane.intersection_point(view.origin, vertices[i] - view.origin) - view_plane_center;
        // Project up vector onto intersection line.
        printf("Ray: ");
        view.origin.print();
        (vertices[i] - view.origin).print();
        printf("\nIntersection point: %f, %f, %f\n", intersection_point.x, intersection_point.y, intersection_point.z);
        float y = intersection_point.dot(up.normalize()) / up.length();
        float x = intersection_point.dot(side.normalize()) / side.length();
        printf("screen_point: %f, %f\n", x, y);
        projections[i] = Vec2(min(0.5, max(-0.5, x)), min(0.5, max(-0.5, y))); // Limit vertices outside screen to be on the edge, otherwise weird stuff happens.
    }

    Vec2 minimum = projections[0];
    Vec2 maximum = projections[0];

    for (int i = 1; i < 8; i++) {
        printf("projection %i: (%f, %f)\n", i, projections[i].x, projections[i].y);
        printf("minmax: (%f, %f) (%f, %f)\n", minimum.x, minimum.y, maximum.x, maximum.y);
        minimum = minimum.minimum(projections[i]);
        maximum = maximum.maximum(projections[i]);
    }

    return AABB2(minimum, maximum);
}