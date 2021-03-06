#include "plane.cuh"

__host__ __device__ Plane3::Plane3() {
    a = Vec3();
    b = Vec3();
    c = Vec3();
}

__host__ __device__ Plane3::Plane3(Vec3 a, Vec3 b, Vec3 c) {
    this->a = a;
    this->b = b;
    this->c = c;
}

__host__ __device__ float determinant(Vec3 a, Vec3 b, Vec3 c) {
    return (a.x * b.y * c.z) + (b.x * c.y * a.z) + (c.x * a.y * b.z) - (c.x * b.y * a.z) - (b.x * a.y * c.z) - (a.x * c.y * b.z);
}

__host__ __device__ float Plane3::outside(Vec3 x) {
    Vec3 b_prime = b - a;
    Vec3 c_prime = c - a;
    Vec3 x_prime = x - a;

    return determinant(b_prime, c_prime, x_prime);
}

__host__ __device__ Vec3 Plane3::intersection_point(Vec3 line_origin, Vec3 line_direction) {
    Vec3 plane_normal = (b - a).cross(c - a);

    float t  = (plane_normal.dot(a) - plane_normal.dot(line_origin)) / plane_normal.dot(line_direction.normalize());
    if (t < 0) // Falling of screen
        t = 99999999.0;
    return line_origin + ((line_direction).normalize() * t);
}