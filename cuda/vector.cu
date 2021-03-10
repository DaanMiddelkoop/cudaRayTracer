#include "vector.cuh"

#include <assert.h>
#include <stdio.h>

__host__ __device__ Vec3::Vec3() {
    x = 0.0;
    y = 0.0;
    z = 0.0;
}

__host__ __device__ Vec3::Vec3(float a, float b, float c) {
    x = a;
    y = b;
    z = c;
}

__host__ __device__ float Vec3::length() {
    return sqrt(x * x + y * y + z * z);
}

__host__ __device__ Vec3 Vec3::normalize() {
    float l = this->length();
    assert(l != 0);
    return Vec3(x / l, y / l, z / l);
}

__host__ __device__ Vec3 Vec3::minimum(Vec3 other) {
    float mx = min(x, other.x);
    float my = min(y, other.y);
    float mz = min(z, other.z);
    return Vec3(mx, my, mz);
}

__host__ __device__ Vec3 Vec3::maximum(Vec3 other) {
    float mx = max(x, other.x);
    float my = max(y, other.y);
    float mz = max(z, other.z);
    return Vec3(mx, my, mz);
}

__host__ __device__ float Vec3::dot(Vec3 other) {
    return x * other.x + y * other.y + z * other.z;
}

__host__ __device__ Vec3 Vec3::cross(Vec3 other) {
    return Vec3(y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x);
}

__host__ __device__ void Vec3::print() {
    printf("V(%f, %f, %f)", x, y, z);
}

__host__ __device__ Vec3 Vec3::rotate(Vec3 axis, float rotation) {
    return (*this * cos(rotation)) + (axis.cross(*this) * sin(rotation)) + (axis * (axis.dot(*this)) * (1 - cos(rotation)));
}




__host__ __device__ Vec2::Vec2() {
    x = 0.0;
    y = 0.0;
}

__host__ __device__ Vec2::Vec2(float x, float y) {
    this->x = x;
    this->y = y;
}

__host__ __device__ Vec2 Vec2::minimum(Vec2 other) {
    float mx = min(x, other.x);
    float my = min(y, other.y);
    return Vec2(mx, my);
}

__host__ __device__ Vec2 Vec2::maximum(Vec2 other) {
    float mx = max(x, other.x);
    float my = max(y, other.y);
    return Vec2(mx, my);
}