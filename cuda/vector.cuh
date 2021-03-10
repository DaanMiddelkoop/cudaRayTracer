#ifndef HEADER_VECTOR
#define HEADER_VECTOR

class Vec3 {
public:
    __host__ __device__ Vec3();
    __host__ __device__ Vec3(float e0, float e1, float e2);

    __host__ __device__ Vec3 minimum(Vec3 other);
    __host__ __device__ Vec3 maximum(Vec3 other);

    float x;
    float y;
    float z;

    __host__ __device__ float length();
    __host__ __device__ Vec3 normalize();
    __host__ __device__ float dot(Vec3 other);
    __host__ __device__ Vec3 cross(Vec3 other);
    __host__ __device__ void print();
    __host__ __device__ Vec3 rotate(Vec3 axis, float radians);

    __host__ __device__ Vec3& operator+=(const Vec3& a) {
        this->x += a.x;
        this->y += a.y;
        this->z += a.z;
        return *this;
    }

    __host__ __device__ Vec3& operator-=(const Vec3& a) {
        this->x -= a.x;
        this->y -= a.y;
        this->z -= a.z;
        return *this;
    }
};

class Vec2 {
public:
    __host__ __device__ Vec2();
    __host__ __device__ Vec2(float x, float y);

    float x; 
    float y;

    __host__ __device__ Vec2 minimum(Vec2 other);
    __host__ __device__ Vec2 maximum(Vec2 other);
};

__host__ __device__ inline Vec3 operator+(const Vec3& a, const Vec3& b) {
    return Vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline Vec3 operator-(const Vec3& a, const Vec3& b) {
    return Vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline Vec3 operator*(const Vec3& b, float a) {
    return Vec3(b.x * a, b.y * a, b.z * a);
}

__host__ __device__ inline Vec3 operator*(float a, const Vec3& b) {
    return b * a;
}

__host__ __device__ inline Vec3 operator-=(const Vec3& a, const Vec3& b) {
    return Vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline Vec2 operator+(const Vec2& a, const Vec2& b) {
    return Vec2(a.x + b.x, a.y + b.y);
}

__host__ __device__ inline Vec2 operator-(const Vec2& a, const Vec2& b) {
    return Vec2(a.x - b.x, a.y - b.y);
}

__host__ __device__ inline Vec2 operator*(const Vec2& a, float b) {
    return Vec2(a.x * b, a.y * b);
}

__host__ __device__ inline Vec2 operator*(float a, Vec2& b) {
    return b * a;
}

#endif