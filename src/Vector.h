//
// Created by zhanx on 11/12/2020.
//
#pragma once
#ifndef CUDA_PATH_TRACING_VECTOR_H
#define CUDA_PATH_TRACING_VECTOR_H

#ifdef __CUDACC__
#define __HOSTDEV__ __host__ __device__
#else
#define __HOSTDEV__
#endif

#include <iostream>
#include <algorithm>
#include <math.h>
#include <cuda.h>
#include <curand_kernel.h>

class Vector3f {
public:
    float x, y, z;

    __HOSTDEV__ Vector3f() : x(0), y(0), z(0) {}
    __HOSTDEV__ Vector3f(float xx) : x(xx), y(xx), z(xx) {}
    __HOSTDEV__ Vector3f(float xx, float yy, float zz) : x(xx), y(yy), z(zz) {}
    __HOSTDEV__ Vector3f operator * (const float &r) const { return Vector3f(x * r, y * r, z * r); }
    __HOSTDEV__ Vector3f operator / (const float &r) const { return Vector3f(x / r, y / r, z / r); }

    __HOSTDEV__ float norm() {return sqrt(x * x + y * y + z * z);}
    __HOSTDEV__ Vector3f normalized() {
        float n = std::sqrt(x * x + y * y + z * z);
        return Vector3f(x / n, y / n, z / n);
    }

    __HOSTDEV__ Vector3f operator * (const Vector3f &v) const { return Vector3f(x * v.x, y * v.y, z * v.z); }
    __HOSTDEV__ Vector3f operator - (const Vector3f &v) const { return Vector3f(x - v.x, y - v.y, z - v.z); }
    __HOSTDEV__ Vector3f operator + (const Vector3f &v) const { return Vector3f(x + v.x, y + v.y, z + v.z); }
    __HOSTDEV__ Vector3f operator - () const { return Vector3f(-x, -y, -z); }
    __HOSTDEV__ Vector3f& operator += (const Vector3f &v) { x += v.x, y += v.y, z += v.z; return *this; }
    __HOSTDEV__ friend Vector3f operator * (const float &r, const Vector3f &v)
    { return Vector3f(v.x * r, v.y * r, v.z * r); }
//    __HOSTDEV__ friend std::ostream & operator << (std::ostream &os, const Vector3f &v)
//    { return os << v.x << ", " << v.y << ", " << v.z; }
    __HOSTDEV__ double       operator[](int index) const;
    __HOSTDEV__ double&      operator[](int index);


    static Vector3f Min(const Vector3f &p1, const Vector3f &p2) {
        return Vector3f(std::min(p1.x, p2.x), std::min(p1.y, p2.y),
                        std::min(p1.z, p2.z));
    }

    static Vector3f Max(const Vector3f &p1, const Vector3f &p2) {
        return Vector3f(std::max(p1.x, p2.x), std::max(p1.y, p2.y),
                        std::max(p1.z, p2.z));
    }
};

__HOSTDEV__ inline double Vector3f::operator[](int index) const {
    return (&x)[index];
}


class Vector2f
{
public:
    float x, y;

    __HOSTDEV__ Vector2f() : x(0), y(0) {}
    __HOSTDEV__ Vector2f(float xx) : x(xx), y(xx) {}
    __HOSTDEV__ Vector2f(float xx, float yy) : x(xx), y(yy) {}
    __HOSTDEV__ Vector2f operator * (const float &r) const { return Vector2f(x * r, y * r); }
    __HOSTDEV__ Vector2f operator + (const Vector2f &v) const { return Vector2f(x + v.x, y + v.y); }

};

__HOSTDEV__ inline Vector3f lerp(const Vector3f &a, const Vector3f& b, const float &t)
{ return a * (1 - t) + b * t; }

__HOSTDEV__ inline Vector3f normalize(const Vector3f &v)
{
    float mag2 = v.x * v.x + v.y * v.y + v.z * v.z;
    if (mag2 > 0) {
        float invMag = 1 / sqrtf(mag2);
        return Vector3f(v.x * invMag, v.y * invMag, v.z * invMag);
    }

    return v;
}

__HOSTDEV__ inline float dotProduct(const Vector3f &a, const Vector3f &b)
{ return a.x * b.x + a.y * b.y + a.z * b.z; }

__HOSTDEV__ inline Vector3f crossProduct(const Vector3f &a, const Vector3f &b)
{
    return Vector3f(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
    );
}


#endif //CUDA_PATH_TRACING_VECTOR_H
