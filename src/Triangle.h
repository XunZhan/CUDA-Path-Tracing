//
// Created by zhanx on 11/16/2020.
//

#pragma once
#ifndef CUDA_PATH_TRACING_TRIANGLE_H
#define CUDA_PATH_TRACING_TRIANGLE_H

#include "Global.h"
#include "Ray.h"
#include "Intersection.h"
#include "Bounds3.h"
#include "math.h"
#include <cassert>
#include <array>
#include <cuda.h>
#include <curand_kernel.h>

class Triangle
{
public:
    Vector3f v0, v1, v2; // vertices A, B ,C , counter-clockwise order
    Vector3f e1, e2;     // 2 edges v1-v0, v2-v0;
    Vector3f t0, t1, t2; // texture coords
    Vector3f normal;
    float area;
    Material* m;

    __HOSTDEV__ Triangle(Vector3f _v0, Vector3f _v1, Vector3f _v2, Material* _m = nullptr)
            : v0(_v0), v1(_v1), v2(_v2), m(_m)
    {
        e1 = v1 - v0;
        e2 = v2 - v0;
        normal = normalize(crossProduct(e1, e2));
        area = crossProduct(e1, e2).norm()*0.5f;
    }

    __HOSTDEV__ Intersection getIntersection(Ray ray) {
        Intersection inter;

        if (dotProduct(ray.direction, normal) > 0)
            return inter;
        double u, v, t_tmp = 0;
        Vector3f pvec = crossProduct(ray.direction, e2);
        double det = dotProduct(e1, pvec);
        if (fabs(det) < FLT_EPSILON)
            return inter;

        double det_inv = 1. / det;
        Vector3f tvec = ray.origin - v0;
        u = dotProduct(tvec, pvec) * det_inv;
        if (u < 0 || u > 1)
            return inter;
        Vector3f qvec = crossProduct(tvec, e1);
        v = dotProduct(ray.direction, qvec) * det_inv;
        if (v < 0 || u + v > 1)
            return inter;
        t_tmp = dotProduct(e2, qvec) * det_inv;

        // TODO find ray triangle intersection
        if (t_tmp < 0)
            return inter;

        inter.happened = true;
        inter.coords = Vector3f(ray.origin + t_tmp * ray.direction);
        inter.normal = this->normal;
        inter.m = this->m;
        inter.obj = this;
        inter.distance = t_tmp;

        return inter;
    }

    __HOSTDEV__ void getSurfaceProperties(const Vector3f& P, const Vector3f& I,
                              const uint32_t& index, const Vector2f& uv,
                              Vector3f& N, Vector2f& st) const
    {
        N = normal;
        //        throw std::runtime_error("triangle::getSurfaceProperties not
        //        implemented.");
    }

    __HOSTDEV__ Vector3f evalDiffuseColor(const Vector2f&) const {
        return Vector3f(0.5, 0.5, 0.5);
    }

    inline Bounds3 getBounds() {
        return Union(Bounds3(v0, v1), v2);
    }

    void Sample(Intersection &pos, float &pdf){
        float x = std::sqrt(get_random_float()), y = get_random_float();
        pos.coords = v0 * (1.0f - x) + v1 * (x * (1.0f - y)) + v2 * (x * y);
        pos.normal = this->normal;
        pdf = 1.0f / area;
    }

    __device__ void CudaSample (Intersection &pos, float &pdf, curandState *state, int pid);

    __HOSTDEV__ float getArea(){
        return area;
    }

    bool hasEmit(){
        return m->hasEmission();
    }

    __device__ bool cudaHasEmit() {
        return m->cudaHasEmission();
    }
};


#endif //CUDA_PATH_TRACING_TRIANGLE_H
