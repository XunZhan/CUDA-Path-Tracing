//
// Created by zhanx on 11/11/2020.
//

#pragma once
#ifndef CUDA_PATH_TRACING_AREALIGHT_H
#define CUDA_PATH_TRACING_AREALIGHT_H

#include "Vector.h"
#include "Light.h"
#include "global.h"

class AreaLight : public Light
{
public:
    AreaLight(const Vector3f &p, const Vector3f &i) : Light(p, i)
    {
        normal = Vector3f(0, -1, 0);
        u = Vector3f(1, 0, 0);
        v = Vector3f(0, 0, 1);
        length = 100;
    }

    Vector3f SamplePoint() const
    {
        auto random_u = get_random_float();
        auto random_v = get_random_float();
        return position + random_u * u + random_v * v;
    }

    float length;
    Vector3f normal;
    Vector3f u;
    Vector3f v;
};


#endif //CUDA_PATH_TRACING_AREALIGHT_H
