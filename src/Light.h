//
// Created by zhanx on 11/11/2020.
//

#pragma once
#ifndef CUDA_PATH_TRACING_LIGHT_H
#define CUDA_PATH_TRACING_LIGHT_H

#include "Vector.h"

class Light
{
public:
    Light(const Vector3f &p, const Vector3f &i) : position(p), intensity(i) {}
    virtual ~Light() = default;
    Vector3f position;
    Vector3f intensity;
};


#endif //CUDA_PATH_TRACING_LIGHT_H
