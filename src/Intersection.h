//
// Created by zhanx on 11/11/2020.
//

#ifndef CUDA_PATH_TRACING_INTERSECTION_H
#define CUDA_PATH_TRACING_INTERSECTION_H

#include "Material.h"

class Triangle;

struct Intersection
{
    __HOSTDEV__ Intersection(){
        happened=false;
        coords=Vector3f();
        normal=Vector3f();
        distance = DBL_MAX;
        obj =nullptr;
        m=nullptr;
    }
    bool happened;
    Vector3f coords;
    Vector3f tcoords;
    Vector3f normal;
    Vector3f emit;
    double distance;
    Triangle* obj;
    Material* m;
};

#endif //CUDA_PATH_TRACING_INTERSECTION_H
