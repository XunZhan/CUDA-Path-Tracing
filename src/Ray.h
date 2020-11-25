//
// Created by zhanx on 11/11/2020.
//

#ifndef CUDA_PATH_TRACING_RAY_H
#define CUDA_PATH_TRACING_RAY_H

#include "Vector.h"
struct Ray{
    //Destination = origin + t*direction
    Vector3f origin;
    Vector3f direction, direction_inv;
    double t;//transportation time,
    double t_min, t_max;

    __HOSTDEV__ Ray(const Vector3f& ori, const Vector3f& dir, const double _t = 0.0): origin(ori), direction(dir),t(_t) {
        direction_inv = Vector3f(1./direction.x, 1./direction.y, 1./direction.z);
        t_min = 0.0;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))  // execute on GPU
        t_max = DBL_MAX;
#else
        t_max = std::numeric_limits<double>::max();
#endif

    }

    Vector3f operator()(double t) const{return origin+direction*t;}

//    friend std::ostream &operator<<(std::ostream& os, const Ray& r){
//        os<<"[origin:="<<r.origin<<", direction="<<r.direction<<", time="<< r.t<<"]\n";
//        return os;
//    }
};

#endif //CUDA_PATH_TRACING_RAY_H
