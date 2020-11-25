//
// Created by zhanx on 11/11/2020.
//

#ifndef CUDA_PATH_TRACING_BOUNDS3_H
#define CUDA_PATH_TRACING_BOUNDS3_H

#include "Ray.h"
#include "Vector.h"
#include <math.h>
#include <limits>
#include <array>

class Bounds3
{
public:
    Vector3f pMin, pMax; // two points to specify the bounding box
    Bounds3()
    {
        double minNum = std::numeric_limits<double>::lowest();
        double maxNum = std::numeric_limits<double>::max();
        pMax = Vector3f(minNum, minNum, minNum);
        pMin = Vector3f(maxNum, maxNum, maxNum);
    }
    Bounds3(const Vector3f p) : pMin(p), pMax(p) {}
    Bounds3(const Vector3f p1, const Vector3f p2)
    {
        pMin = Vector3f(fmin(p1.x, p2.x), fmin(p1.y, p2.y), fmin(p1.z, p2.z));
        pMax = Vector3f(fmax(p1.x, p2.x), fmax(p1.y, p2.y), fmax(p1.z, p2.z));
    }

    Vector3f Diagonal() const { return pMax - pMin; }
    int maxExtent() const
    {
        Vector3f d = Diagonal();
        if (d.x > d.y && d.x > d.z)
            return 0;
        else if (d.y > d.z)
            return 1;
        else
            return 2;
    }

    double SurfaceArea() const
    {
        Vector3f d = Diagonal();
        return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
    }

    Vector3f Centroid() { return 0.5 * pMin + 0.5 * pMax; }
    Bounds3 Intersect(const Bounds3& b)
    {
        return Bounds3(Vector3f(fmax(pMin.x, b.pMin.x), fmax(pMin.y, b.pMin.y),
                                fmax(pMin.z, b.pMin.z)),
                       Vector3f(fmin(pMax.x, b.pMax.x), fmin(pMax.y, b.pMax.y),
                                fmin(pMax.z, b.pMax.z)));
    }

    Vector3f Offset(const Vector3f& p) const
    {
        Vector3f o = p - pMin;
        if (pMax.x > pMin.x)
            o.x /= pMax.x - pMin.x;
        if (pMax.y > pMin.y)
            o.y /= pMax.y - pMin.y;
        if (pMax.z > pMin.z)
            o.z /= pMax.z - pMin.z;
        return o;
    }

    bool Overlaps(const Bounds3& b1, const Bounds3& b2)
    {
        bool x = (b1.pMax.x >= b2.pMin.x) && (b1.pMin.x <= b2.pMax.x);
        bool y = (b1.pMax.y >= b2.pMin.y) && (b1.pMin.y <= b2.pMax.y);
        bool z = (b1.pMax.z >= b2.pMin.z) && (b1.pMin.z <= b2.pMax.z);
        return (x && y && z);
    }

    bool Inside(const Vector3f& p, const Bounds3& b)
    {
        return (p.x >= b.pMin.x && p.x <= b.pMax.x && p.y >= b.pMin.y &&
                p.y <= b.pMax.y && p.z >= b.pMin.z && p.z <= b.pMax.z);
    }
    inline const Vector3f& operator[](int i) const
    {
        return (i == 0) ? pMin : pMax;
    }

    __HOSTDEV__ inline bool IntersectP(const Ray& ray, const Vector3f& invDir,
                           const int* dirisNeg) const;
};



inline bool Bounds3::IntersectP(const Ray& ray, const Vector3f& invDir,
                                const int* dirIsNeg) const
{
    // invDir: ray direction(x,y,z), invDir=(1.0/x,1.0/y,1.0/z), use this because Multiply is faster that Division
    // dirIsNeg: ray direction(x,y,z), dirIsNeg=[int(x>0),int(y>0),int(z>0)], use this to simplify your logic
    // TODO test if ray bound intersects
    float tmp;
    float x_tmin = (pMin.x - ray.origin.x) * invDir.x;
    float x_tmax = (pMax.x - ray.origin.x) * invDir.x;
    float y_tmin = (pMin.y - ray.origin.y) * invDir.y;
    float y_tmax = (pMax.y - ray.origin.y) * invDir.y;
    float z_tmin = (pMin.z - ray.origin.z) * invDir.z;
    float z_tmax = (pMax.z - ray.origin.z) * invDir.z;

    if (dirIsNeg[0])
    {
        tmp = x_tmax;
        x_tmax = x_tmin;
        x_tmin = tmp;
    }

    if (dirIsNeg[1])
    {
        tmp = y_tmax;
        y_tmax = y_tmin;
        y_tmin = tmp;
    }

    if (dirIsNeg[2])
    {
        tmp = z_tmax;
        z_tmax = z_tmin;
        z_tmin = tmp;
    }

    float t_min, t_max;
    t_min = fmax(fmax(x_tmin, y_tmin), z_tmin);
    t_max = fmin(fmin(x_tmax, y_tmax), z_tmax);
//#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))  // execute on GPU
//    t_min = fmaxf(fmaxf(x_tmin, y_tmin), z_tmin);
//    t_max = fminf(fminf(x_tmax, y_tmax), z_tmax);
//#else
//    t_min = std::max(std::max(x_tmin, y_tmin), z_tmin);
//    t_max = std::min(std::min(x_tmax, y_tmax), z_tmax);
//#endif

    if (t_max < 0 || t_min > t_max)
        return false;

    return true;
}

inline Bounds3 Union(const Bounds3& b1, const Bounds3& b2)
{
    Bounds3 ret;
    ret.pMin = Vector3f::Min(b1.pMin, b2.pMin);
    ret.pMax = Vector3f::Max(b1.pMax, b2.pMax);
    return ret;
}

inline Bounds3 Union(const Bounds3& b, const Vector3f& p)
{
    Bounds3 ret;
    ret.pMin = Vector3f::Min(b.pMin, p);
    ret.pMax = Vector3f::Max(b.pMax, p);
    return ret;
}

#endif //CUDA_PATH_TRACING_BOUNDS3_H
