//
// Created by zhanx on 11/10/2020.
//

#ifndef CUDA_PATH_TRACING_MATERIAL_H
#define CUDA_PATH_TRACING_MATERIAL_H

#include "Global.h"
#include "Vector.h"
#include "float.h"
#include <cuda.h>
#include <curand_kernel.h>

enum MaterialType { DIFFUSE};

class Material {
public:
    MaterialType m_type;
    //Vector3f m_color;
    Vector3f m_emission;
    float ior;
    Vector3f Kd, Ks;
    float specularExponent;
    //Texture tex;

    // from line23 to line31 previously have inline
    // TODO: to use inline, move cpp content to header file
    __HOSTDEV__ Material(MaterialType t=DIFFUSE, Vector3f e=Vector3f(0,0,0)) {
        m_type = t;
        //m_color = c;
        m_emission = e;
    }

    bool hasEmission() {
        if (m_emission.norm() > EPSILON) return true;
        else return false;
    }

    __device__ bool cudaHasEmission() {
        if (m_emission.norm() > FLT_EPSILON)
            return true;
        return false;
    }

    Vector3f sample(const Vector3f &wi, const Vector3f &N) {                       // sample a ray by Material properties
        switch(m_type){
            case DIFFUSE:
            {
                // uniform sample on the hemisphere
                float x_1 = get_random_float(), x_2 = get_random_float();
                float z = std::fabs(1.0f - 2.0f * x_1);
                float r = std::sqrt(1.0f - z * z), phi = 2 * M_PI * x_2;
                Vector3f localRay(r*std::cos(phi), r*std::sin(phi), z);
                return toWorld(localRay, N);

                break;
            }
        }
        return Vector3f(0, 0, 0);
    }

    __device__ Vector3f cudaSample (const Vector3f &wi, const Vector3f &N , curandState *state, int pid);

    __HOSTDEV__ float pdf(const Vector3f &wi, const Vector3f &wo, const Vector3f &N){
        switch(m_type){
            case DIFFUSE:
            {
                // uniform sample probability 1 / (2 * PI)
                if (dotProduct(wo, N) > 0.0f)
                    return 0.5f / M_PI;
                else
                    return 0.0f;
                break;
            }
        }
        return 0.0f;
    }


    __HOSTDEV__ Vector3f eval(const Vector3f &wi, const Vector3f &wo, const Vector3f &N){
        switch(m_type){
            case DIFFUSE:
            {
                // calculate the contribution of diffuse   model
                float cosalpha = dotProduct(N, wo);
                if (cosalpha > 0.0f) {
                    Vector3f diffuse = Kd / M_PI;
                    return diffuse;
                }
                else
                    return Vector3f(0.0f);
                break;
            }
        }
        return Vector3f(0.0f);
    }

private:
    __HOSTDEV__  Vector3f toWorld(const Vector3f &a, const Vector3f &N){
        Vector3f B, C;
        if (fabs(N.x) > fabs(N.y)){
            float invLen = 1.0f / sqrt(N.x * N.x + N.z * N.z);
            C = Vector3f(N.z * invLen, 0.0f, -N.x *invLen);
        }
        else {
            float invLen = 1.0f / sqrt(N.y * N.y + N.z * N.z);
            C = Vector3f(0.0f, N.z * invLen, -N.y *invLen);
        }
        B = crossProduct(C, N);
        return a.x * B + a.y * C + a.z * N;
    }
    //TODO: 2 functions for Fresnel equation
};


#endif //CUDA_PATH_TRACING_MATERIAL_H
