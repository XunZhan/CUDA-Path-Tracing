//
// Created by zhanx on 11/10/2020.
//
#pragma once
#ifndef CUDA_PATH_TRACING_SCENE_H
#define CUDA_PATH_TRACING_SCENE_H

#include <vector>
#include "Vector.h"
#include "Light.h"
#include "AreaLight.h"
#include "Ray.h"
#include "BVH.h"

class Scene
{
public:
    // setting up options
    int width = 1280;
    int height = 960;
    double fov = 40;
    Vector3f backgroundColor = Vector3f(0.235294, 0.67451, 0.843137);
    int maxDepth = 1;
    float RussianRoulette = 0.8;
    BVHAccel *bvh;

    Scene(int w, int h) : width(w), height(h)
    {}

    void Add(MeshTriangle *object) { objects.push_back(object); }
    void buildBVH();


    // creating the scene (adding objects and lights)
    std::vector<MeshTriangle* > objects;
};

class SceneDev {
public:
    // setting up options
    int width = 1280;
    int height = 960;
    double fov = 40;
    Vector3f backgroundColor = Vector3f(0.235294, 0.67451, 0.843137);
    int maxDepth = 1;
    float RussianRoulette = 0.8;
    SceneDev(int w, int h, double f, Vector3f b, int md, float rr) :
    width(w), height(h), fov(f), backgroundColor(b), maxDepth(md), RussianRoulette(rr) {}
};


#endif //CUDA_PATH_TRACING_SCENE_H
