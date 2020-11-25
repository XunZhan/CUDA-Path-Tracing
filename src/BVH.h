//
// Created by zhanx on 11/11/2020.
//
#pragma once
#ifndef CUDA_PATH_TRACING_BVH_H
#define CUDA_PATH_TRACING_BVH_H

#include <atomic>
#include <vector>
#include <memory>
#include <ctime>
#include "MeshTriangle.h"

struct BVHBuildNode;
// BVHAccel Forward Declarations
struct BVHPrimitiveInfo;

// BVHAccel Declarations
// inline int leafNodes, totalLeafNodes, totalPrimitives, interiorNodes;
class BVHAccel {

public:
    // BVHAccel Public Types
    enum class SplitMethod { NAIVE, SAH };

    // BVHAccel Public Methods
    BVHAccel(std::vector<MeshTriangle*> p, int maxPrimsInNode = 1, SplitMethod splitMethod = SplitMethod::NAIVE);
    ~BVHAccel();

    BVHBuildNode* root;

    // BVHAccel Private Methods
    BVHBuildNode* recursiveBuild(std::vector<MeshTriangle*>objects);

    // BVHAccel Private Data
    const int maxPrimsInNode;
    const SplitMethod splitMethod;
    int nodeCount;
    std::vector<MeshTriangle*> primitives;

};

struct BVHBuildNode {
    int nodeIdx;
    Bounds3 bounds;
    BVHBuildNode *left;
    BVHBuildNode *right;
    MeshTriangle* object;
    float area;

public:
    int splitAxis=0, firstPrimOffset=0, nPrimitives=0;
    // BVHBuildNode Public Methods
    BVHBuildNode(){
        bounds = Bounds3();
        left = nullptr;
        right = nullptr;
        object = nullptr;
    }
};

struct BVHElem {
    int boundIdx; // actually boundIdx=nodeIdx
    int leftIdx;
    int rightIdx;

    int triStartIdx;
    int triNum;
    bool isLeaf;
    bool visited;
};



#endif //CUDA_PATH_TRACING_BVH_H
