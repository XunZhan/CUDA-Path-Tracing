//
// Created by zhanx on 11/10/2020.
//
#include <cuda.h>
#include <curand_kernel.h>
#include <cstdio>
#include <stdlib.h>
#include <assert.h>
#include <cmath>
#include <queue>
#include "Scene.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
                  file << ":" << line << " '" << func << "' \n";
        std::cerr << cudaGetErrorString(result) << "\n";
        cudaDeviceReset(); // Make sure we call CUDA Device Reset before exiting
        exit(99);
    }
}

#define CELL_X_NUM 32 // cell number in x axis
#define CELL_Y_NUM 32 // cell number in y axis


// CPU Memory
static dim3 blocks;
static dim3 threads;
static int num_pixles;
static int num_bvhElems;
static int num_triangles;
static Scene* scene_host = NULL;
BVHElem* bvhElem_host = NULL;
Bounds3* bound_host = NULL;
Triangle* triangle_host = NULL;

// GPU Memory
curandState *devStates;
static Ray* ray_dev = NULL;
BVHElem* bvhElem_dev = NULL;
Material* material_dev = NULL;
Triangle* triangle_dev = NULL;
Bounds3* bound_dev = NULL;       // bounding box for objects not triangle

// Unified Memory
float *frameBuffer = NULL;


// cpu internal function definition
void BuildBvhNodeList(Scene* scene);


__device__ Vector3f Material::cudaSample (const Vector3f &wi, const Vector3f &N , curandState *state, int pid) {                       // sample a ray by Material properties
    switch(m_type){
        case DIFFUSE:
        {
            // uniform sample on the hemisphere
            curandState localState = state[pid];
            curand_init((unsigned int) clock64(), pid, 0, &localState);
            float x_1 = curand_uniform(&localState);
            curand_init((unsigned int) clock64(), pid, 0, &localState);
            float x_2 = curand_uniform(&localState);
            float z = fabs(1.0f - 2.0f * x_1);
            float r = sqrt(1.0f - z * z), phi = 2 * M_PI * x_2;
            Vector3f localRay(r*std::cos(phi), r*std::sin(phi), z);
            return toWorld(localRay, N);

            break;
        }
    }
    return Vector3f(0, 0, 0);
}

__device__ void Triangle::CudaSample (Intersection &pos, float &pdf, curandState *state, int pid) {
    curandState localState = state[pid];
    float x = sqrt(curand_uniform(&localState)), y = curand_uniform(&localState);
    pos.emit = this->m->m_emission;
    pos.coords = v0 * (1.0f - x) + v1 * (x * (1.0f - y)) + v2 * (x * y);
    pos.normal = this->normal;
    pdf = 1.0f / area;
}



void InitRender(Scene* scene) {

    // thread limit per block is 1024
    blocks = dim3(CELL_X_NUM, CELL_Y_NUM);
    threads = dim3((scene->width + blocks.x - 1) / blocks.x,
                   (scene->height + blocks.y - 1) / blocks.y);

    num_pixles = scene->width * scene->height;
    num_bvhElems = scene->bvh->nodeCount;

    // convert BVH to BVHElem
    BuildBvhNodeList(scene);

    // Load to CPU Memory
    scene_host = scene;


    checkCudaErrors(cudaMalloc(&ray_dev, num_pixles*sizeof(Ray)));

    if (num_bvhElems > 0) {
        checkCudaErrors(cudaMalloc(&bvhElem_dev, num_bvhElems*sizeof(BVHElem)));
        checkCudaErrors(cudaMemcpy(bvhElem_dev,bvhElem_host,
                                   num_bvhElems*sizeof(BVHElem), cudaMemcpyHostToDevice));
    }

    if (num_triangles > 0) {
        checkCudaErrors(cudaMalloc(&material_dev, num_triangles*sizeof(Material)));
        for (int i = 0; i<num_triangles; i++) {
            checkCudaErrors(cudaMemcpy(&material_dev[i], triangle_host[i].m, sizeof(Material), cudaMemcpyHostToDevice));
        }
    }

    if (num_triangles > 0) {
        checkCudaErrors(cudaMalloc(&triangle_dev, num_triangles*sizeof(Triangle)));
        checkCudaErrors(cudaMemcpy(triangle_dev, triangle_host, num_triangles*sizeof(Triangle), cudaMemcpyHostToDevice));
        // TODO: allcoate material_dev address to triangle_dev, failed
//        for (int i = 0; i<num_triangles; i++) {
//            checkCudaErrors(cudaMemcpy(&triangle_dev[i].m, &material_dev[i], sizeof(Material*), cudaMemcpyDeviceToDevice));
//        }
    }

    if (num_bvhElems > 0) {
        checkCudaErrors(cudaMalloc(&bound_dev, num_bvhElems*sizeof(Bounds3)));
        checkCudaErrors(cudaMemcpy(bound_dev, bound_host,
                                   num_bvhElems*sizeof(Bounds3), cudaMemcpyHostToDevice));
    }

    // Init frame buffer in Unified Memory
    checkCudaErrors(cudaMallocManaged((void **)&frameBuffer, 3*num_pixles*sizeof(float)));
    for (int i = 0; i<3*num_pixles; i++) {
        frameBuffer[i] = 0;
    }

    // init cuda random generator
    int threadNum = blocks.x * blocks.y * threads.x * threads.y;
    checkCudaErrors(cudaMalloc((void **)&devStates, threadNum * sizeof(curandState)));

}

void FreeRender() {
    // free CPU
    free(bvhElem_host);
    free(bound_host);
    free(triangle_host);

    // free GPU
    cudaFree(ray_dev);
    cudaFree(bvhElem_dev);
    cudaFree(material_dev);
    cudaFree(triangle_dev);
    cudaFree(bound_dev);
    cudaFree(frameBuffer);
    cudaFree(devStates);
}


__device__ float cudaRandomFloat(curandState *state, int pid) {

    curandState localState = state[pid];
    curand_init((unsigned int) clock64(), pid, 0, &localState);
    return curand_uniform(&localState);
}

__device__ void sampleLight(Intersection &pos, float &pdf,
                            int triangleNum, int pid,
                            Triangle* triangles, curandState *state) {
    float emit_area_sum = 0;

    // assume we only have one light in the scene
    for (int k = 0; k < triangleNum; k++) {
        if (triangles[k].cudaHasEmit()){
            emit_area_sum += triangles[k].getArea();
        }
    }
    float p = cudaRandomFloat(state, pid) * emit_area_sum;
    emit_area_sum = 0;
    for (int k = 0; k < triangleNum; k++) {
        if (triangles[k].cudaHasEmit()){
            emit_area_sum += triangles[k].getArea();
            if (p <= emit_area_sum){
                triangles[k].CudaSample(pos, pdf, state, pid);
                break;
            }
        }
    }
}

__device__ Intersection SceneIntersect(int pid, Ray ray, int bvhElemNum, int triangleNum,
                                       BVHElem* bvhElems, Triangle* triangles, Bounds3* bounds) {
    Intersection inter;
    inter.coords = Vector3f(-1);
    if (bvhElems == NULL)
        return inter;

    bool visited[32];
    for (int i = 0; i<bvhElemNum; i++) {
        bvhElems[i].visited = false;
        visited[i] = false;
    }


    int arr[3] = {(ray.direction.x <= 0), (ray.direction.y <= 0), (ray.direction.z <= 0)};

    // DFS BVHElem
    int istack[32];
    istack[0] = bvhElems[0].boundIdx;
    int curSize = 1;

    while (curSize > 0) {

        BVHElem &curElem = bvhElems[istack[curSize - 1]];

        bool vl = (curElem.leftIdx < 0 || visited[curElem.leftIdx]);
        bool vr = (curElem.rightIdx < 0 || visited[curElem.rightIdx]);


        if (vl && vr) {
            visited[curElem.boundIdx] = true;
            //curElem.visited = true;
            curSize--;

            if (curElem.isLeaf) { // node is leaf
                for (int a = 0; a < curElem.triNum; a++) { // find intersection with all triangles of this object
                    Intersection ci = triangles[curElem.triStartIdx + a].getIntersection(ray);
                    if (ci.happened && (ci.distance<inter.distance)) {
                        inter = ci;
                    }
                }
            }
        }
        else {
            if (curElem.leftIdx >= 0 && !visited[curElem.leftIdx]) {
//          if (curElem.leftIdx >= 0 && !bvhElems[curElem.leftIdx].visited) {
                BVHElem &left = bvhElems[curElem.leftIdx];
                if (bounds[left.boundIdx].IntersectP(ray,
                                                     Vector3f(1 / (float) ray.direction.x, 1 / (float) ray.direction.y,
                                                              1 / (float) ray.direction.z), arr)) {
                    // if hit left bounding box
                    istack[curSize] = curElem.leftIdx;
                    curSize++;
                } else {
                    visited[left.boundIdx] = true;
                    //left.visited = true;
                }
            }

            if (curElem.rightIdx >= 0 && !visited[curElem.rightIdx]) {
                BVHElem &right = bvhElems[curElem.rightIdx];
                if (bounds[right.boundIdx].IntersectP(ray,
                                                      Vector3f(1 / (float) ray.direction.x, 1 / (float) ray.direction.y,
                                                               1 / (float) ray.direction.z), arr)) {
                    istack[curSize] = curElem.rightIdx;
                    curSize++;
                } else {
                    visited[right.boundIdx] = true;
                    //right.visited = true;
                }
            }

            if (curSize > 32) {
                // TODO: handle CUDA kernel error
                printf("stack overflow %d\n", 32);
            }
        }
    }

    return inter;
}

__device__ Vector3f CalcColor(int pid, int bvhElemNum, int triangleNum,
                              Ray* rays, BVHElem* bvhElems, Triangle* triangles,  Bounds3* bounds, Material* materials,
                              curandState *state) {


    Vector3f backgroundColor = Vector3f(0.235294, 0.67451, 0.843137);

    Ray curRay = rays[pid];
    Vector3f pixelColor = Vector3f(0,0,0);
    int maxDepth = 3;  // maxDepth cannot exceed stackSize/2
    float RussianRoulette = 0.8;

    Vector3f vstack[32];
    for (int i = 0; i<32; i++) {
        vstack[i] = Vector3f(0, 0, 0);
    }


    // 这里用stack实现递归，stack内部：
    // 0，1： depth = 0时， in_dir 的颜色 + dir color 需要的系数
    // 2，3： depth = 1时， in_dir 的颜色 + dir color 需要的系数， 以此类推
    int curDepth = 0;
    for (int d = 0; d <= maxDepth; d++) {

        curDepth = d;
        Intersection intersection = SceneIntersect(pid, curRay, bvhElemNum, triangleNum, bvhElems, triangles, bounds);
        if(!intersection.happened) {
            vstack[d*2+0] = backgroundColor;
            break;
        }

        if (intersection.m != NULL && intersection.m->cudaHasEmission()) {
            vstack[d*2+0] = Vector3f(1.0,1.0,1.0);
            break;
        }

        // contribution from the light source
        Vector3f dir_color = Vector3f(0, 0, 0);
        float pdf_light;
        Intersection lightPoint;
        sampleLight(lightPoint, pdf_light, triangleNum, pid, triangles, state);
        lightPoint.normal.normalized();

        Vector3f w_dir = normalize(lightPoint.coords - intersection.coords);
        Ray shadowRay(intersection.coords, w_dir);
        Intersection shadowRayInter = SceneIntersect(pid, shadowRay, bvhElemNum, triangleNum, bvhElems, triangles, bounds);

        // if light ray not blocked in the middle
        if (!shadowRayInter.happened || shadowRayInter.m->cudaHasEmission())
        {
            if (pdf_light < FLT_EPSILON)
                pdf_light = FLT_EPSILON;

            Vector3f f_r1 = intersection.m->eval(-curRay.direction, w_dir, intersection.normal);
            float kk = dotProduct(intersection.coords - lightPoint.coords, intersection.coords - lightPoint.coords);
            dir_color = lightPoint.emit * f_r1 * dotProduct(w_dir, intersection.normal)
                        * dotProduct(-w_dir, lightPoint.normal) / kk / pdf_light;
        }
        vstack[d*2+0] = dir_color;

        // contribution from other objects
        // Russian Roulette
        bool needBreak = true;
        Vector3f indir_color = Vector3f(0,0,0);
        float testrr = cudaRandomFloat(state, pid);
        Vector3f randomDir;
        if (testrr <= RussianRoulette) {
            randomDir = intersection.m->cudaSample(-curRay.direction, intersection.normal, state, pid);
            randomDir = randomDir.normalized();
            float pdf_object = intersection.m->pdf(-curRay.direction, randomDir,intersection.normal);
            Ray ro(intersection.coords, randomDir);

            Intersection objRayInter = SceneIntersect(pid, ro,  bvhElemNum, triangleNum, bvhElems, triangles, bounds);
            if (objRayInter.happened)
                if (!objRayInter.m->cudaHasEmission()) {
                    if (pdf_object < FLT_EPSILON)
                        pdf_object = FLT_EPSILON;

                    Vector3f f_r2 = intersection.m->eval(-curRay.direction, ro.direction, intersection.normal);
                    indir_color = f_r2 * dotProduct(ro.direction, intersection.normal) / pdf_object / RussianRoulette;
                    curRay = ro;
                    needBreak = false;
                }
        }

        vstack[d * 2 + 1] = indir_color;
        if (needBreak)
            break;
    }

    // 这里反过来推算颜色
    for (int i = curDepth; i>0; i--) {
        // in_dir + dir
        Vector3f prev = vstack[i*2] + vstack[i*2+1];
        vstack[2*i-1] = vstack[2*i-1] * prev;
    }
    pixelColor = vstack[0] + vstack[1];


    return pixelColor;
}

__global__ void SetKernelRand(curandState *state, int h, int w)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    int id = i + j * w;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(100*sizeof(curandState), id, 0, &state[id]);
}

__device__ float deg2rad(const float& deg) { return deg * M_PI / 180.0; }

__global__ void GenerateRay(int width, int height, double fov, float* fb, Ray* rays) {
    Vector3f eye_pos(278, 273, -800);

    float scale = tan(deg2rad(fov * 0.5));
    float imageAspectRatio = width / (float)height;

    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (i < (float)width && j < (float)height)
    {
        int pixelIdx = i + (j * width);
        float x = (2 * (i + 0.5) / (float)width - 1) *
                  imageAspectRatio * scale;
        float y = (1 - 2 * (j + 0.5) / (float)height) * scale;
        Vector3f dir = normalize(Vector3f(-x, y, 1));

        Ray &ray = rays[pixelIdx];
        ray.origin = eye_pos;
        ray.direction = dir;

    }
}

__global__ void CastRay(int width, int height, float* fb, Ray* rays,
                        int bvhElemNum, int triangleNum,
                        BVHElem* bvhElems, Triangle* triangles, Bounds3* bounds, Material* materials,
                        curandState *state) {
    int spp = 8;

    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (i < width && j < height) {
        int pixelIdx = i + (j * width);
        fb[pixelIdx*3+0] = 0;
        fb[pixelIdx*3+1] = 0;
        fb[pixelIdx*3+2] = 0;
        for (int time = 0; time < spp; time++) {
            Vector3f c = CalcColor(pixelIdx, bvhElemNum, triangleNum,
                                   rays, bvhElems, triangles, bounds, materials, state);
            fb[pixelIdx*3+0] += c.x;
            fb[pixelIdx*3+1] += c.y;
            fb[pixelIdx*3+2] += c.z;
        }

        fb[pixelIdx*3+0] /= (float) spp;
        fb[pixelIdx*3+1] /= (float) spp;
        fb[pixelIdx*3+2] /= (float) spp;


    }
}


__global__ void SetTriangleValue(int triangleNum, Triangle* triangles, Material* materials) {
    for (int i = 0; i<triangleNum; i++) {
        triangles[i].m = materials+i;
    }
}

void Render() {

    SetKernelRand<<<blocks, threads>>>(devStates, scene_host->height, scene_host->width);
    SetTriangleValue<<<blocks, threads>>>(num_triangles, triangle_dev, material_dev);

    GenerateRay<<<blocks, threads>>>(scene_host->width, scene_host->height, scene_host->fov, frameBuffer, ray_dev);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    CastRay<<<blocks, threads>>>(scene_host->width, scene_host->height, frameBuffer,
                                 ray_dev,num_bvhElems, num_triangles,
                                 bvhElem_dev, triangle_dev, bound_dev, material_dev,
                                 devStates);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());


    // save color data to ppm file
    FILE* fp = fopen("image.ppm", "wb");
    (void)fprintf(fp, "P6\n%d %d\n255\n", scene_host->width, scene_host->height);
    for (auto i = 0; i < num_pixles; ++i) {
        static unsigned char color[3];
        color[0] = (unsigned char)(255 * std::pow(clamp(0, 1, frameBuffer[i*3+0]), 0.6f));
        color[1] = (unsigned char)(255 * std::pow(clamp(0, 1, frameBuffer[i*3+1]), 0.6f));
        color[2] = (unsigned char)(255 * std::pow(clamp(0, 1, frameBuffer[i*3+2]), 0.6f));
        fwrite(color, 1, 3, fp);
    }
    fclose(fp);

}

void BuildBvhNodeList(Scene* scene) {

    bvhElem_host = (BVHElem*) malloc(num_bvhElems * sizeof(BVHElem));
    bound_host = (Bounds3*) malloc(num_bvhElems * sizeof(Bounds3));
    std::vector<int> leafIdx;


    for (int i = 0; i<scene->objects.size(); i++) {
        MeshTriangle* mt = (MeshTriangle*)(scene->objects[i]);
        num_triangles += mt->triangles.size();
    }

    triangle_host = (Triangle*) malloc(num_triangles * sizeof(Triangle));

    // BFS for BVH Tree
    BVHBuildNode* root = scene->bvh->root;
    std::queue<BVHBuildNode*> nodeQueue;

    if (root != NULL)
        nodeQueue.push(root);
    int triCount = 0;
    while (!nodeQueue.empty())
    {
        BVHBuildNode* nd = nodeQueue.front();

        BVHElem &curElem = bvhElem_host[nd->nodeIdx];
        curElem.boundIdx = nd->nodeIdx;
        curElem.leftIdx = (nd->left) ? (nd->left->nodeIdx) : -1;
        curElem.rightIdx = (nd->right) ? (nd->right->nodeIdx) : -1;
        curElem.isLeaf = (nd->object);
        if (curElem.isLeaf) {
            MeshTriangle* mt = (MeshTriangle*)(nd->object);
            curElem.triStartIdx = triCount;
            curElem.triNum = mt->triangles.size();
            for (int j = 0; j < curElem.triNum; j++) {
                triangle_host[curElem.triStartIdx + j] = mt->triangles[j];
                triangle_host[curElem.triStartIdx + j].m = mt->m;
                triCount++;
            }
            leafIdx.push_back(curElem.boundIdx);
        }
        else {
            curElem.triStartIdx = -1;
            curElem.triNum = 0;
        }

        bound_host[nd->nodeIdx] = nd->bounds;


        nodeQueue.pop();

        if (nd->left)
        {
            nodeQueue.push(nd->left);
        }
        if (nd->right)
        {
            nodeQueue.push(nd->right);
        }
    }
}



