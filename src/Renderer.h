//
// Created by zhanx on 11/10/2020.
//

#ifndef CUDA_PATH_TRACING_RENDERER_H
#define CUDA_PATH_TRACING_RENDERER_H

#include "Scene.h"

#pragma once

void InitRender(Scene* scene);
void FreeRender();
void Render();


#endif //CUDA_PATH_TRACING_RENDERER_H
