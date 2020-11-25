//
// Created by zhanx on 11/9/2020.
//

#include <chrono>
#include "Renderer.h"

int main(int argc, char** argv)
{
//    cuda_main();
    // Change the definition here to change resolution
    Scene scene(512, 512); //784

    Material* red = new Material(DIFFUSE, Vector3f(0.0f));
    red->Kd = Vector3f(0.63f, 0.065f, 0.05f);
    Material* green = new Material(DIFFUSE, Vector3f(0.0f));
    green->Kd = Vector3f(0.14f, 0.45f, 0.091f);

//    Material* yellow = new Material(DIFFUSE, Vector3f(0.0f));
//    yellow->Kd = Vector3f(0.898, 0.850, 0.321);
//    Material* blue = new Material(DIFFUSE, Vector3f(0.0f));
//    blue->Kd = Vector3f(0.321, 0.698, 0.898);
//    Material* pink = new Material(DIFFUSE, Vector3f(0.0f));
//    pink->Kd = Vector3f(0.898, 0.321, 0.850);

    Material* white = new Material(DIFFUSE, Vector3f(0.0f));
    white->Kd = Vector3f(0.725f, 0.71f, 0.68f);
    Material* light = new Material(DIFFUSE, (8.0f * Vector3f(0.747f+0.058f, 0.747f+0.258f, 0.747f) + 15.6f * Vector3f(0.740f+0.287f,0.740f+0.160f,0.740f) + 18.4f *Vector3f(0.737f+0.642f,0.737f+0.159f,0.737f)));
    light->Kd = Vector3f(0.65f);

    MeshTriangle floor("models/cornellbox/floor.obj", white);
    MeshTriangle shortbox("models/cornellbox/shortbox.obj", white);
    MeshTriangle tallbox("models/cornellbox/tallbox.obj", white);
    MeshTriangle left("models/cornellbox/left.obj", red);
    MeshTriangle right("models/cornellbox/right.obj", green);
    MeshTriangle light_("models/cornellbox/light.obj", light);

    scene.Add(&floor);
    scene.Add(&shortbox);
    scene.Add(&tallbox);
    scene.Add(&left);
    scene.Add(&right);
    scene.Add(&light_);

    scene.buildBVH();

    InitRender(&scene);

    auto start = std::chrono::system_clock::now();
    Render();
    auto stop = std::chrono::system_clock::now();

    FreeRender();

    std::cout << "Render complete: \n";
    std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::hours>(stop - start).count() << " hours\n";
    std::cout << "          : " << std::chrono::duration_cast<std::chrono::minutes>(stop - start).count() << " minutes\n";
    std::cout << "          : " << std::chrono::duration_cast<std::chrono::seconds>(stop - start).count() << " seconds\n";

    return 0;
}
