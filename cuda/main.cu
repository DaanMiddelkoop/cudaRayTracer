#include "screen.cuh"
#include "trace_manager.cuh"

#include <stdio.h>
#include <assert.h>
#include <math.h>

#define PI 3.1415


int main() {
    int width = 1920;
    int height = 1080;

    create_screen(width, height);
    TraceManager trace_manager = TraceManager(width, height);


    int w = 20;
    int h = 20;
    Block world[w * h];

    for (int x = 0; x < w; x++) {
        for (int y = 0; y < h; y++) {
            world[x * h + y] = Block(AABB3(Vec3((float)x, -1.0, (float)y), Vec3((float)x + 1.0, 0.0, (float)y + 1.0)), Vec3(1.0 * (float)x, 1.0, 0.0));
            printf("(%f, %f)\n", (float)x * 0.5, (float)y * 0.5);
        }
    }

    trace_manager.upload_world(world, w * h);

    float t = 0.0;
    int i = 0;
    while (true)
    {
        t += PI / 120.0;
        // Vec3 position = Vec3(cos(t) * 110.0, 3.0, sin(t) * 110.0);
        // Vec3 forward = position - Vec3(0.0, 3.0, 0.0);
        // Vec3 side = Vec3(-cos(t - 0.5 * PI), 0.0, -sin(t - 0.5 * PI));
        // Vec3 up = forward.cross(side);
        // forward = forward.normalize();

        // printf("Forward: %f, %f, %f\n", forward.x, forward.y, forward.z);

        printf("t: %f\n", t);

        Vec3 position = Vec3(5.0, 5.0, -t);
        Vec3 forward = Vec3(0.0, 0.0, 1.0);
        Vec3 up = Vec3(0.0, 1.0, 0.0);
        Vec3 side = Vec3(1.0, 0.0, 0.0);

        Frustrum view = Frustrum(
            position, // position
            forward.normalize(), // forward
            up.normalize() * 1.08, // up
            side.normalize() * 1.92// side
        );

        float* surface = trace_manager.render_frame(view);
        render_frame(surface);
        printf("frame: %i\n", i);
        i++;
    }

    opengl_exit();
}