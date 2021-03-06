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

    Block world[1] = {
        Block(AABB3(Vec3(-1.0, -1.0, -1.0), Vec3(1.0, 1.0, 1.0)), Vec3(0.05, 0.05, 0.05))
    };

    trace_manager.upload_world(world, 1);

    float t = 0.0;
    int i = 0;
    while (true)
    {
        t += PI / 120.0;
        Vec3 position = Vec3(cos(t) * 10.0, 3.0, sin(t) * 10.0);
        Vec3 forward = position.normalize() * -1.0;
        Vec3 side = Vec3(-cos(t - 0.5 * PI), 0.0, -sin(t - 0.5 * PI));
        Vec3 up = Vec3(0.0, 1.0, 0.0);

        Frustrum view = Frustrum(
            position, // position
            forward, // forward
            up * 1.08, // up
            side * 1.92 // side
        );

        float* surface = trace_manager.render_frame(view);
        render_frame(surface);
        printf("frame: %i\n", i);
        i++;
    }

    opengl_exit();
}