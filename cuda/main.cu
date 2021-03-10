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


    int w = 128;
    int h = 128;
    Block* world = new Block[w * h];

    for (int x = 0; x < w; x++) {
        for (int y = 0; y < h; y++) {

            float r = (x + y) % 3 == 0 ? 1.0 : 0.0;
            float g = (x + y) % 3 == 1 ? 1.0 : 0.0;
            float b = (x + y) % 3 == 2 ? 1.0 : 0.0;

            int r2 = 10000;
            int y2 = r2 - (x - 100) * (x - 100) - (y - 100) * (y - 100);
            
            float py = 10.0;
            if (y2 > 0.0) {
                py = sqrt(y2) + 10.0;
            }
            

            world[x * h + y] = Block(AABB3(Vec3((float)x, py - 10.0, (float)y), Vec3((float)x + 1.0, py, (float)y + 1.0)), Vec3(r, g, b));
            printf("(%f, %f)\n", (float)x, (float)y);
        }
    }

    trace_manager.upload_world(world, w * h);

    Vec3 position = Vec3(5.5, 85.0, -2.0);
    Vec3 f = Vec3(0.0, 0.0, 1.0).normalize();
    Vec3 side = Vec3(1.0, 0.0, 0.0).normalize();
    Vec3 up = Vec3(0.0, 1.0, 0.0).normalize();

    Vec3 rotation_axis = Vec3(0.0, 1.0, 0.0);

    float rotation_x = 0.0;
    float rotation_y = 0.0;

    float t = 1.0;
    int i = 0;
    while (true)
    {
        double mouse_delta_x, mouse_delta_y;
        mouse_delta(&mouse_delta_x, &mouse_delta_y);
        rotation_x += mouse_delta_x / 100.0;
        rotation_y += mouse_delta_y / 100.0;

        Vec3 forward;
        forward = f.rotate(side, rotation_y);
        forward = forward.rotate(up, rotation_x);


        if (key_pressed(KEY::W)) {
            position += Vec3(forward.x, 0.0, forward.z).normalize() * 1.0;
        }

        if (key_pressed(KEY::A)) {
            position += Vec3(-forward.z, 0.0, forward.x).normalize() * 1.0;
        }

        if (key_pressed(KEY::S)) {
            position -= Vec3(forward.x, 0.0, forward.z).normalize() * 1.0;
        }

        if (key_pressed(KEY::D)) {
            position += Vec3(forward.z, 0.0, -forward.x).normalize() * 1.0;
        }

        if (key_pressed(KEY::UP)) {
            rotation_y += 0.016;
        }

        if (key_pressed(KEY::DOWN)) {
            rotation_y -= 0.016;
        }

        // if (key_pressed(KEY::A))
        // Vec3 position = Vec3(cos(t) * 110.0, 3.0, sin(t) * 110.0);
        // Vec3 forward = position - Vec3(0.0, 3.0, 0.0);
        // Vec3 side = Vec3(-cos(t - 0.5 * PI), 0.0, -sin(t - 0.5 * PI));
        // Vec3 up = forward.cross(side);
        // forward = forward.normalize();

        // printf("Forward: %f, %f, %f\n", forward.x, forward.y, forward.z);

        // printf("t: %f\n", t);

        

        Frustrum view = Frustrum(position, forward * t, 1920.0 / 1080.0);

        float* surface = trace_manager.render_frame(view);
        render_frame(surface);
        // printf("frame: %i\n", i);
        i++;
    }

    opengl_exit();
}