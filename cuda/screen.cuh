void create_screen(int width, int height);

void render_frame(void* data);

void opengl_exit();

enum KEY {
    W = 0,
    A = 1,
    S = 2,
    D = 3,
    UP = 4,
    DOWN = 5
};

bool key_pressed(KEY key);

void mouse_delta(double* mouse_x, double* mouse_y);