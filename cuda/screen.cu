#include "screen.cuh"

// Include standard headers
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// Include GLEW
#include <GL/glew.h>

// Include GLFW
#include <GLFW/glfw3.h>
GLFWwindow* window;

#include <cuda_gl_interop.h>

// Include GLM
#include <glm/glm.hpp>
using namespace glm;


GLuint viewGLTexture;
int width;
int height;

void create_screen(int w, int h)
{
	width = w;
	height = h;
	// Initialise GLFW
	if( !glfwInit() )
	{
		fprintf( stderr, "Failed to initialize GLFW\n" );
		getchar();
		assert(false);
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Open a window and create its OpenGL context
	window = glfwCreateWindow( width, height, "Tutorial 01", NULL, NULL);
	if( window == NULL ){
		fprintf( stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n" );
		getchar();
		glfwTerminate();
		assert(false);
	}
	glfwMakeContextCurrent(window);

	// Initialize GLEW
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		getchar();
		glfwTerminate();
		assert(false);
	}

	// Ensure we can capture the escape key being pressed below
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

	// Dark blue background
	glClearColor(0.5f, 0.0f, 0.2f, 0.0f);

	// Clear the screen. It's not mentioned before Tutorial 02, but it can cause flickering, so it's there nonetheless.
	glClear( GL_COLOR_BUFFER_BIT );

	return;
}

void opengl_exit() {
	glfwTerminate();
}

void render_frame(void* data)
{
	glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &viewGLTexture);

    glBindTexture(GL_TEXTURE_2D, viewGLTexture); 
    {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_FLOAT, data);
    } 
	GLenum err = glGetError();
	if (err != GL_NO_ERROR)
		printf("OPENGL Error: %u:%s\n", err, glewGetErrorString(err));

	GLuint frame_buffer_obj;
	glGenFramebuffers(1, &frame_buffer_obj);
	glBindFramebuffer(GL_READ_FRAMEBUFFER, frame_buffer_obj);
	glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, viewGLTexture, 0);
	err = glGetError();
	if (err != GL_NO_ERROR)
		printf("OPENGL Error: %u:%s\n", err, glewGetErrorString(err));

	// Draw frame to screen.
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
	glBlitFramebuffer(0, 0, width, height, 0, 0, width, height,
		GL_COLOR_BUFFER_BIT, GL_NEAREST);

	err = glGetError();
	if (err != GL_NO_ERROR)
		printf("OPENGL Error: %u:%s\n", err, glewGetErrorString(err));

	glfwSwapBuffers(window);
	glfwPollEvents();
}