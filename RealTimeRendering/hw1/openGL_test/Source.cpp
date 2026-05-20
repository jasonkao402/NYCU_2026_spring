#include <iostream>
#include <vector>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <GLM/glm.hpp>

#include "asset.h"
#include "camera.h"
#include "utils.h"
#include "model.h"
#include "shader.h"
#include "gl_function.h"
#include "imgui.h"
#include "Mouse.h"
#include "CameraActionImp.h"

#ifdef __cplusplus
extern "C" {
#endif
	// enable nv, amd gpu
	__declspec(dllexport) uint32_t NvOptimusEnablement = 1;
	__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;

#ifdef __cplusplus
}
#endif

int main() {
	std::vector<glm::vec3> controlPoints = {
		{- 5.f,  10.f,   0.f},
		{  0.f,   5.f,   0.f},
		{  5.f,   5.f,   0.f},
		{ 10.f,  10.f,   0.f},
		{- 5.f,   5.f,   5.f},
		{  0.f, -10.f,   5.f},
		{  5.f,   5.f,   5.f},
		{ 10.f,   0.f,   5.f},
		{- 5.f,   0.f,  10.f},
		{  0.f,   5.f,  10.f},
		{  5.f,   0.f,  10.f},
		{ 10.f, - 5.f,  10.f},
		{- 5.f,   0.f,  15.f},
		{  0.f,   5.f,  15.f},
		{  5.f, - 5.f,  15.f},
		{ 10.f, - 5.f,  15.f}
	};

	srand((unsigned)time(NULL));

	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	const int width = 1024, height = 768;

	GLFWwindow* window = glfwCreateWindow(width, height, "Exercise #2", NULL, NULL);
	if (window == NULL) {
		std::cout << "error";
		glfwTerminate();
		return -1;
	}

	glfwSetWindowPos(window, 700, 100);
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1); // v sync 0-off 1-on

	// openGL
	gladLoadGL();

	// callback
	glfwSetMouseButtonCallback(window, mouse_button_callback);
	glfwSetScrollCallback(window, scroll_callback);

	// GL setting
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glEnable(GL_CULL_FACE);

	// GUI init
	guiDatas guiInput;
	imguiInit(window);

	// Camera and Mouse
	Camera camera(window, width, height, { 4.5, 8, 15.0f }, 
		{ std::make_shared<CameraAction::mouseControlBoth>(),std::make_shared<CameraAction::WASDControlMove>() });

	Mouse& mouse = Mouse::get();

	// ubo
	GLuint mvp_point, mvp_buffer;
	gl::genUniformbuffer(mvp_point, mvp_buffer, sizeof(mat4) * 2);

	glBindBuffer(GL_UNIFORM_BUFFER, mvp_buffer);
	glBufferSubData(GL_UNIFORM_BUFFER, 64, sizeof(mat4), &camera.Projection[0][0]);

	// Asset
	auto& asset = Asset::get();

	// objects
	Grid grid(30, 5);
	CustomModel dragon("Asset/dragon.ply");
	CustomModel lightIndicator("Asset/octahedron.ply");

	// shader program
	Program vertexProgram({ "vertex.vs.glsl", "vertex.fs.glsl"});
	Program surfaceProgram({ "quaternion.vs.glsl", "surface_gooch.fs.glsl" });

	// init or hot reload shader
	auto loadShader = [&]() {
		vertexProgram.enable()
			.bindUniformBlock("MVP", mvp_point)
			.setUniform("colors", vec3(1.0f));

		// TODO: Setup necessary informations for dragon program
		surfaceProgram.enable()
			.bindUniformBlock("MVP", mvp_point)
			.setUniform("currentTime", 0.0f);
	};
	loadShader();

	// time
	double currentTime = 0;
	double lastTime = 0;
	double deltaTime = 0.0;
	glm::vec3 lightPositions[3] = {
		glm::vec3(20.0f, 5.0f, 20.0f),
		glm::vec3(-20.0f, 5.0f, -20.0f),
		glm::vec3(0.0f, 0.0f, 0.0f)
	};
	while (!glfwWindowShouldClose(window)) {
		// fps control
		currentTime = glfwGetTime();
		if ((currentTime - lastTime)* guiInput.fps < 1) 
			continue;
		deltaTime = currentTime - lastTime;
		lastTime = currentTime;

		// reset before poll event
		mouse.reset(); 

		glfwPollEvents();

		// update camera and mouse
		mouse.update(deltaTime);
		glfwGetCursorPos(window, &mouse.cursorPos.x, &mouse.cursorPos.y);
		
		camera.action(deltaTime);
		glBindBuffer(GL_UNIFORM_BUFFER, mvp_buffer);
		glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(mat4), &camera.View[0][0]);

		// default screen buffer
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glClearColor(0.1215f, 0.1215f, 0.1215f, 0.0f);
		glDepthFunc(GL_LESS);
		// draw
		{
			vertexProgram.enable();
			{
				grid.Draw(vertexProgram);
				
				for (int i = 0; i < 3; ++i) {
					glm::mat4 model = glm::translate(glm::mat4(1.0f), lightPositions[i]) * glm::scale(glm::mat4(1.0f), glm::vec3(0.5f));
					lightIndicator.Draw(vertexProgram, model);
				}
			}
			
			surfaceProgram.enable();
			// Pass light position, view position, and object color to the fragment shader
			// surfaceProgram.setUniform("lightPos", vec3(3.0f, 5.0f, 4.0f)); // Example light position
			surfaceProgram.setUniform("viewPos", camera.position); // Camera position as view position
			surfaceProgram.setUniform("currentTime", (float)glfwGetTime()); // Rotate over time
			surfaceProgram.setUniform("orbitRadius", guiInput.orbitRadius);
			surfaceProgram.setUniform("orbitSpeed", guiInput.orbitSpeed);
			surfaceProgram.setUniform("spinSpeed", guiInput.spinSpeed);
			// surfaceProgram.setUniform("lightPos", vec3(guiInput.lightPos[0], guiInput.lightPos[1], guiInput.lightPos[2]));
			// surfaceProgram.setUniform("shininess", guiInput.shininess);
			// surfaceProgram.setUniform("specularStrength", guiInput.specularStrength);
			lightPositions[2] = glm::vec3(guiInput.lightPos[0], guiInput.lightPos[1], guiInput.lightPos[2]); // Dynamic light position from GUI
			for (int i = 0; i < 3; ++i) {
				std::string uniformName = "LightPos[" + std::to_string(i) + "]";
				surfaceProgram.setUniform(uniformName.c_str(), lightPositions[i]);
			}
			// toggle normal visualization / gooch shading
			// surfaceProgram.setUniform("showNormals", guiInput.showNormals);
			surfaceProgram.setUniform("showNormals", guiInput.showNormals);
			dragon.Draw(surfaceProgram);
			
		}

		// render gui
		imguiRender(guiInput); 

		// swap buffer and event
		glfwSwapBuffers(window);

		// hot reload
		if (guiInput.reloadBtn) {
			asset.getFiles(asset.root());
			loadShader();
		}
		
		// close window if ESC pressed
		if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
			glfwSetWindowShouldClose(window, true);
		}
	}

	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;
}