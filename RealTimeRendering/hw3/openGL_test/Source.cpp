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

static std::string buildMode() {
#ifdef _DEBUG
	return "[! Debug mode]";
#else
	return "[Release mode]";
#endif
}

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

	srand((unsigned)time(NULL));

	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	const int width = 1024, height = 768;

	GLFWwindow* window = glfwCreateWindow(width, height, "Exercise #3", NULL, NULL);
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
	Camera camera(window, width, height, { 0, 8, 30.0f }, 
		{ std::make_shared<CameraAction::mouseControlBoth>(),std::make_shared<CameraAction::WASDControlMove>() });

	Mouse& mouse = Mouse::get();

	// ubo
	GLuint mvp_point, mvp_buffer;
	gl::genUniformbuffer(mvp_point, mvp_buffer, sizeof(mat4) * 2);

	glBindBuffer(GL_UNIFORM_BUFFER, mvp_buffer);
	glBufferSubData(GL_UNIFORM_BUFFER, 64, sizeof(mat4), &camera.Projection[0][0]);

	// Asset
	Asset::AssetController asset;

	// TODO #3: Load square
	SquareData quadData = makeSquare();

	// objects
	Grid grid(30, 5);

	// TODO #3: Load cube as light source
	// std::vector<glm::vec3> lightVertices = makeCube();
	LightCubeData lightCubeData;
	lightCubeData.setupGL(makeCube());

	// shader programs
	Program vertexProgram(asset.getShaders({ "vertex.vs.glsl", "vertex.fs.glsl" }));
	// TODO #3: set shader path for your program
	Program surfaceProgram(asset.getShaders({ "surface.vs.glsl", "surface.fs.glsl" }));

	// init or hot reload shader
	auto resetProgram = [&]() {
		vertexProgram.enable()
			.bindUniformBlock("MVP", mvp_point)
			.setUniform("colors", vec3(0.2f));

		// TODO #3: setup necessary "constant" informations for your program
		surfaceProgram.enable()
			.bindUniformBlock("MVP", mvp_point)
			.setUniform("diffuseMap", 0)  // Matches GL_TEXTURE0
            .setUniform("normalMap", 1)   // Matches GL_TEXTURE1
            .setUniform("depthMap", 2)    // Matches GL_TEXTURE2
            .setUniform("depthFactor", 0.1f); // Default POM scale, tweak as needed
	};
	resetProgram();

	// time
	double currentTime = 0;
	double lastTime = 0;
	double deltaTime = 0.0;

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
		
		camera.action(deltaTime, mouse);
		glBindBuffer(GL_UNIFORM_BUFFER, mvp_buffer);
		glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(mat4), &camera.View[0][0]);

		// default screen buffer
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glClearColor(0.8f, 0.8f, 0.8f, 0.0f);
		glDepthFunc(GL_LESS);
		// draw
		{
			vertexProgram.enable();
			{
				grid.Draw(vertexProgram);
				
				glm::mat4 lightModel = glm::mat4(1.0f);
                lightModel = glm::translate(lightModel, guiInput.lightPos);
                lightModel = glm::scale(lightModel, glm::vec3(0.5f));

				vertexProgram.setUniform("M", lightModel);
                vertexProgram.setUniform("colors", glm::vec3(1.0f, 1.0f, 1.0f)); // Make the cube white
                
                lightCubeData.Draw(vertexProgram);
			}

			// TODO #3: prepare light source and render your model
			//        : setup necessary "dynamic" informations for your program
			surfaceProgram.enable();
			surfaceProgram.setUniform("M", glm::mat4(1.0));
			surfaceProgram.setUniform("viewPos", camera.position);
			surfaceProgram.setUniform("renderMode", guiInput.renderMode);
			surfaceProgram.setUniform("depthFactor", guiInput.depthFactor);

            surfaceProgram.setUniform("numPointLights", 1);
            surfaceProgram.setUniform("pointLights[0].position", guiInput.lightPos);
            surfaceProgram.setUniform("pointLights[0].color", glm::vec3(1.0f, 1.0f, 1.0f));
            surfaceProgram.setUniform("pointLights[0].constant", 1.0f);
            surfaceProgram.setUniform("pointLights[0].linear", 0.045f);
            surfaceProgram.setUniform("pointLights[0].quadratic", 0.0075f);

			GLuint idDiffuse = asset.getTexture(guiInput.textureIndex == 0 ? "wood.png" : "bricks.png")->getId();
			GLuint idNormal = asset.getTexture(guiInput.textureIndex == 0 ? "wood_normal.png" : "bricks_normal.png")->getId();
			GLuint idDepth = asset.getTexture(guiInput.textureIndex == 0 ? "wood_disp.png" : "bricks_disp.png")->getId();

			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, idDiffuse);

			glActiveTexture(GL_TEXTURE1);
			glBindTexture(GL_TEXTURE_2D, idNormal);

			glActiveTexture(GL_TEXTURE2);
			glBindTexture(GL_TEXTURE_2D, idDepth);

			 glBindVertexArray(quadData.VAO); 
             glDrawArrays(GL_TRIANGLES, 0, quadData.vertexCount); // or glDrawElements if using an EBO
			//quadData->Draw(surfaceProgram);
            
			glBindVertexArray(0);
		}

		// render gui
		// TODO #3: control variables
		guiInput.viewPos = camera.position;
		imguiRender(guiInput);
		camera.position = guiInput.viewPos;

		// swap buffer and event
		glfwSwapBuffers(window);

		// hot reload (recompile shaders)
		if (guiInput.reloadBtn) {
			asset.getFiles(asset.root());
			resetProgram();
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