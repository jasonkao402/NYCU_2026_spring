#pragma once
#include <GLM/glm.hpp>
#include <string>
#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

using namespace glm;

struct guiDatas {
	float fps = 30.0;
	bool reloadBtn = false;
	float depthFactor = 0.05f;
	int renderMode = 0;
	int textureIndex = 0;
	glm::vec3 lightPos = glm::vec3(0.0f, 10.0f, 10.0f);
	glm::vec3 viewPos = glm::vec3(0.0f, 8.0f, 30.0f);
};

inline void imguiRender(guiDatas& values) {
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
	ImGuiIO& io = ImGui::GetIO();
	values.reloadBtn = false;
	{
		ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.
		
		ImGui::SliderFloat("fps", &values.fps, 3.0f, 120.0f);

		if (ImGui::Button("Recompile shader")) {
			values.reloadBtn = true;
		}
		ImGui::Text("Application average %.1f FPS(%.3f ms/frame)", io.Framerate, 1000.0f / io.Framerate);
		ImGui::SliderFloat("Depth Factor", &values.depthFactor, 0.0f, 0.3f);
		ImGui::SliderInt("Texture", &values.textureIndex, 0, 1);
		ImGui::SliderInt("Render Mode", &values.renderMode, 0, 2);
		ImGui::SliderFloat3("Light Position", &values.lightPos[0], -20.0f, 20.0f);
		ImGui::SliderFloat3("View Position", &values.viewPos[0], -50.0f, 50.0f);
		ImGui::End();
	}

	// Rendering
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

inline void imguiInit(GLFWwindow* window) {
	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

	// Setup Dear ImGui style
	ImGui::StyleColorsDark();
	//ImGui::StyleColorsLight();

	// Setup Platform/Renderer backends
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	const char* glsl_version = "#version 460";

	ImGui_ImplOpenGL3_Init(glsl_version);
	ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
}