#pragma once
#include <GLM/glm.hpp>
#include <string>
#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

using namespace glm;

struct guiDatas {
	float fps = 30.0;
	float orbitRadius = 2.5f;
	float orbitSpeed = 80.0f;
	float spinSpeed = 240.0f;

	float lightPos[3] = { 3.0f, 5.0f, 4.0f };
	float shininess = 10.0f;
	float specularStrength = 0.5f;
	bool reloadBtn = false;
};

inline void imguiRender(guiDatas& values) {
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
	ImGuiIO& io = ImGui::GetIO();
	values.reloadBtn = false;
	{
		ImGui::Begin("313552011 Hello, world!");                          // Create a window called "Hello, world!" and append into it.
		
		ImGui::SliderFloat("fps", &values.fps, 3.0f, 120.0f);
		ImGui::SliderFloat("orbit radius", &values.orbitRadius, 0.0f, 5.0f);
		ImGui::SliderFloat("orbit speed", &values.orbitSpeed, 0.0f, 360.0f);
		ImGui::SliderFloat("spin speed", &values.spinSpeed, 0.0f, 360.0f);
		ImGui::Separator();
		ImGui::SliderFloat3("Light Position", values.lightPos, -5.0f, 5.0f, "%.2f");
		ImGui::SliderFloat("Shininess", &values.shininess, 0.0f, 128.0f, "%.2f");
		ImGui::SliderFloat("Specular Strength", &values.specularStrength, 0.0f, 1.0f, "%.2f");

		if (ImGui::Button("Reload shader")) {
			values.reloadBtn = true;
		}
		ImGui::Text("Application average %.1f FPS(%.3f ms/frame)", io.Framerate, 1000.0f / io.Framerate);
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