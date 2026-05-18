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
#include <fstream>
#include <sstream>

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

struct MeshData {
	// TODO #2: add other member to store data read from ply file (e.g. normal, faces ...)
	MeshData() {}
	std::vector<glm::vec3> vertices;
	
	// std::vector<glm::vec3> positions;
	std::vector<unsigned int> indices;
	std::vector<glm::vec3> normals;
	std::vector<float> vertexData;
	int vertexCount = 0;
};

/* 
* This function is called the first mesh.getVAO() is called.
* Async loading of ply files. meshData will be passed to uploadMeshData
*/
std::shared_ptr<MeshData> loadPlyFile(const char* path) {
	if (fs::exists(path) == false) {
		std::cout << "Mesh: " << path << " not found!\n";
		return nullptr;
	}
	std::shared_ptr<MeshData> data = std::make_shared<MeshData>();

	// Start timer
	auto start = std::chrono::high_resolution_clock::now();
	std::cout << "Loading mesh: " << path << "...\n";
	
	// TODO #2: read ply file to MeshData
	// Template example: copy cube vertex
	// data->vertices = makeCube();
	
	std::ifstream plyFile(path);
	if (!plyFile.is_open()) {
		std::cerr << "Failed to open file: " << path << std::endl;
		return nullptr;
	}
	std::string line;
	bool headerEnded = false;
	while (std::getline(plyFile, line)) {
		if (!headerEnded) {
			if (line == "end_header") {
				headerEnded = true;
			}
			continue; // skip header lines
		}
		std::istringstream iss(line);
		std::string prefix;
		iss >> prefix;
		if (prefix == "3") { // face line (assuming triangular faces)
			unsigned int v1, v2, v3;
			iss >> v1 >> v2 >> v3;
			data->indices.emplace_back(v1);
			data->indices.emplace_back(v2);
			data->indices.emplace_back(v3);
		} else { // vertex line
			float x, y, z;
			iss >> x >> y >> z;
			data->vertices.emplace_back(x, y, z);
		}
	}
	plyFile.close();

	// For each triangle face, compute normal and add to each vertex
	size_t indexCount = data->indices.size();
	data->vertexCount = data->vertices.size();
	data->normals.resize(data->vertexCount, glm::vec3(0.0f));
	for (size_t i = 0; i < indexCount; i += 3) {
		unsigned int i1 = data->indices[i];
		unsigned int i2 = data->indices[i+1];
		unsigned int i3 = data->indices[i+2];

		const glm::vec3& v1 = data->vertices[i1];
		const glm::vec3& v2 = data->vertices[i2];
		const glm::vec3& v3 = data->vertices[i3];

		// Compute face normal (cross product of two edges)
		glm::vec3 edge1 = v2 - v1;
		glm::vec3 edge2 = v3 - v1;
		glm::vec3 faceNormal = glm::normalize(glm::cross(edge1, edge2));

		// Accumulate face normal to each vertex of the triangle
		data->normals[i1] += faceNormal;
		data->normals[i2] += faceNormal;
		data->normals[i3] += faceNormal;
	}

	// Normalize accumulated normals
	for (auto& n : data->normals) {
		if (glm::length(n) > 0.0f)
			n = glm::normalize(n);
		else
			n = glm::vec3(0.0f, 1.0f, 0.0f); // fallback
	}

	for (size_t i = 0; i < data->vertexCount; ++i) {
		data->vertexData.emplace_back(data->vertices[i].x);
		data->vertexData.emplace_back(data->vertices[i].y);
		data->vertexData.emplace_back(data->vertices[i].z);
		data->vertexData.emplace_back(data->normals[i].x);
		data->vertexData.emplace_back(data->normals[i].y);
		data->vertexData.emplace_back(data->normals[i].z);
	}
	// Completion time
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration = (end - start);
	std::cout << buildMode() << " Loading mesh " << path << " with " << data->vertices.size() << " vertices takes " << duration.count() << " s\n";
	return data;
}

/*
* The function is called in mesh.getVAO(), after loading the ply data.
* Upload the mesh data to OpenGL, and keep VAO on mesh.VAO
*/
void Mesh::uploadMeshData(std::shared_ptr<MeshData> data) {
	if (data == nullptr) {
		return;
	}

	glGenVertexArrays(1, &this->VAO);
	glBindVertexArray(this->VAO);
	data->vertexCount = static_cast<int>(data->vertices.size());
	// TODO #2: upload data to GPU
	//		  : you can add other buffer object to Mesh member

	glGenBuffers(1, &this->VBO);
	glBindBuffer(GL_ARRAY_BUFFER, this->VBO);

	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * data->vertexData.size(), data->vertexData.data(), GL_STATIC_DRAW);

	
	// glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);

	// unbind buffer
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

int main() {

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

	// TODO #2: Load mesh
	std::shared_ptr<Mesh> model = asset.getMesh("dragon.ply");

	// objects
	Grid grid(30, 5);

	// shader programs
	Program vertexProgram(asset.getShaders({ "vertex.vs.glsl", "vertex.fs.glsl" }));
	// TODO #2: set shader path for your program
	Program surfaceProgram(asset.getShaders({ "surface.vs.glsl", "surface.fs.glsl" }));

	// init or hot reload shader
	auto resetProgram = [&]() {
		vertexProgram.enable()
			.bindUniformBlock("MVP", mvp_point)
			.setUniform("colors", vec3(0.2f));

		// TODO #2: setup necessary "constant" informations for your program
		surfaceProgram.enable()
			.bindUniformBlock("MVP", mvp_point)
			.setUniform("colors", vec3(0.3f, 0.6f, 0.9f));
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
			}

			// TODO #2: prepare light source and render your model
			//        : setup necessary "dynamic" informations for your program
			surfaceProgram.enable();
			// 1. Matrix and View setup
			surfaceProgram.setUniform("M", glm::mat4(1.0f));
			surfaceProgram.setUniform("ViewPos", camera.position); // Pass camera position
			
			// 2. Material setup
			surfaceProgram.setUniform("SurfaceColor", glm::vec3(0.7f, 0.7f, 0.7f));
			surfaceProgram.setUniform("DiffuseWarm", 0.5f);
			surfaceProgram.setUniform("DiffuseCool", 0.5f);
			surfaceProgram.setUniform("LightColor", glm::vec3(1.0f, 1.0f, 1.0f)); // Pass light color
			glm::vec3 lightPositions[3] = {
				glm::vec3(10.0f, 10.0f, 10.0f),
				glm::vec3(-10.0f, 10.0f, 10.0f),
				glm::vec3(0.0f, 10.0f, -10.0f),
			};
			// surfaceProgram.setUniformArray("LightPositions", lightPositions, 3); // Pass light positions
			// GLuint lightPosLocation = glGetUniformLocation(surfaceProgram.ID, "LightPositions");
			// glUniform3fv(lightPosLocation, 3, &lightPositions[0][0]);
			for (int i = 0; i < 3; ++i) {
				std::string uniformName = "LightPositions[" + std::to_string(i) + "]";
				surfaceProgram.setUniform(uniformName.c_str(), lightPositions[i]);
			}
			if (model) {
				auto VAO = model->getVAO(); // async load
				if (VAO > 0) {
					glBindVertexArray(VAO);
					// glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(model->vertexCount)); // render loaded model
					glDrawArrays(GL_TRIANGLES, 0, 36); // render template cube
					glBindVertexArray(0);
				}
			}
		}

		// render gui
		// TODO #2: control variables
		imguiRender(guiInput); 

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