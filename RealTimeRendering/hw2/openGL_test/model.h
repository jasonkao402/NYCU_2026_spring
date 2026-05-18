#pragma once
#include <glad/glad.h>
#include <GLM/glm.hpp>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

#include "shader.h"
#include "gl_function.h"
#include "asset.h"

#include "texture.h"

struct MeshData;

std::shared_ptr<MeshData> loadPlyFile(const char* path);

class Mesh : public Asset::UsedBy<Asset::dummyUser>
{
public:
	Mesh(std::string filePath) :filePath(filePath) {
	}

	// async load, return VAO Id if mesh is ready, otherwise return 0
	GLuint getVAO() {
		if (isloaded == false) {
			if (!futureData.valid())
				futureData = std::async(loadPlyFile, filePath.c_str());
			if (futureData.wait_for(std::chrono::milliseconds(5)) == std::future_status::ready) {
				this->isloaded = true;
				uploadMeshData(futureData.get());
			}
		}
		return VAO;
	}
	int vertexCount = 0;
	
private:
	void uploadMeshData(std::shared_ptr<MeshData> data);

	const std::string filePath;
	bool isloaded = false;
	std::future<std::shared_ptr<MeshData>> futureData;

	// TODO #2: add other buffer object to keep openGL buffer
	GLuint VAO = 0;
	GLuint VBO = 0;
	
	int total_indices = 0;
};


class Grid
{
public:
	Grid(int n_line, float interval) {
		total_line = 2 * n_line;

		glGenVertexArrays(1, &VAO);
		glBindVertexArray(VAO);

		
		glGenBuffers(1, &VBO);
		// The following commands will talk about our 'vertex buffer' buffer
		glBindBuffer(GL_ARRAY_BUFFER, VBO);

		float maxLen = float(n_line) / 2.0f * interval;
		float offset = -maxLen;
		int eachLineSize = 3 * 2; // each lines has 2 vertex which has 3 float
		std::unique_ptr<float[]> data = std::make_unique<float[]>(total_line * static_cast<size_t>(eachLineSize)); // x and z axis each n_line 
		for (int i = 0; i < n_line; i++) {
			float* t = data.get() + i * eachLineSize;
			t[0] = offset;
			t[1] = 0.0f;
			t[2] = -maxLen;
			t[3] = offset;
			t[4] = 0.0f;
			t[5] = maxLen;

			t = data.get() + n_line * eachLineSize + i * eachLineSize;
			t[0] = -maxLen;
			t[1] = 0.0f;
			t[2] = offset;
			t[3] = maxLen;
			t[4] = 0.0f;
			t[5] = offset;
			offset += interval;
		}

		glBufferData(GL_ARRAY_BUFFER, total_line * static_cast<unsigned long long>(eachLineSize) * sizeof(float), data.get(), GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

		// unbind buffer
		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
	~Grid() {};
	
	inline void Draw(Program& program) const {
		program.setUniform("M", glm::mat4(1.0));
		glBindVertexArray(this->VAO);
		glDrawArrays(GL_LINES, 0, 2 * this->total_line);

		// unbind buffer
		glBindVertexArray(0);
	}
	
private:
	int total_line;
	GLuint VAO;
	GLuint VBO;
};

inline std::vector<glm::vec3> makeCube() {
	return {
		// Front face (z = +0.5)
		{-0.5f, -0.5f,  0.5f},
		{ 0.5f, -0.5f,  0.5f},
		{ 0.5f,  0.5f,  0.5f},
		{-0.5f, -0.5f,  0.5f},
		{ 0.5f,  0.5f,  0.5f},
		{-0.5f,  0.5f,  0.5f},

		// Back face (z = -0.5)
		{ 0.5f, -0.5f, -0.5f},
		{-0.5f, -0.5f, -0.5f},
		{-0.5f,  0.5f, -0.5f},
		{ 0.5f, -0.5f, -0.5f},
		{-0.5f,  0.5f, -0.5f},
		{ 0.5f,  0.5f, -0.5f},

		// Left face (x = -0.5)
		{-0.5f, -0.5f, -0.5f},
		{-0.5f, -0.5f,  0.5f},
		{-0.5f,  0.5f,  0.5f},
		{-0.5f, -0.5f, -0.5f},
		{-0.5f,  0.5f,  0.5f},
		{-0.5f,  0.5f, -0.5f},

		// Right face (x = +0.5)
		{ 0.5f, -0.5f,  0.5f},
		{ 0.5f, -0.5f, -0.5f},
		{ 0.5f,  0.5f, -0.5f},
		{ 0.5f, -0.5f,  0.5f},
		{ 0.5f,  0.5f, -0.5f},
		{ 0.5f,  0.5f,  0.5f},

		// Top face (y = +0.5)
		{-0.5f,  0.5f,  0.5f},
		{ 0.5f,  0.5f,  0.5f},
		{ 0.5f,  0.5f, -0.5f},
		{-0.5f,  0.5f,  0.5f},
		{ 0.5f,  0.5f, -0.5f},
		{-0.5f,  0.5f, -0.5f},

		// Bottom face (y = -0.5)
		{-0.5f, -0.5f, -0.5f},
		{ 0.5f, -0.5f, -0.5f},
		{ 0.5f, -0.5f,  0.5f},
		{-0.5f, -0.5f, -0.5f},
		{ 0.5f, -0.5f,  0.5f},
		{-0.5f, -0.5f,  0.5f}
	};
}
