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

class Grid
{
public:
	Grid(int n_line, float interval) {
		total_line = 2 * n_line;

		glGenVertexArrays(1, &VAO);
		glBindVertexArray(VAO);

		GLuint vbo;
		glGenBuffers(1, &vbo);
		// The following commands will talk about our 'vertexbuffer' buffer
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		float maxLen = float(n_line) / 2.0f * interval;
		float offset = -maxLen;
		int eachLineSize = 3 * 2; // each lines has 2 vertex which has 3 float
		std::unique_ptr<float[]> data = std::make_unique<float[]>(total_line * eachLineSize); // x and z axis each n_line 
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
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

		// unbind buffer
		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
	~Grid() {};
	
	inline void Draw(Program& program) {
		program.setUniform("M", glm::mat4(1.0));
		glBindVertexArray(this->VAO);
		glDrawArrays(GL_LINES, 0, 2 * this->total_line);

		// unbind buffer
		glBindVertexArray(0);
	}
	
private:
	int total_line;
	GLuint VAO;
};

#include <fstream>
#include <sstream>

class TeapotModel
{
public:
	TeapotModel(const std::string& objPath) {
		// TODO: load obj file and create buffers
		std::vector<float> vertices;
		std::vector<unsigned int> indices;
		std::ifstream objFile(objPath);
		if (!objFile.is_open()) {
			std::cerr << "Failed to open file: " << objPath << std::endl;
			return;
		}
		std::string line;
		while (std::getline(objFile, line)) {
			std::istringstream iss(line);
			std::string prefix;
			iss >> prefix;
			if (prefix == "v") {
				float x, y, z;
				iss >> x >> y >> z;
				vertices.emplace_back(x);
				vertices.emplace_back(y);
				vertices.emplace_back(z);
			}
			else if (prefix == "f") {
				unsigned int v1, v2, v3;
				char slash; // to ignore the '/' character
				iss >> v1 >> slash >> slash >> v2 >> slash >> slash >> v3;
				indices.emplace_back(v1 - 1); // OBJ indices are 1-based
				indices.emplace_back(v2 - 1);
				indices.emplace_back(v3 - 1);
			}
		}
		objFile.close();
	}
	~TeapotModel() {}

	inline void Draw(Program& program) {
		program.setUniform("M", glm::mat4(1.0));
		glBindVertexArray(this->VAO);
		glDrawElements(GL_TRIANGLES, total_indices, GL_UNSIGNED_INT, 0);

		// unbind buffer
		glBindVertexArray(0);
	}

private:
	int total_indices;
	GLuint VAO;
};