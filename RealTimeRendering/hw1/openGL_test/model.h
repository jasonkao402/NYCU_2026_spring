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

static std::string buildMode() {
#ifdef _DEBUG
	return "[! Debug mode]";
#else
	return "[Release mode]";
#endif
}

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
		std::vector<glm::vec3> positions;
		std::vector<unsigned int> indices;
		std::ifstream objFile(objPath);
		if (!objFile.is_open()) {
			std::cerr << "Failed to open file: " << objPath << std::endl;
			return;
		}
		auto start = std::chrono::high_resolution_clock::now();
		std::string line;
		bool hasNormals = false;
		while (std::getline(objFile, line)) {
			std::istringstream iss(line);
			std::string prefix;
			iss >> prefix;
			if (prefix == "v") {
				float x, y, z;
				iss >> x >> y >> z;
				positions.emplace_back(x, y, z);
				// vertices.emplace_back(x);
				// vertices.emplace_back(y);
				// vertices.emplace_back(z);
			} else if (prefix == "f") {
				unsigned int v1, v2, v3;
				// char slash; // to ignore the '/' character
				iss >> v1 >> v2 >> v3;
				indices.emplace_back(v1 - 1); // OBJ indices are 1-based
				indices.emplace_back(v2 - 1);
				indices.emplace_back(v3 - 1);
			}
			else if (prefix == "vn")
			{
				hasNormals = true;
			}
			
		}
		objFile.close();
		// ---------- Compute per-vertex normals ----------
		std::vector<glm::vec3> normals(positions.size(), glm::vec3(0.0f));

		// For each triangle face, compute normal and add to each vertex
		for (size_t i = 0; i < indices.size(); i += 3) {
			unsigned int i1 = indices[i];
			unsigned int i2 = indices[i+1];
			unsigned int i3 = indices[i+2];

			const glm::vec3& v1 = positions[i1];
			const glm::vec3& v2 = positions[i2];
			const glm::vec3& v3 = positions[i3];

			// Compute face normal (cross product of two edges)
			glm::vec3 edge1 = v2 - v1;
			glm::vec3 edge2 = v3 - v1;
			glm::vec3 faceNormal = glm::normalize(glm::cross(edge1, edge2));

			// Accumulate face normal to each vertex of the triangle
			normals[i1] += faceNormal;
			normals[i2] += faceNormal;
			normals[i3] += faceNormal;
		}

		// Normalize accumulated normals
		for (auto& n : normals) {
			if (glm::length(n) > 0.0f)
				n = glm::normalize(n);
			else
				n = glm::vec3(0.0f, 1.0f, 0.0f); // fallback
		}

		// ---------- Build interleaved vertex buffer (position + normal) ----------
		std::vector<float> vertexData;
		for (size_t i = 0; i < positions.size(); ++i) {
			vertexData.push_back(positions[i].x);
			vertexData.push_back(positions[i].y);
			vertexData.push_back(positions[i].z);
			vertexData.push_back(normals[i].x);
			vertexData.push_back(normals[i].y);
			vertexData.push_back(normals[i].z);
		}
		// Generate and bind VAO
		glGenVertexArrays(1, &VAO);
		glBindVertexArray(VAO);

		// Generate and bind VBO
		GLuint VBO;
		glGenBuffers(1, &VBO);
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), vertexData.data(), GL_STATIC_DRAW);

		// Generate and bind EBO
		GLuint EBO;
		glGenBuffers(1, &EBO);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

		// Position attribute (offset 0, stride 6 floats)
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(0);

		// Normal attribute (offset 3 floats, stride 6 floats)
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
		glEnableVertexAttribArray(1);

		// Unbind VAO
		glBindVertexArray(0);

		total_indices = static_cast<int>(indices.size());
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> duration = (end - start);
		std::cout << buildMode() << " Loaded mesh " << " with " << vertexData.size() << " vertices and " << indices.size() << " indices, takes " << duration.count() << " s\n";
	
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