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


struct SquareData {
	std::vector<glm::vec3> vertices;
	std::vector<glm::vec3> normals;
	std::vector<glm::vec2> texCoords;
	std::vector<glm::vec3> tangents;
};

inline SquareData makeSquare() {
	return {
		{
			{-10.0f, 0.0f, 0.0f},
			{ 10.0f, 0.0f, 0.0f},
			{ 10.0f, 20.0f, 0.0f},
			{-10.0f, 0.0f, 0.0f},
			{ 10.0f, 20.0f, 0.0f},
			{-10.0f, 20.0f, 0.0f}
		},
		std::vector<glm::vec3>(6, {0.0f, 0.0f, 1.0f}),
		{
			{0.0f, 0.0f},
			{1.0f, 0.0f},
			{1.0f, 1.0f},
			{0.0f, 0.0f},
			{1.0f, 1.0f},
			{0.0f, 1.0f}
		},
		std::vector<glm::vec3>(6, {1.0f, 0.0f, 0.0f})
	};
}


class Grid
{
public:
	Grid(int n_line, float interval) {
		total_line = 2 * n_line;

		glGenVertexArrays(1, &VAO);
		glBindVertexArray(VAO);

		
		glGenBuffers(1, &VBO);
		// The following commands will talk about our 'vertexbuffer' buffer
		glBindBuffer(GL_ARRAY_BUFFER, VBO);

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
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
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