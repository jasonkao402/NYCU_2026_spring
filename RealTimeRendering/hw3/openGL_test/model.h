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
	// Added OpenGL handles and properties
    GLuint VAO = 0;
    GLuint VBO = 0;
    int vertexCount = 0;

    // Generates buffers and uploads the vector data to the GPU
    void setupGL() {
        vertexCount = static_cast<int>(vertices.size());

        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);

        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);

        // 1. Calculate the total size required for all attributes
        size_t posSize = vertices.size() * sizeof(glm::vec3);
        size_t normSize = normals.size() * sizeof(glm::vec3);
        size_t tcSize = texCoords.size() * sizeof(glm::vec2);
        size_t tanSize = tangents.size() * sizeof(glm::vec3);
        size_t totalSize = posSize + normSize + tcSize + tanSize;

        // 2. Allocate an empty buffer of the total size
        glBufferData(GL_ARRAY_BUFFER, totalSize, nullptr, GL_STATIC_DRAW);

        // 3. Upload each vector into its respective chunk of the buffer
        GLintptr offset = 0;
        glBufferSubData(GL_ARRAY_BUFFER, offset, posSize, vertices.data());
        offset += posSize;
        
        glBufferSubData(GL_ARRAY_BUFFER, offset, normSize, normals.data());
        offset += normSize;
        
        glBufferSubData(GL_ARRAY_BUFFER, offset, tcSize, texCoords.data());
        offset += tcSize;
        
        glBufferSubData(GL_ARRAY_BUFFER, offset, tanSize, tangents.data());

        // 4. Tell OpenGL how to read the buffer chunks (Location 0 to 3)
        offset = 0;
        
        // Location 0: Positions (aPos)
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)offset);
        offset += posSize;

        // Location 1: Normals (aNormal)
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)offset);
        offset += normSize;

        // Location 2: Texture Coords (aTexCoords)
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void*)offset);
        offset += tcSize;

        // Location 3: Tangents (aTangent)
        glEnableVertexAttribArray(3);
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)offset);

        // Unbind to prevent accidental modifications
        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

	inline void Draw(Program& program) {
		program.setUniform("M", glm::mat4(1.0));
		glBindVertexArray(this->VAO);
		glDrawArrays(GL_TRIANGLES, 0, this->vertexCount);
		glBindVertexArray(0);
	}
};

inline SquareData makeSquare() {
    SquareData sq;
    sq.vertices = {
        {-10.0f, 0.0f, 0.0f},
        { 10.0f, 0.0f, 0.0f},
        { 10.0f, 20.0f, 0.0f},
        {-10.0f, 0.0f, 0.0f},
        { 10.0f, 20.0f, 0.0f},
        {-10.0f, 20.0f, 0.0f}
    };
    sq.normals = std::vector<glm::vec3>(6, {0.0f, 0.0f, 1.0f});
    sq.texCoords = {
        {0.0f, 0.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f},
        {0.0f, 0.0f},
        {1.0f, 1.0f},
        {0.0f, 1.0f}
    };
    sq.tangents = std::vector<glm::vec3>(6, {1.0f, 0.0f, 0.0f});
    
    // Upload the data to OpenGL before returning
    sq.setupGL(); 
    return sq;
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
struct LightCubeData {
    GLuint VAO = 0;
    GLuint VBO = 0;
    int vertexCount = 0;

    void setupGL(const std::vector<glm::vec3>& vertices) {
        vertexCount = static_cast<int>(vertices.size());
        
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);

        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), vertices.data(), GL_STATIC_DRAW);

        // Location 0: Positions
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);

        glBindVertexArray(0);
    }

	inline void Draw(Program& program) {
		program.enable();
		glBindVertexArray(this->VAO);
		glDrawArrays(GL_TRIANGLES, 0, this->vertexCount);
		glBindVertexArray(0);
	}
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