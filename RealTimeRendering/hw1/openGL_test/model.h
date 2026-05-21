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

class CustomModel
{
public:
    CustomModel(const std::string& filePath) {
        // Determine file type by extension
        std::string ext;
        size_t dot = filePath.find_last_of('.');
        if (dot != std::string::npos)
            ext = filePath.substr(dot);

        if (ext == ".ply")
            loadPLY(filePath);
        else
            loadOBJ(filePath);   // original OBJ loader
    }

    ~CustomModel() {}

    inline void Draw(Program& program, glm::mat4 M = glm::mat4(1.0)) {
        program.setUniform("M", M);
        glBindVertexArray(this->VAO);
        glDrawElements(GL_TRIANGLES, total_indices, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }

private:
    int total_indices = 0;
    GLuint VAO = 0;

    // ---------- OBJ loader (original code) ----------
    void loadOBJ(const std::string& objPath) {
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
            } else if (prefix == "f") {
                unsigned int v1, v2, v3;
                iss >> v1 >> v2 >> v3;
                indices.emplace_back(v1 - 1); // OBJ is 1‑based
                indices.emplace_back(v2 - 1);
                indices.emplace_back(v3 - 1);
            } else if (prefix == "vn") {
                hasNormals = true;
            }
        }
        objFile.close();
        buildBuffers(positions, indices);
    }

    // ---------- PLY loader (new) ----------
    void loadPLY(const std::string& plyPath) {
        std::ifstream plyFile(plyPath);
        if (!plyFile.is_open()) {
            std::cerr << "Failed to open file: " << plyPath << std::endl;
            return;
        }
        auto start = std::chrono::high_resolution_clock::now();

        // -- Parse header --
        std::string line;
        std::getline(plyFile, line);
        if (line != "ply") {
            std::cerr << "Not a valid PLY file (missing 'ply')." << std::endl;
            return;
        }

        size_t vertexCount = 0, faceCount = 0;
        while (std::getline(plyFile, line)) {
            // Trim leading whitespace
            size_t first = line.find_first_not_of(" \t");
            if (first == std::string::npos) continue;
            line = line.substr(first);
            if (line.rfind("comment", 0) == 0) continue;

            std::istringstream iss(line);
            std::string keyword;
            iss >> keyword;
            if (keyword == "format") {
                std::string fmt;
                iss >> fmt;
                if (fmt != "ascii") {
                    std::cerr << "Only ASCII PLY is supported." << std::endl;
                    return;
                }
            } else if (keyword == "element") {
                std::string elem;
                size_t count;
                iss >> elem >> count;
                if (elem == "vertex") vertexCount = count;
                else if (elem == "face") faceCount = count;
            } else if (keyword == "end_header") {
                break;
            }
            // ignore 'property' lines
        }

        // -- Read vertices --
        std::vector<glm::vec3> positions;
        positions.reserve(vertexCount);
        for (size_t i = 0; i < vertexCount; ++i) {
            if (!std::getline(plyFile, line)) break;
            std::istringstream iss(line);
            float x, y, z;
            if (!(iss >> x >> y >> z)) {
                std::cerr << "Error reading vertex " << i << std::endl;
                return;
            }
            positions.emplace_back(x, y, z);
            // ignore any extra properties per vertex
        }

        // -- Read faces (and triangulate n‑gons) --
        std::vector<unsigned int> indices;
        indices.reserve(faceCount * 3); // guess for triangles
        for (size_t i = 0; i < faceCount; ++i) {
            if (!std::getline(plyFile, line)) break;
            std::istringstream iss(line);
            int count;
            if (!(iss >> count)) {
                std::cerr << "Error reading face " << i << std::endl;
                return;
            }
            std::vector<unsigned int> faceVerts(count);
            for (int j = 0; j < count; ++j) {
                if (!(iss >> faceVerts[j])) {
                    std::cerr << "Error reading face " << i << " vertex " << j << std::endl;
                    return;
                }
                // PLY indices are 0‑based → keep as‑is
            }

            // Triangulate polygon as a fan (works for convex/quads etc.)
            for (int j = 1; j < count - 1; ++j) {
                indices.push_back(faceVerts[0]);
                indices.push_back(faceVerts[j]);
                indices.push_back(faceVerts[j + 1]);
            }
        }

        plyFile.close();
		auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << buildMode() << " Loaded mesh with "
                  << positions.size() << " vertices and "
                  << indices.size() << " indices, took "
                  << duration.count() << " s\n";
        buildBuffers(positions, indices);
    }

    // ---------- Common buffer building ----------
    void buildBuffers(const std::vector<glm::vec3>& positions,
                      const std::vector<unsigned int>& indices) {
        if (positions.empty() || indices.empty()) return;

        // Compute per‑vertex normals
        std::vector<glm::vec3> normals(positions.size(), glm::vec3(0.0f));
        for (size_t i = 0; i < indices.size(); i += 3) {
            unsigned int i1 = indices[i];
            unsigned int i2 = indices[i+1];
            unsigned int i3 = indices[i+2];

            const glm::vec3& v1 = positions[i1];
            const glm::vec3& v2 = positions[i2];
            const glm::vec3& v3 = positions[i3];

            glm::vec3 edge1 = v2 - v1;
            glm::vec3 edge2 = v3 - v1;
            glm::vec3 faceNormal = glm::normalize(glm::cross(edge1, edge2));

            normals[i1] += faceNormal;
            normals[i2] += faceNormal;
            normals[i3] += faceNormal;
        }
        for (auto& n : normals) {
            float len = glm::length(n);
            if (len > 0.0f)
                n /= len;
            else
                n = glm::vec3(0.0f, 1.0f, 0.0f); // fallback
        }

        // Interleaved vertex data (position + normal)
        std::vector<float> vertexData;
        vertexData.reserve(positions.size() * 6);
        for (size_t i = 0; i < positions.size(); ++i) {
            vertexData.push_back(positions[i].x);
            vertexData.push_back(positions[i].y);
            vertexData.push_back(positions[i].z);
            vertexData.push_back(normals[i].x);
            vertexData.push_back(normals[i].y);
            vertexData.push_back(normals[i].z);
        }

        // OpenGL buffers
        glGenVertexArrays(1, &VAO);
        glBindVertexArray(VAO);

        GLuint VBO, EBO;
        glGenBuffers(1, &VBO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float),
                     vertexData.data(), GL_STATIC_DRAW);

        glGenBuffers(1, &EBO);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int),
                     indices.data(), GL_STATIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                              (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);

        glBindVertexArray(0);

        total_indices = static_cast<int>(indices.size());
    }
};
