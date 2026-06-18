#pragma once
#include <GLM/glm.hpp>
#include <glad/glad.h>
#include <vector>
#include <cstring>

#include "asset.h"

class shader;

class Program
{
public:
	Program();
	Program(std::vector<std::shared_ptr<shader>>& shaders);
	Program(std::vector<std::shared_ptr<shader>>&& shaders);
	~Program();
	GLuint ID = 0;
	
	Program& attachShader(std::shared_ptr<shader>);
	Program& attachShaders(std::vector<std::shared_ptr<shader>>& shaders);

	Program& detachShader(std::shared_ptr<shader>);
	Program& reloadShader(GLenum type, GLuint newId, GLuint previousId);

	Program& link();

	// forward type (rval, lval)
	template <typename T>
	inline Program& setUniform(const char* target, T&& value) {
		return setUniform(target, value);
	}

	Program& setUniform(const char* target, GLuint value);
	Program& setUniform(const char* target, int value);
	Program& setUniform(const char* target, float value);
	Program& setUniform(const char* target, glm::vec3& value);
	Program& setUniform(const char* target, glm::mat4& value);

	Program& setTexture(const char* target, GLuint textureID, int binding);
	// bind block indices to points indeices
	Program& bindUniformBlock(const char* target, unsigned int points);

	Program& bindShaderStorageBlock(const char* target, unsigned int points);

	Program& enable();
	static enum shaderType {
		vs, fs, gs, tcs, tes
	};
private:
	std::shared_ptr<shader> _shader[5];
};
