#pragma once

#include<iostream>
#include<glad/glad.h>
#include<GLFW/glfw3.h>
#include <GLM/glm.hpp>
#include <GLM/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <memory>

namespace gl {
	/*
	* generate uniform buffer
	*
	* @param[in] allocSize
	* @return bindding point, buffer
	*/
	void genUniformbuffer(GLuint& point, GLuint& buffer, unsigned long long allocSize);

	void genShaderStorageBuffer(GLuint& point, GLuint& buffer, unsigned long long allocSize);


}