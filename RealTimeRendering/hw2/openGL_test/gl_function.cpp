#include "gl_function.h"
#include "utils.h"

void gl::genUniformbuffer(GLuint& point, GLuint& buffer, unsigned long long allocSize)
{
	static GLuint point_counter = 0;
	point = point_counter++;

	glGenBuffers(1, &buffer);
	glBindBuffer(GL_UNIFORM_BUFFER, buffer);
	glBufferData(GL_UNIFORM_BUFFER, allocSize, nullptr, GL_DYNAMIC_DRAW);
	glBindBufferBase(GL_UNIFORM_BUFFER, point, buffer);
	glBindBuffer(GL_UNIFORM_BUFFER, 0); // unbind

}

void gl::genShaderStorageBuffer(GLuint& point, GLuint& buffer, unsigned long long allocSize)
{
	static GLuint point_counter = 0;
	point = point_counter++;
	glGenBuffers(1, &buffer);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
	glBufferData(GL_SHADER_STORAGE_BUFFER, allocSize, nullptr, GL_DYNAMIC_READ);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, point, buffer);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // unbind
}


