
#include "Program.h"
#include "shader.h"

using namespace glm;

// GLuint to int
static std::map<GLuint, int> Type2Index = {
	{GL_VERTEX_SHADER, Program::vs},
	{GL_FRAGMENT_SHADER, Program::fs},
	{GL_GEOMETRY_SHADER, Program::gs},
	{GL_TESS_CONTROL_SHADER, Program::tcs},
	{GL_TESS_EVALUATION_SHADER, Program::tes}
};

Program::Program()
{
	GLuint ProgramID = glCreateProgram();
	this->ID = ProgramID;
}

Program::Program(std::vector<const char*>&& names)
{
	auto& asset = Asset::get();
	GLuint ProgramID = glCreateProgram();
	this->ID = ProgramID;
	this->attachShaders(asset.getShaders(std::move(names)));
	this->link();
}


Program::~Program()
{
	if(this->ID != 0)
		glDeleteProgram(this->ID);
}

Program& Program::attachShader(std::shared_ptr<shader> shader)
{
	if (shader == nullptr) return (*this);

	auto index = Type2Index[shader->type];

	// detach previous shader
	if (this->_shader[index])
		this->detachShader(this->_shader[index]);

	shader->addUser(this);
	glAttachShader(this->ID, shader->getId());

	this->_shader[index] = shader;
	return *this;
}

Program& Program::attachShaders(std::vector<std::shared_ptr<shader>>&& shaders)
{
	for (auto& shader : shaders) {
		this->attachShader(shader);
	}
	return *this;
}

Program& Program::detachShader(std::shared_ptr<shader> shader)
{
	if (shader == nullptr) return (*this);

	auto index = Type2Index[shader->type];

	if (shader == this->_shader[index]) {
		glDetachShader(this->ID, shader->getId());
		shader->removeUser(this);
	}

	this->_shader[index] = nullptr;
	return *this;
}

Program& Program::reloadShader(GLenum type, GLuint newId, GLuint previousId)
{
	glDetachShader(this->ID, previousId);
	glAttachShader(this->ID, newId);
	link();
	return *this;
}

Program& Program::link()
{
	GLint Result = GL_FALSE;
	int InfoLogLength;
	glLinkProgram(this->ID);
	// Check the Program
	glGetProgramiv(this->ID, GL_LINK_STATUS, &Result);
	glGetProgramiv(this->ID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0) {
		std::vector<char> ProgramErrorMessage(InfoLogLength + 1);
		glGetProgramInfoLog(this->ID, InfoLogLength, NULL, &ProgramErrorMessage[0]);
		printf("%s\n", &ProgramErrorMessage[0]);
	}

	
	return *this;
}

Program& Program::setUniform(const char* target, mat4& value)
{
	GLuint uniform_id = glGetUniformLocation(this->ID, target);
	if (uniform_id != -1)
		glUniformMatrix4fv(uniform_id, 1, GL_FALSE, &value[0][0]);
	return *this;
}

Program& Program::setUniform(const char* target, glm::vec3& value)
{
	GLuint uniform_id = glGetUniformLocation(this->ID, target);
	if (uniform_id != -1)
		glUniform3fv(uniform_id, 1, &value[0]);
	return *this;
}

Program& Program::setUniform(const char* target, glm::vec4& value)
{
	GLuint uniform_id = glGetUniformLocation(this->ID, target);
	if (uniform_id != -1)
		glUniform4fv(uniform_id, 1, &value[0]);
	return *this;
}

Program& Program::setUniform(const char* target, float value)
{
	GLuint uniform_id = glGetUniformLocation(this->ID, target);
	if (uniform_id != -1)
		glUniform1f(uniform_id, value);
	return *this;
}

Program& Program::setUniform(const char* target, GLuint value)
{
	GLuint uniform_id = glGetUniformLocation(this->ID, target);
	if (uniform_id != -1)
		glUniform1ui(uniform_id, value); // set it manually
	return *this;
}

Program& Program::setUniform(const char* target, int value)
{
	GLuint uniform_id = glGetUniformLocation(this->ID, target);
	if (uniform_id != -1)
		glUniform1i(uniform_id, value); // set it manually
	return *this;
}

Program& Program::setTexture(const char* target, GLuint textureID, int binding)
{
	if (binding >= 32) {
		throw std::exception();
		return *this;
	}

	this->setUniform(target, binding); //set index
	glActiveTexture(GL_TEXTURE0 + binding);
	glBindTexture(GL_TEXTURE_2D, textureID);
	return *this;
}


Program& Program::bindUniformBlock(const char* target, unsigned int points)
{
	GLuint index = glGetUniformBlockIndex(this->ID, target);
	glUniformBlockBinding(this->ID, index, points);
	return *this;
}

Program& Program::bindShaderStorageBlock(const char* target, unsigned int points) {
	GLuint index = glGetProgramResourceIndex(this->ID, GL_SHADER_STORAGE_BLOCK, target);
	glShaderStorageBlockBinding(this->ID, index, points);
	return *this;
}


Program& Program::enable()
{
	glUseProgram(this->ID);
	return *this;
}
