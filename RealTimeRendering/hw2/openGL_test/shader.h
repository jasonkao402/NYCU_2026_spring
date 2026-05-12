#pragma once
#include <fstream>
#include <sstream>
#include "asset.h"
#include "program.h"


class shader: public Asset::UsedBy<Program>
{
public:
	shader() {
	}

	shader(GLenum type, std::string&& filePath):type(type), filePath(filePath) {
		
	}

	// move
	shader(shader&& that) noexcept :filePath(that.filePath) {
		this->type = that.type;
		this->id = that.id;
		that.id = 0;
	}
	~shader() {
		shader::deleteShader(id);
	}

	void reload() {
		GLuint oldId = id;
		if (oldId == 0)
			return;

		this->id = loadShader(type, filePath);
		for (auto user : users) {
			user->reloadShader(type, this->id, oldId);
		}
		shader::deleteShader(oldId);
	}

	inline GLuint getId() {
		if (id == 0) {
			this->id = loadShader(type, filePath);
		}
		return this->id;
	}
	GLenum type = 0;

private:
	const std::string filePath;
	GLuint id = 0;

	static void deleteShader(GLuint _id) {
		if (_id != 0)
			glDeleteShader(_id);
	}
	static GLuint loadShader(GLenum type, const std::string& filePath) {

		std::ifstream ShaderStream(filePath, std::ios::in);
		std::string ShaderCode;

		if (ShaderStream.is_open()) {
			std::stringstream sstr;
			sstr << ShaderStream.rdbuf();
			ShaderCode = sstr.str();
			ShaderStream.close();
		}
		else {
			printf("Impossible to open  %s.\n", filePath.c_str());
			return 0;
		}

		GLint Result = GL_FALSE;
		int InfoLogLength;

		GLuint ShaderID = glCreateShader(type);
		printf("Compiling shader : %s.\n", filePath.c_str());
		char const* VertexSourcePointer = ShaderCode.c_str();
		glShaderSource(ShaderID, 1, &VertexSourcePointer, NULL);
		glCompileShader(ShaderID);

		glGetShaderiv(ShaderID, GL_COMPILE_STATUS, &Result);
		glGetShaderiv(ShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
		if (InfoLogLength > 0) {
			std::vector<char> ShaderErrorMessage(InfoLogLength + 1);
			glGetShaderInfoLog(ShaderID, InfoLogLength, NULL, &ShaderErrorMessage[0]);
			printf("%s\n", &ShaderErrorMessage[0]);
		}

		return ShaderID;
	}
};



