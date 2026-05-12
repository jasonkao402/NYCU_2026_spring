#pragma once

#include <glad/glad.h>
#include "asset.h"
#include <string>
#include <vector>
#include <chrono>
#include <future>


struct TextureData
{
	TextureData() {}
	int width = 0;
	int height = 0;
	std::shared_ptr<unsigned char[]> data = nullptr;
};


class Texture: public Asset::UsedBy<Asset::dummyUser>
{
public:

	Texture() {}
	Texture(int id) :id(id) {}
	Texture(std::string filePath):filePath(filePath) {
	}
	~Texture() {
		if (id != 0) {
			glDeleteTextures(1, &id);
		}
	}
	// async load, return texture Id if texture is ready, otherwise return 0
	GLuint getId() {
		if (isLoaded == false) {
			if(!futureData.valid())
				futureData = std::async(loadImage, filePath.c_str());
			if(futureData.wait_for(std::chrono::milliseconds(5)) == std::future_status::ready) {
				isLoaded = true;
				id = loadTexture(futureData.get());
			}
		}
		return id;
	}
	
private:
	GLuint id = 0;
	bool isLoaded = false;

	const std::string filePath;
	
	std::future<TextureData> futureData;



	GLuint loadTexture(TextureData texture)
	{
		GLuint textureID;
		if (texture.data == nullptr) {
			return 0;
		}

		glGenTextures(1, &textureID);
		// "Bind" the newly created texture : all future texture functions will modify this texture
		glBindTexture(GL_TEXTURE_2D, textureID);

		// Give the image to OpenGL
		// change parameter 3 to GL_COMPRESSED_RGBA to save memory
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texture.width, texture.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture.data.get());

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		glGenerateMipmap(GL_TEXTURE_2D);

		return textureID;
	}

	static TextureData loadImage(const char* path);
	

};


