#include "texture.h"
#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif // !STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

TextureData Texture::loadImage(const char* path)
{
	TextureData texture;
	int n;
	stbi_set_flip_vertically_on_load(true);

	stbi_uc* data = stbi_load(path, &texture.width, &texture.height, &n, 4);
	if (data != NULL)
	{
		std::cout << "Load texture: " << path << "\n";

		texture.data.reset(data);
	}
	else {
		std::cout << "No file exist: " << path << "\n";
	}
	return texture;
}