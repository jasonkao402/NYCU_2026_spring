#include "asset.h"
#include "shader.h"
#include "texture.h"
#include "model.h"

#include <map>

// extension to GLenum
static std::map<std::string, GLuint> convertTpye = {
    {".vs", GL_VERTEX_SHADER},
    {".fs", GL_FRAGMENT_SHADER},
    {".gs", GL_GEOMETRY_SHADER},
    {".tcs", GL_TESS_CONTROL_SHADER},
    {".tes", GL_TESS_EVALUATION_SHADER}
};


void Asset::AssetController::getFiles(std::string&& directoryPath)
{
    this->getFiles(directoryPath);
}

void Asset::AssetController::getFiles(std::string& directoryPath)
{
    if (!fs::exists(directoryPath) || !fs::is_directory(directoryPath)) {
        std::cout << "The path: " << directoryPath << " does not exist or is not a directory.\n";
        return;
    }

    for (const auto& entry : fs::directory_iterator(directoryPath)) {
        if (fs::is_directory(entry)) {
            this->getFiles(entry.path().string());
        }
        if (!fs::is_regular_file(entry)) {
            continue;
        }

        auto& filePath = entry.path();
        auto time = entry.last_write_time();

        auto timestamp_It = timestamp.find(filePath.string());
        if (timestamp_It == timestamp.end()) {
            // new item
            getFile(filePath);
            timestamp[filePath.string()] = time;
        }
        else {
            // has value
            if (timestamp_It->second != time) {
                // change
                reloadFile(filePath);
                timestamp[filePath.string()] = time;

            }
        }
    }
}


void Asset::AssetController::getFile(const fs::path& filePath)
{
    // new item
    auto extension = filePath.extension();
    if (extension == ".glsl") {
        loadShader(filePath);
    }
    else if (extension == ".png" || extension == ".jpg" || extension == ".jpeg") {
        loadTexture(filePath);
    }
}

void Asset::AssetController::reloadFile(const fs::path& filePath)
{
    // file changed
    auto extension = filePath.extension();
    if (filePath.extension() == ".glsl") {
        reloadShader(filePath);
    }
    else if (extension == ".png" || extension == ".jpg" || extension == ".jpeg") {
        reloadTexture(filePath);
    }
}


void Asset::AssetController::loadShader(const fs::path& filePath)
{
    // shaderType is one of vs, fs, gs, tcs, tes.
    const auto& shaderType = filePath.stem().extension().string();

    auto newShader = std::make_shared<shader>(convertTpye[shaderType], filePath.string());
    shaders.emplace(filePath.filename().string(), newShader);
}

void Asset::AssetController::reloadShader(const fs::path& filePath)
{
    auto it = shaders.find(filePath.filename().string());
    if (it != shaders.end())
        it->second->reload();
}

void Asset::AssetController::loadTexture(const fs::path& filePath)
{
    auto newTexture = std::make_shared<Texture>(filePath.string());
    textures.emplace(filePath.filename().string(), newTexture);
}

void Asset::AssetController::reloadTexture(const fs::path& filePath)
{
    auto it = textures.find(filePath.filename().string());
    if (it != textures.end())
        it->second->reload();
}
