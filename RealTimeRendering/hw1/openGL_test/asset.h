#pragma once
#include <string>
#include <filesystem>
#include <iostream>
#include <optional>
#include <map>
#include <memory>

namespace fs = std::filesystem;

class shader;
class Texture;
class mesh;

namespace Asset {
class dummyUser;

template <typename USER>
class UsedBy
{
public:
	UsedBy() {

	}
	~UsedBy() {

	}

	virtual void reload() {
		static bool print = false;
		if(!print)
			std::cout << "Error: no implement reload function.\n";
		print = true;
	}
	void addUser(USER* user) {
		users.push_back(user);
	}
	void removeUser(USER* user) {
		users.erase(std::remove(users.begin(), users.end(), user), users.end());
	}
	std::vector<USER*> users;

private:
	
};



class AssetController
{
public:
	static inline AssetController& get() {
		static std::shared_ptr<AssetController> asset = nullptr;
		if (asset == nullptr) {
			asset = std::make_shared<AssetController>();
		}
		return *asset;
	}

	AssetController() {
		getFiles(this->root());
	}
	~AssetController() {

	}

	// create file entry or call reload callback
	void getFiles(std::string& directoryPath);
	// create file entry or call reload callback
	void getFiles(std::string&& directoryPath);

	// get shader file entry. When call getID(), the file will be loaded
	inline std::shared_ptr<shader> getShader(const char* name) {
		return getItem(shaders, name);
	};

	inline std::vector<std::shared_ptr<shader>> getShaders(std::vector<const char*>&& names) {
		return getItems(shaders, names);
	};

	// get texture file entry. When call getId(), the file will be loaded
	inline std::shared_ptr<Texture> getTexture(const char* name) {
		return getItem(textures, name);
	};

	inline std::vector<std::shared_ptr<Texture>> getTextures(std::vector<const char*>&& names) {
		return getItems(textures, names);
	};

	inline std::string root() {
		return rootPath.string();
	}
private:
	fs::path rootPath = "./Asset";

	std::map<std::string, std::shared_ptr<shader>> shaders;
	std::map<std::string, std::shared_ptr<Texture>> textures;

	std::map<std::string, fs::file_time_type > timestamp;

	// called from getFiles, forward by file extension
	void getFile(const fs::path& filePath);
	// called from getFiles, forward by file extension
	void reloadFile(const fs::path& filePath);

	// called by getFile 
	void loadShader(const fs::path& filePath);
	// called by reloadFile
	void reloadShader(const fs::path& filePath);

	// called by getFile
	void loadTexture(const fs::path& filePath);
	// called by reloadFile
	void reloadTexture(const fs::path& filePath);


	// find the file from itemSet
	template <typename T>
	inline std::shared_ptr<T> getItem(std::map<std::string, std::shared_ptr<T>>& itemSet, const char* name) {
		auto it = itemSet.find(name);
		if (it != itemSet.end())
			return it->second;
		std::cout << "Cannot found " << name << "\n";
		return nullptr;
	};

	template <typename T>
	inline std::vector<std::shared_ptr<T>> getItems(std::map<std::string, std::shared_ptr<T>>& itemSet, std::vector<const char*>& names) {
		std::vector<std::shared_ptr<T>> collect;
		for (auto& name : names) {
			collect.emplace_back(this->getItem(itemSet, name));
		}
		return collect;
	};

};

static inline AssetController& get() {
	return AssetController::get();
}

} // namespace Asset
