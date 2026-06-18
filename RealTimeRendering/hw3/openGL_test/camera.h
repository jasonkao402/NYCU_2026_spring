#pragma once
#include <GLM/gtc/matrix_transform.hpp>
#include <GLM/glm.hpp>
#include <GLFW/glfw3.h>
#include <memory>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <vector>
#include "Mouse.h"

class CameraActionImp;

class Camera
{
public:
	Camera(GLFWwindow* window, int width, int height, glm::vec3 initPos,
		std::vector<std::shared_ptr<CameraActionImp>> impl);
	~Camera();
	GLFWwindow* window;

	float speed = 20.0f;
	float mouseSpeed = 0.006f;
	float far = 300.0f;
	float near = 0.5f;

	glm::quat rotation;

	glm::vec3 position;
	glm::vec3 direction;
	glm::vec3 up;
	glm::vec3 right;
	glm::vec2 viewAngle;

	glm::mat4 Projection;
	glm::mat4 View;

	void action(float deltaTime, Mouse& mouse) ;

	void setDirection();
private:
	std::vector<std::shared_ptr<CameraActionImp>> impl;


};




