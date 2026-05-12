#include "camera.h"
#include "Mouse.h"
#include "CameraActionImp.h"

Camera::Camera(GLFWwindow* window, int width, int height, glm::vec3 initPos, std::vector<std::shared_ptr<CameraActionImp>> impl)
{
	this->impl = impl;
	for (auto i : impl) {
		i->setInstance(this);
	}

	this->position = initPos;
	this->window = window;

	this->direction = glm::vec3();
	this->right = glm::vec3();
	this->up = glm::vec3();
	this->viewAngle = glm::vec2(glm::pi<float>(), 0.0f);

	this->Projection = glm::perspective(glm::radians(45.0f), (float)width / (float)height, this->near, this->far);


	this->setDirection();

	this->View = glm::lookAt(
		this->position, 
		this->position + this->direction, 
		this->up
	);
}

Camera::~Camera()
{
}

void Camera::action(float deltaTime, Mouse& mouse)
{
	for(auto i : impl)
		i->action(deltaTime, mouse);

	this->View = glm::lookAt(
		this->position,
		this->position + this->direction,
		this->up
	);
}

void Camera::setDirection()
{
	double x = (double)this->viewAngle.x;
	double y = (double)this->viewAngle.y;

	// limit y max to 90
	if (y > glm::half_pi<double>()) {
		this->viewAngle.y = glm::half_pi<float>();
	}
	if (y < -glm::half_pi<double>()) {
		this->viewAngle.y = -glm::half_pi<float>();
	}
	this->direction.x = float(cos(y) * sin(x));
	this->direction.y = (float)sin(y);
	this->direction.z = float(cos(y) * cos(x));

	this->right.x = (float)sin(x - 3.14f / 2.0f);
	this->right.z = (float)cos(x - 3.14f / 2.0f);


	this->up = glm::cross(this->right, this->direction);
}