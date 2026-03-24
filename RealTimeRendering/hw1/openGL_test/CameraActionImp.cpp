#include "CameraActionImp.h"
#include "camera.h"

namespace CameraAction {

	void _WASDControlMove::move(float deltaTime)
	{
		auto window = instance->window;
		float speed = instance->speed * deltaTime;

		if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
			instance->position += instance->direction * speed;
		}
		if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
			instance->position -= instance->direction * speed;
		}
		if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
			instance->position += instance->right * speed;
		}
		if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
			instance->position -= instance->right * speed;
		}
	}

	template _DragControlView<Mouse::LEFT>;
	template _DragControlView<Mouse::MIDDLE>;
	template _DragControlView<Mouse::RIGHT>;

	
	template <int BUTTON>
	void _DragControlView<BUTTON>::view(float deltaTime)
	{
		auto& mouse = Mouse::get();
		if (!mouse.buttons[BUTTON].drag) {
			return;
		}

		instance->viewAngle -= vec2(mouse.cursorPos - mouse.previousCursorPos) * instance->mouseSpeed;
		instance->setDirection();
		
	}

	void mouseControlBoth::move(float deltaTime)
	{
		auto& mouse = Mouse::get();
		float speed = instance->speed * mouse.scroll * 0.5;
		instance->position += instance->direction * speed;

		if (mouse.buttons[Mouse::MIDDLE].drag) {
			auto offest = vec2(mouse.cursorPos - mouse.previousCursorPos);
			auto dir = 0.1f *  glm::mat2x3(-instance->right, instance->up) * offest;
			instance->position += dir;
		}

		
	}

	void FPS_Control::view(float deltaTime)
	{
		auto& mouse = Mouse::get();
		instance->viewAngle -= vec2(mouse.cursorPos - mouse.previousCursorPos) * instance->mouseSpeed;
		instance->setDirection();
	}

}