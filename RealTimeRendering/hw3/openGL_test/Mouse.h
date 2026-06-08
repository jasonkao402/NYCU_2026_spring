#pragma once
#include <memory>
#include <GLFW/glfw3.h>
#include "imgui.h"


class Button {
public:

	bool press = false;
	bool release = false;
	bool drag = false;
	bool click = false;
	GLenum BUTTON_ID = 0;

	void setID(GLenum BUTTON_ID) {
		this->BUTTON_ID = BUTTON_ID;
	}
	void reset() {
		release = false;
	}
	void callback(int button, int action) {
		if (button == this->BUTTON_ID && action == GLFW_PRESS)
		{
			press = true;
		}
		if (button == this->BUTTON_ID && action == GLFW_RELEASE)
		{
			release = true;
		}
	}
	void update(double deltaTime, bool mouseMove = false) {
		click = false;
		if (press) {
			t_mouseDowm += deltaTime;
			if (t_mouseDowm >= t_clickInterval || mouseMove) {
				drag = true;
			}
		}
		if (release) {
			if (t_mouseDowm < t_clickInterval) {
				click = true;
				
			}
			press = false;
			drag = false;
			t_mouseDowm = 0.0;
		}
	}

private:
	double t_mouseDowm = 0.0;
	double t_clickInterval = 0.2;
};

class Mouse
{
public:
	enum ButtonId {LEFT, MIDDLE, RIGHT};

	Mouse() {
		buttons[LEFT].setID(GLFW_MOUSE_BUTTON_LEFT);
		buttons[MIDDLE].setID(GLFW_MOUSE_BUTTON_MIDDLE);
		buttons[RIGHT].setID(GLFW_MOUSE_BUTTON_RIGHT);
	}
	

	static Mouse& get() {
		static std::shared_ptr<Mouse> instance = nullptr;
		if (!instance) {
			instance = std::make_shared<Mouse>();
		}
		return *instance;
	}

	void reset() {
		this->previousCursorPos = this->cursorPos;
		this->scroll = 0.0;
		buttons[LEFT].reset();
		buttons[MIDDLE].reset();
		buttons[RIGHT].reset();
	}
	void update(double deltaTime) {
		auto move = this->previousCursorPos == this->cursorPos;
		buttons[LEFT].update(deltaTime, move);
		buttons[MIDDLE].update(deltaTime, move);
		buttons[RIGHT].update(deltaTime, move);
	};

	void buttonCallback(int button, int action) {
		buttons[LEFT].callback(button, action);
		buttons[MIDDLE].callback(button, action);
		buttons[RIGHT].callback(button, action);
	};
	void scrollCallback(double offset) {
		scroll = offset;
	};

	Button buttons[3];

	double scroll = 0.0;
	vec<2, double> previousCursorPos = vec2();
	vec<2, double> cursorPos = vec2();
private:
	
};


void inline mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
	ImGuiIO& io = ImGui::GetIO();
	Mouse& mouse = Mouse::get();
	if (io.WantCaptureMouse)
		return;

	mouse.buttonCallback(button, action);
}

void inline scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
	Mouse& mouse = Mouse::get();
	mouse.scrollCallback(yoffset);
}