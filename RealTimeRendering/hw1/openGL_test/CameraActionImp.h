#pragma once
#include "Mouse.h"

class Camera;

class CameraActionImp
{
public:
	CameraActionImp() {};

	void setInstance(Camera* camera) {
		instance = camera;
	}
	CameraActionImp(Camera& camera) :instance(&camera) {};
	~CameraActionImp() {};
	virtual void move(float deltaTime) = 0;
	virtual void view(float deltaTime) = 0;
	void action(float deltaTime) {
		this->view(deltaTime);
		this->move(deltaTime);
	};

protected:
	Camera* instance = nullptr;

};

namespace CameraAction {
class doNothing :public CameraActionImp
{
public:
	void move(float deltaTime) override {};
	void view(float deltaTime) override {};
private:

};

class _WASDControlMove :virtual public CameraActionImp
{
public:
	void move(float deltaTime) override;
private:

};

template <int BUTTON>
class _DragControlView :virtual public CameraActionImp
{
public:
	void view(float deltaTime) override;
private:

};


/* FPS game control mode
Move: WASD
View: mouse
*/

class WASDControlMove :virtual public _WASDControlMove
{
public:
	void view(float deltaTime) {};
};

class FPS_Control :public _WASDControlMove
{
public:
	void view(float deltaTime) override;
};

/*
Move: WASD
View: mouse(Drag)
*/
class mouseControlView :public _WASDControlMove, _DragControlView<Mouse::LEFT>
{};

/*
Move: mouse(middle)
View: mouse(Drag)
*/
class mouseControlBoth :public _DragControlView<Mouse::RIGHT>
{
public:
	void move(float deltaTime) override;
};
} // namespace CameraAction