#ifndef MSS_GLFWAPP_H
#define MSS_GLFWAPP_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "dart/gui/Trackball.hpp"
#include "ShapeRenderer.h"

#include "NumPyHelper.h"
#include <list>

namespace py = pybind11;
struct GLFWwindow;
namespace MASS
{

	struct NN
	{
		int cascadingType = 0;
		bool useMuscle = false;
		bool isLegacy = true;
		bool useRef = false;
		py::object sim;
		py::object muscle;
		py::object ref;
		Eigen::VectorXd minv;
		Eigen::VectorXd maxv;
	};

	class Environment;
	class Muscle;

	struct GraphElem
	{
		std::string name;
		std::deque<std::pair<double, double>> r_left;
		std::deque<std::pair<double, double>> r_right;
		std::deque<std::pair<double, double>> r_ref;
	};

	class GLFWApp
	{
	public:
		GLFWApp(int argc, char **argv);
		~GLFWApp();
		void startLoop();
		void setEnv(Environment *env, int argc, char **argv);

	private:
		py::object mns;
		py::object load_from_checkpoint;
		py::object metadata_from_checkpoint;

		std::vector<NN> mNNs;
		std::vector<py::object> mMuscleNNs;
		std::vector<double> mWeights;
		std::vector<Eigen::VectorXd> mActions;
		std::vector<double> mValues;
		std::vector<double> mAxis;

		std::vector<Eigen::VectorXd> mRefs;
		std::vector<Eigen::VectorXd> mProjState;
		std::vector<std::vector<int>> mCascadingMap;

		void drawSimFrame();
		void drawUiFrame();
		void update();

		void keyboardPress(int key, int scancode, int action, int mods);
		void mouseMove(double xpos, double ypos);
		void mousePress(int button, int action, int mods);
		void mouseScroll(double xoffset, double yoffset);
		void initGL();
		void initFog();
		void initLights();
		void draw();
		void SetFocusing();
		void Reset();

		GLFWwindow *window;
		Environment *mEnv;

		bool mFocus;
		bool mSimulating;
		bool mDrawOBJ;
		bool mDrawShadow;
		bool mPhysics;

		Eigen::Affine3d mViewMatrix;
		dart::gui::Trackball mTrackball;
		Eigen::Vector3d mTrans;
		Eigen::Vector3d mEye;
		Eigen::Vector3d mUp;

		float mZoom;
		float mPersp;
		float mMouseX, mMouseY;
		bool mMouseDown, mMouseDrag, mCapture = false;
		bool mRotate = false, mTranslate = false, mZooming = false;
		double width, height;
		double viewportWidth, imguiWidth;

		Eigen::Vector3d mCameraPos;
		ShapeRenderer mShapeRenderer;

		void DrawEntity(const dart::dynamics::Entity *entity);
		void DrawBodyNode(const dart::dynamics::BodyNode *bn);
		void DrawSingleBodyNode(const BodyNode *bn);
		void DrawSkeleton(const dart::dynamics::SkeletonPtr &skel);
		void DrawShapeFrame(const dart::dynamics::ShapeFrame *shapeFrame);
		void DrawShape(const dart::dynamics::Shape *shape, const Eigen::Vector4d &color);
		void DrawMuscles(const std::vector<Muscle *> &muscles);
		void DrawGround(double y);

		void DrawAnchorForce();
		void DrawMuscleTorque();
		void DrawCollision();
		void DrawFootStep();
		void DrawPhase(double phase, double global_phase);
		void DrawShadow(const Eigen::Vector3d &scale, const aiScene *mesh, double y);
		void DrawAiMesh(const struct aiScene *sc, const struct aiNode *nd, const Eigen::Affine3d &M, double y);

		int mSimCount;

		void DrawUIController();
		void DrawUIDisplay();

		void DrawJoint();
		void DrawNodeCOM();

		void SetMuscleColor();
		void CreateGraphData();
		void AddGraphData();

		void UpdateMuscleTorque();

		double GetProjectedAngle(Eigen::Vector3d pos, int axis);

		void SaveGraphData();

		bool mDrawFootStep;
		bool mDrawCollision;
		bool mDrawReference;
		bool mDrawMuscleTorque;
		bool mDrawBodyFrame;
		bool mUseEOE;

		std::vector<std::map<std::string, double>> mRewardBuffer;
		double reward_gx[2];

		int mCameraMoving;
		bool *mSelectedParameter;
		bool mIsSelectedMode;

		int mFramerate;
		std::vector<GraphElem> mGraphData;

		std::vector<GraphElem> mSavedGraphData;

		std::vector<std::pair<double, double>> Align(std::vector<std::pair<double, double>> l, double offset);
		int mIncludeAction;

		Eigen::VectorXd mDisplacement;
		Eigen::VectorXd mMuscleTorque;

		bool mDrawNodeCOM;
		bool mDrawJoint;

		bool mIsTorqueClip;

		bool mDrawActivation;
		bool mDrawPassiveForce;
		bool mDrawActiveForce;
		bool mDrawAnchorForce;

		bool mCalibrateGraph;

		int mResolution;
		float mPlayRate;

		double mPrevPasssiveNorm;
		float mGaitShift;
		Eigen::VectorXd prev_pos;
		GraphElem mContactData;

		std::vector<bool> mUseWeights;
	};
};

#endif
