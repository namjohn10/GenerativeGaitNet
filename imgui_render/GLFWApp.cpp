#include "GLFWApp.h"
#include <glad/glad.h>
#include <GL/glu.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <implot.h>
#include <examples/imgui_impl_glfw.h>
#include <examples/imgui_impl_opengl3.h>
#include <imgui_internal.h>
#include <iostream>
#include <experimental/filesystem>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include "Environment.h"
#include "BVH.h"
#include "Muscle.h"
#include "GLfunctions.h"

using namespace MASS;

static const double PI = acos(-1);

namespace fs = std::experimental::filesystem;

bool compare(std::pair<double, double> a, std::pair<double, double> b)
{
    if (a.first == b.first)
        return true;
    else
        return a.first < b.first;
}

GLFWApp::GLFWApp(int argc, char **argv)
    : mFocus(true), mSimulating(false), mDrawOBJ(true), mDrawShadow(false),
      mPhysics(true),
      mTrans(0.0, 0.0, 0.0),
      mEye(0.0, 0.0, 1.0),
      mUp(0.0, 1.0, 0.0),
      mZoom(1.0),
      mPersp(45.0),
      mRotate(false),
      mTranslate(false),
      mZooming(false),
      mFramerate(60),

      mDrawFootStep(false),
      mDrawCollision(true),
      mDrawReference(false),
      mDrawMuscleTorque(false),

      mIsSelectedMode(false),
      mIncludeAction(0),

      mDrawNodeCOM(false),
      mDrawJoint(false),

      mDrawActivation(false),
      mDrawPassiveForce(false),
      mDrawActiveForce(false),
      mDrawAnchorForce(false),

      mResolution(100),

      mPlayRate(1),

      mPrevPasssiveNorm(100000),

      mGaitShift(0),
      mCalibrateGraph(false),
      mDrawBodyFrame(false),
      mUseEOE(false),
      mIsTorqueClip(false)

{

    mSimCount = 0;
    width = 1920;
    height = 1080;
    viewportWidth = 1920;
    imguiWidth = 400;
    mTrackball.setTrackball(Eigen::Vector2d(viewportWidth * 0.5, height * 0.5), viewportWidth * 0.5);
    mTrackball.setQuaternion(Eigen::Quaterniond(Eigen::AngleAxisd(0, Eigen::Vector3d::UnitY())));

    mZoom = 0.25;
    mFocus = false;

    mns = py::module::import("__main__").attr("__dict__");
    py::module::import("sys").attr("path").attr("insert")(1, (std::string(MASS_ROOT_DIR) + "/python").c_str());

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
    window = glfwCreateWindow(width, height, "render", nullptr, nullptr);
    if (window == NULL)
    {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        exit(EXIT_FAILURE);
    }
    glViewport(0, 0, width, height);

    glfwSetWindowUserPointer(window, this);
    auto framebufferSizeCallback = [](GLFWwindow *window, int width, int height)
    {
        GLFWApp *app = static_cast<GLFWApp *>(glfwGetWindowUserPointer(window));
        app->width = width;
        app->height = height;
        glViewport(0, 0, width, height);
    };
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    auto keyCallback = [](GLFWwindow *window, int key, int scancode, int action, int mods)
    {
        auto &io = ImGui::GetIO();
        if (!io.WantCaptureKeyboard)
        {
            GLFWApp *app = static_cast<GLFWApp *>(glfwGetWindowUserPointer(window));
            app->keyboardPress(key, scancode, action, mods);
        }
    };

    glfwSetKeyCallback(window, keyCallback);
    auto cursorPosCallback = [](GLFWwindow *window, double xpos, double ypos)
    {
        auto &io = ImGui::GetIO();
        if (!io.WantCaptureMouse)
        {
            GLFWApp *app = static_cast<GLFWApp *>(glfwGetWindowUserPointer(window));
            app->mouseMove(xpos, ypos);
        }
    };

    glfwSetCursorPosCallback(window, cursorPosCallback);

    auto mouseButtonCallback = [](GLFWwindow *window, int button, int action, int mods)
    {
        auto &io = ImGui::GetIO();
        if (!io.WantCaptureMouse)
        {
            GLFWApp *app = static_cast<GLFWApp *>(glfwGetWindowUserPointer(window));
            app->mousePress(button, action, mods);
        }
    };
    glfwSetMouseButtonCallback(window, mouseButtonCallback);

    auto scrollCallback = [](GLFWwindow *window, double xoffset, double yoffset)
    {
        auto &io = ImGui::GetIO();
        if (!io.WantCaptureMouse)
        {
            GLFWApp *app = static_cast<GLFWApp *>(glfwGetWindowUserPointer(window));
            app->mouseScroll(xoffset, yoffset);
        }
    };
    glfwSetScrollCallback(window, scrollCallback);

    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 150");
    ImPlot::CreateContext();
    SetFocusing();

    reward_gx[0] = 0.0;
    reward_gx[1] = 4.0;

    mCameraMoving = 0;
    mTrackball.setQuaternion(Eigen::Quaterniond::Identity());
    mSavedGraphData.clear();
}

GLFWApp::~GLFWApp()
{
    ImPlot::DestroyContext();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
}

void GLFWApp::setEnv(Environment *env, int argc, char **argv)
{
    mEnv = env;
    int num_checkpoint = argc - 1;
    std::vector<std::string> checkpoint_paths;

    for (int i = 1; i <= argc - 1; i++)
        checkpoint_paths.push_back(argv[i]);

    if (!strcmp(&argv[1][strlen(argv[1]) - 3], "txt"))
    {
        mEnv->Initialize_from_path(argv[1]);
        num_checkpoint--;
        checkpoint_paths.erase(checkpoint_paths.begin());
    }
    else
    {
        py::object metadata_from_checkpoint = py::module::import("ray_model").attr("metadata_from_checkpoint");
        std::string metadata = metadata_from_checkpoint(checkpoint_paths.back()).cast<std::string>();
        mEnv->Initialize_from_text(metadata);
    }

    py::exec("import torch", mns);
    py::exec("import torch.nn as nn", mns);
    py::exec("import torch.optim as optim", mns);
    py::exec("import torch.nn.functional as F", mns);
    py::exec("import torchvision.transforms as T", mns);
    py::exec("import numpy as np", mns);

    std::cout << std::endl
              << "Environment Information " << std::endl;
    std::cout << "====================" << std::endl;
    std::cout << "[Network 01]" << std::endl;
    std::cout << "Input : " << mEnv->GetNumState() << std::endl;
    std::cout << "Output : " << mEnv->GetNumAction() << std::endl;
    std::cout << "====================" << std::endl;

    if (mEnv->GetUseMuscle())
    {
        std::cout << "[Network 02]" << std::endl;
        std::cout << "Input : " << mEnv->GetNumTotalRelatedDofs() + mEnv->GetNumAction() << std::endl;
        std::cout << "( TotalRelatedDof : " << mEnv->GetNumTotalRelatedDofs() << ", NumAction : " << mEnv->GetNumAction() << " ) " << std::endl;
        std::cout << "Output : " << mEnv->GetCharacter()->GetMuscles().size() << std::endl;
        std::cout << "====================" << std::endl;
    }

    load_from_checkpoint = py::module::import("ray_model").attr("load_from_checkpoint");
    metadata_from_checkpoint = py::module::import("ray_model").attr("metadata_from_checkpoint");
    int num_state = mEnv->GetNumState();
    int num_active_dof = mEnv->GetNumActiveDof();
    int num_action = mEnv->GetNumAction();

    for (int i = 0; i < checkpoint_paths.size(); i++)
    {

        NN NN_elem;
        py::tuple ret;

        int num_param = mEnv->GetNumParamState();
        NN_elem.useMuscle = mEnv->GetUseMuscle();

        int cascading_type = mEnv->GetCascadingType(metadata_from_checkpoint(checkpoint_paths[i]).cast<std::string>());
        std::pair<Eigen::VectorXd, Eigen::VectorXd> state_space = mEnv->GetSpace(metadata_from_checkpoint(checkpoint_paths[i]).cast<std::string>());

        NN_elem.cascadingType = cascading_type;
        int action_diff = 0;
        if (cascading_type != 0)
            action_diff += 1;

        if (NN_elem.useMuscle)
        {
            int num_total_muscle_related_dofs = mEnv->GetNumTotalRelatedDofs();
            int num_muscles = mEnv->GetCharacter()->GetMuscles().size();
            int num_proj_state = mEnv->GetProjState(state_space.first, state_space.second).rows();
            ret = load_from_checkpoint(checkpoint_paths[i], num_proj_state, num_action + action_diff, num_param, num_active_dof,
                                       num_muscles, num_total_muscle_related_dofs, (i == (checkpoint_paths.size() - 1)));

            NN_elem.muscle = ret[1];
        }
        else
            ret = load_from_checkpoint(checkpoint_paths[i], num_state, num_action + action_diff, num_param);

        NN_elem.sim = ret[0];

        NN_elem.minv = state_space.first;
        NN_elem.maxv = state_space.second;

        NN_elem.useRef = mEnv->UseDisplacement();

        if (NN_elem.useRef)
            NN_elem.ref = ret[3];

        NN_elem.isLegacy = !ret[4].cast<bool>();
        mNNs.push_back(NN_elem);
        mUseWeights.push_back(true);
        mUseWeights.push_back(true);
    }

    for (int i = 0; i < checkpoint_paths.size(); i++)
    {
        std::vector<int> child_models;
        mCascadingMap.push_back(child_models);
        mWeights.push_back(1.0);
    }

    std::ifstream ifs;
    ifs.open("../data/cascading_map.txt");
    if (ifs.is_open())
    {
        std::vector<int> child_model;
        while (!ifs.eof())
        {
            std::string str;
            std::stringstream ss;
            std::getline(ifs, str);
            ss.str(str);
            int current_model_idx = 0;
            int child_model_idx = 0;

            ss >> current_model_idx >> child_model_idx;

            if (current_model_idx == child_model_idx)
                continue;

            if (current_model_idx < mCascadingMap.size() && child_model_idx < mCascadingMap.size())
            {
                if (std::find(mCascadingMap[current_model_idx].begin(), mCascadingMap[current_model_idx].end(), child_model_idx) == mCascadingMap[current_model_idx].end())
                    mCascadingMap[current_model_idx].push_back(child_model_idx);
            }
        }
        ifs.close();
    }
    else
        exit(-1);

    mEnv->CreateTotalParams();
    mEnv->SetParamState(mEnv->GetNormalV());

    mMuscleTorque = Eigen::VectorXd::Zero(mEnv->GetCharacter()->GetSkeleton()->getNumDofs());
    CreateGraphData();
    Reset();

    mSelectedParameter = new bool[mEnv->getMuscleLengthParams().size() + mEnv->getMuscleForceParams().size()]();
    for (int i = 0; i < mEnv->getMuscleLengthParams().size() + mEnv->getMuscleForceParams().size(); i++)
        mSelectedParameter[i] = false;

    mFocus = true;

    mDisplacement = Eigen::VectorXd::Zero(mEnv->GetNumAction());
}
void GLFWApp::startLoop()
{
    double display_time = 0.0;
    while (!glfwWindowShouldClose(window))
    {

        glfwPollEvents();
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        {
            glfwSetWindowShouldClose(window, true);
        }
        if (mSimulating && (!mUseEOE || !mEnv->IsEndOfEpisode()))
            update();

        display_time = glfwGetTime();

        mIsSelectedMode = false;
        SetMuscleColor();
        drawSimFrame();
        drawUiFrame();
        glfwSwapBuffers(window);
    }
}

void GLFWApp::drawSimFrame()
{
    draw();
}

void GLFWApp::DrawUIController()
{
    ImGui::SetNextWindowSize(ImVec2(400, 200), ImGuiCond_Once);
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Once);
    ImGui::Begin("Controller");
    if (ImGui::CollapsingHeader("Parameter"))
    {
        Eigen::VectorXf ParamState = mEnv->GetParamState().cast<float>();
        int idx = 0;
        int muscle_idx = 0;
        for (auto c : mEnv->GetTotalParams())
            if (ImGui::TreeNode(c.name.c_str()))
            {

                for (auto m : c.params)
                {

                    if (abs(m.max_v - m.min_v) < 1E-6)
                        m.max_v = m.min_v + 1E-6;
                    ImGui::SliderFloat(m.name.c_str(), &ParamState[idx++], (float)m.min_v, (float)m.max_v, "%.3f");

                    if (c.name.find("Muscle") != std::string::npos)
                    {
                        ImGui::SameLine();
                        ImGui::Checkbox(("##" + m.name).c_str(), &(mSelectedParameter[muscle_idx++]));
                    }
                }
                ImGui::TreePop();
            }
            else
                for (auto m : c.params)
                    idx++;
        mEnv->SetParamState(ParamState.cast<double>());
    }

    if (ImGui::CollapsingHeader("Joint Angle"))
    {
        for (auto jn : mEnv->GetCharacter()->GetSkeleton()->getJoints())
        {
            if (ImGui::TreeNode(jn->getName().c_str()))
            {
                Eigen::VectorXf p = jn->getPositions().cast<float>();

                for (int i = 0; i < jn->getNumDofs(); i++)
                    ImGui::SliderFloat(std::to_string(i).c_str(), &p[i], (float)(jn->getPositionLowerLimit(i) < -M_PI ? -M_PI : jn->getPositionLowerLimit(i)), (float)(jn->getPositionUpperLimit(i) > M_PI ? M_PI : jn->getPositionUpperLimit(i)), "%.3f");

                ImGui::TreePop();
                jn->setPositions(p.cast<double>());
            }
        }
    }

    if (ImGui::CollapsingHeader("Muscle"))
    {
        for (auto &m : mEnv->GetCharacter()->GetMuscles())
            ImGui::Checkbox((m->name).c_str(), &(m->selected));
    }

    if (ImGui::CollapsingHeader("Rendering Option"))
    {
        ImGui::SliderFloat("PlayRate ", &mPlayRate, 0.5, 1.0);

        ImGui::Checkbox("Draw FootStep", &mDrawFootStep);
        ImGui::Checkbox("Draw Collision", &mDrawCollision);
        ImGui::Checkbox("Draw Reference", &mDrawReference);
        ImGui::Checkbox("Draw Muscle Torque ", &mDrawMuscleTorque);

        ImGui::Checkbox("Draw Activation ", &mDrawActivation);
        ImGui::Checkbox("Draw Passive Force ", &mDrawPassiveForce);
        ImGui::Checkbox("Draw Active Force ", &mDrawActiveForce);
        ImGui::Checkbox("Draw Body Frame ", &mDrawBodyFrame);

        ImGui::Checkbox("Draw Anchor Force", &mDrawAnchorForce);

        ImGui::SliderInt("Resolution Of Drawing ", &mResolution, 1, 1000);
    }

    if (ImGui::CollapsingHeader("Gait Analysis Option"))
    {
        ImGui::SliderFloat("x_shift", &mGaitShift, 0.0, 1.0, "%.3f");
        ImGui::Checkbox("Calibrating Graph", &mCalibrateGraph);

        if (ImGui::Button("Save Current Graph Data"))
            SaveGraphData();
    }

    if (ImGui::CollapsingHeader("Environment Option"))
    {
        ImGui::Checkbox("Torque Clip", &mEnv->GetIsTorqueClip());
        ImGui::Checkbox("Torque Symmetry", &mEnv->GetIsTorqueSymMode());
        ImGui::Checkbox("Muscle Symmetry", &mEnv->GetIsMuscleSymMode());
        ImGui::Checkbox("Use Excitation ", &mEnv->GetUseExcitation());
        ImGui::Checkbox("Use EOE ", &mUseEOE);
        for (int i = 0; i <= 6; i++)
        {
            ImGui::RadioButton(("metabolic type " + std::to_string(i)).c_str(), &mEnv->GetMetabolicType(), i);
            ImGui::SameLine();
        }
    }

    if (ImGui::CollapsingHeader("All Muscle Test"))
    {
        if (ImGui::CollapsingHeader("Muscle Length"))
        {
            for (auto m : mEnv->GetCharacter()->GetMuscles())
            {
                float tmp_v = m->get_l();
                ImGui::SliderFloat(("Length_" + m->name).c_str(), &(tmp_v), (float)(m->l_min), (float)(m->l_max), "%.3f", 0);
                m->change_l(tmp_v);
            }
        }
        if (ImGui::CollapsingHeader("Muscle Force"))
        {
            for (auto m : mEnv->GetCharacter()->GetMuscles())
            {
                float tmp_v = m->get_f();
                ImGui::SliderFloat(("Force_" + m->name).c_str(), &(tmp_v), (float)(m->f_min), (float)(m->f_max), "%.3f", 0);
                m->change_f(tmp_v);
            }
        }
        mEnv->UpdateParamState();
    }

    ImGui::End();
}

void GLFWApp::DrawUIDisplay()
{

    ImGui::SetNextWindowSize(ImVec2(400, 500), ImGuiCond_Once);
    ImGui::SetNextWindowPos(ImVec2(width - 410, 10), ImGuiCond_Once);
    ImGui::Begin("Display");
    if (ImGui::CollapsingHeader("Simulation Information"))
    {
        ImGui::Text("Elapsed    Time  :  %.3f s", mEnv->GetWorld()->getTime());
        ImGui::Text("Human    Weight  :  %.3f kg", mEnv->GetCharacter()->GetSkeleton()->getMass());
        ImGui::Text("Human    Height  :  %.3f m", mEnv->GetHeight());
    }
    if (ImGui::CollapsingHeader("Metadata"))
        ImGui::Text(mEnv->GetMetadata().c_str());

    if (ImGui::CollapsingHeader("Reward"))
    {
        ImPlot::SetNextAxisLimits(0, -3, 0);
        ImPlot::SetNextAxisLimits(3, 0, 4);

        if (ImPlot::BeginPlot("Reward", "x", "r"))
        {
            if (mRewardBuffer.size() > 0)
            {
                double *x = new double[mRewardBuffer.size()]();
                int idx = 0;
                for (int i = mRewardBuffer.size() - 1; i > 0; i--)
                    x[idx++] = -i * (1.0 / mEnv->GetControlHz());

                for (auto rs : mRewardBuffer[0])
                {
                    double *v = new double[mRewardBuffer.size()]();
                    for (int i = 0; i < mRewardBuffer.size(); i++)
                        v[i] = (mRewardBuffer[i].find(rs.first)->second);

                    ImPlot::PlotLine(rs.first.c_str(), x, v, mRewardBuffer.size());
                }
            }
            ImPlot::EndPlot();
        }
    }

    if (ImGui::CollapsingHeader("State"))
    {
        Eigen::VectorXd state = mEnv->GetState();
        ImPlot::SetNextAxisLimits(0, -0.5, state.rows() + 0.5, ImGuiCond_Always);
        ImPlot::SetNextAxisLimits(3, -5, 5);

        double *x = new double[state.rows()]();
        double *y = new double[state.rows()]();
        for (int i = 0; i < state.rows(); i++)
        {
            x[i] = i;
            y[i] = state[i];
        }
        if (ImPlot::BeginPlot("state"))
        {
            ImPlot::PlotBars("", x, y, state.rows(), 1.0);
            ImPlot::EndPlot();
        }
    }

    if (ImGui::CollapsingHeader("Torque"))
    {

        if (mEnv->GetUseMuscle())
        {
            auto tp = mEnv->GetMuscleTuple(true);
            Eigen::VectorXd dt = tp.tau_des;

            Eigen::VectorXd min_tau = tp.Jtp;
            Eigen::VectorXd max_tau = tp.Jtp;

            for (int i = 0; i < tp.L.rows(); i++)
            {
                for (int j = 0; j < tp.L.cols(); j++)
                {
                    if (tp.L(i, j) > 0)
                        max_tau[i] += tp.L(i, j);
                    else
                        min_tau[i] += tp.L(i, j);
                }
            }

            ImPlot::SetNextAxisLimits(0, -0.5, dt.rows() + 0.5, ImGuiCond_Always);
            ImPlot::SetNextAxisLimits(3, -5, 5);
            double *x = new double[dt.rows()]();
            double *y = new double[dt.rows()]();
            double *y_muscle = new double[dt.rows()]();
            double *y_min = new double[dt.rows()]();
            double *y_max = new double[dt.rows()]();

            for (int i = 0; i < dt.rows(); i++)
            {
                x[i] = i;
                y[i] = dt[i];
                y_min[i] = min_tau[i];
                y_max[i] = max_tau[i];
                y_muscle[i] = mMuscleTorque[i + 6 /*root dof*/];
            }
            if (ImPlot::BeginPlot("torque"))
            {
                ImPlot::PlotBars("min", x, y_min, dt.rows(), 1.0);
                ImPlot::PlotBars("dt", x, y, dt.rows(), 1.0);
                ImPlot::PlotBars("rt", x, y_muscle, dt.rows(), 1.0);
                ImPlot::PlotBars("max", x, y_max, dt.rows(), 1.0);
                ImPlot::EndPlot();
            }
        }
        else
        {
            Eigen::VectorXd dt = mEnv->GetDesiredTorquesValue();

            ImPlot::SetNextAxisLimits(0, -0.5, dt.rows() + 0.5, ImGuiCond_Always);
            ImPlot::SetNextAxisLimits(3, -5, 5);
            double *x = new double[dt.rows()]();
            double *y = new double[dt.rows()]();

            for (int i = 0; i < dt.rows(); i++)
            {
                x[i] = i;
                y[i] = dt[i];
            }
            if (ImPlot::BeginPlot("torque"))
            {
                ImPlot::PlotBars("value", x, y, dt.rows(), 1.0);
                ImPlot::EndPlot();
            }
        }
    }

    if (ImGui::CollapsingHeader("Muscle Action"))
    {
        auto activation = mEnv->GetActivationLevels();

        ImPlot::SetNextAxisLimits(0, -0.5, activation.rows() - 0.5, ImGuiCond_Always);
        ImPlot::SetNextAxisLimits(3, 0, 1);

        double *x = new double[activation.rows()]();
        double *y1 = new double[activation.rows()]();

        for (int i = 0; i < activation.rows(); i++)
        {
            x[i] = i;
            y1[i] = activation[i];
        }
        if (ImPlot::BeginPlot("Muscle Action"))
        {
            ImPlot::PlotBars("activation", x, y1, activation.rows(), 0.8);
            ImPlot::EndPlot();
        }
    }

    if (ImGui::CollapsingHeader("Gait Analysis"))
    {

        double l_contact[1];
        double r_contact[1];

        int drawidx = 0;
        int idx = 0;

        bool isSaved = mSavedGraphData.size() > 0;

        for (auto gd : mGraphData)
        {
            GraphElem sgd;

            if (isSaved)
                sgd = mSavedGraphData[idx];

            idx++;

            l_contact[0] = 0;
            r_contact[0] = 0;

            for (int i = 1; i < mContactData.r_left.size(); i++)
            {
                if (mContactData.r_left[i].second - mContactData.r_left[i - 1].second < -1E-4)
                {
                    bool isValid = true;
                    for (int j = 0; j < 5; j++)
                        if (mContactData.r_left[(mContactData.r_left.size() + i - 1 - j) % mContactData.r_left.size()].second < 1E-4)
                            isValid = false;

                    if (isValid)
                        l_contact[0] = fmod(mContactData.r_left[i].first + mGaitShift, 1.0);
                }
                if (mContactData.r_right[i].second - mContactData.r_right[i - 1].second < -1E-4)
                {
                    bool isValid = true;
                    for (int j = 0; j < 5; j++)
                        if (mContactData.r_right[(mContactData.r_left.size() + i - 1 - j) % mContactData.r_right.size()].second < 1E-4)
                            isValid = false;

                    if (isValid)
                        r_contact[0] = fmod(mContactData.r_right[i].first + mGaitShift, 1.0);
                }
            }

            ImPlot::SetNextAxisLimits(0, 0, 1, ImGuiCond_Always);
            ImPlot::SetNextAxisLimits(3, -90, 90);

            if (ImPlot::BeginPlot(gd.name.c_str(), "cycle", "", ImVec2(-1, 250)))
            {
                int size = gd.r_left.size();
                int render_len = mEnv->GetCharacter()->GetBVH()->GetMaxTime() / (mEnv->GetPhaseRatio() < 1E-6 ? 1E-6 : mEnv->GetPhaseRatio()) * mEnv->GetControlHz();
                ;
                bool mUseRef = false;
                if (size < render_len)
                    render_len = size;

                double *x_left = new double[render_len]();
                double *x_right = new double[render_len]();
                double *x_ref = new double[render_len]();

                double *x_save = new double[render_len]();

                double *r_left = new double[render_len]();
                double *r_right = new double[render_len]();
                double *r_ref = new double[render_len]();

                double *r_save = new double[render_len]();

                std::vector<std::pair<double, double>> r_left_;
                std::vector<std::pair<double, double>> r_right_;
                std::vector<std::pair<double, double>> r_ref_;

                std::vector<std::pair<double, double>> r_save_;

                for (int i = size - 1; i >= size - render_len; i--)
                {
                    r_left_.push_back(gd.r_left[i]);

                    if (isSaved)
                        r_save_.push_back(sgd.r_left[i - size + sgd.r_left.size()]);

                    if (gd.r_right.size() > 0)
                        r_right_.push_back(gd.r_right[i]);
                    if (mUseRef)
                        r_ref_.push_back(gd.r_ref[i]);
                }

                if (r_right_.size() > 0)
                    r_right_ = Align(r_right_, mGaitShift + 0.5);

                r_contact[0] = fmod(r_contact[0] + 0.5, 1.0);

                r_left_ = Align(r_left_, mGaitShift);

                if (isSaved)
                    r_save_ = Align(r_save_, mGaitShift);

                if (mUseRef)
                    r_ref_ = Align(r_ref_, 0);

                sort(r_left_.begin(), r_left_.end(), compare);

                if (isSaved)
                    sort(r_save_.begin(), r_save_.end(), compare);

                if (r_right_.size() > 0)
                    sort(r_right_.begin(), r_right_.end(), compare);

                if (mUseRef)
                    sort(r_ref_.begin(), r_ref_.end(), compare);

                for (int i = 0; i < render_len; i++)
                {
                    x_left[i] = r_left_[i].first;
                    r_left[i] = r_left_[i].second;

                    if (isSaved)
                    {
                        x_save[i] = r_save_[i].first;
                        r_save[i] = r_save_[i].second;
                    }
                    if (r_right_.size() > 0)
                    {
                        x_right[i] = r_right_[i].first;
                        r_right[i] = r_right_[i].second;
                    }
                    if (mUseRef)
                    {
                        x_ref[i] = r_ref_[i].first;
                        r_ref[i] = r_ref_[i].second;
                    }
                }

                ImPlot::PlotVLines("r_take_off", r_contact, 1);

                ImPlot::PlotVLines("l_take_off", l_contact, 1);

                if (isSaved)
                {
                    ImPlot::SetNextFillStyle(IMPLOT_AUTO_COL, 0.3f);
                    ImPlot::PlotShaded("save", x_save, r_save, render_len, -INFINITY);
                }

                ImPlot::PlotLine("left", x_left, r_left, render_len);
                if (r_right_.size() > 0)
                {

                    ImPlot::PlotLine("right", x_right, r_right, render_len);
                }
                if (mUseRef)
                    ImPlot::PlotLine("ref", x_ref, r_ref, render_len);

                ImPlot::EndPlot();
                drawidx++;
            }
        }
    }

    static int selected = 0;
    if (ImGui::CollapsingHeader("Muscle Graph"))
    {

        auto m = mEnv->GetCharacter()->GetMuscles()[selected];

        ImPlot::SetNextAxisLimits(3, 500, 0);
        ImPlot::SetNextAxisLimits(0, 0, 2.5, ImGuiCond_Always);
        if (ImPlot::BeginPlot((m->name + "_force_graph").c_str(), "length", "force", ImVec2(-1, 250)))
        {
            std::vector<std::vector<double>> p = m->GetGraphData();

            ImPlot::PlotLine("##active", p[1].data(), p[2].data(), 250);
            ImPlot::PlotLine("##active_with_activation", p[1].data(), p[3].data(), 250);
            ImPlot::PlotLine("##passive", p[1].data(), p[4].data(), 250);

            ImPlot::PlotVLines("current", p[0].data(), 1);
            ImPlot::EndPlot();
        }
    }

    ImGui::End();
}

void GLFWApp::drawUiFrame()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    DrawUIController();

    DrawUIDisplay();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void GLFWApp::keyboardPress(int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        switch (key)
        {
        case GLFW_KEY_SPACE:
            mSimulating = !mSimulating;
            break;
        case GLFW_KEY_R:
            Reset();
            break;
        case GLFW_KEY_O:
            mDrawOBJ = !mDrawOBJ;
            break;
        case GLFW_KEY_F:
            mFocus = !mFocus;
            break;
        case GLFW_KEY_P:
        {
            mPhysics = !mPhysics;
            break;
        }
        case GLFW_KEY_V:
        {
            Eigen::VectorXd minv = mEnv->GetMinV();
            Eigen::VectorXd maxv = mEnv->GetMaxV();
            Eigen::VectorXd v = minv;
            for (int i = 0; i < v.rows(); i++)
                v[i] = minv[i] + (dart::math::Random::uniform(0.0, 1.0)) * (maxv[i] - minv[i]);
            mEnv->SetParamState(v);

            break;
        }
        break;
        case GLFW_KEY_S:
            update();
            break;
        case GLFW_KEY_A:
            mIncludeAction++;
            mIncludeAction %= 3;
            break;
        case GLFW_KEY_Z:
        {
            Reset();
            Eigen::VectorXd p = mEnv->GetCharacter()->GetSkeleton()->getPositions();
            p.setZero();

            mEnv->GetCharacter()->GetSkeleton()->setPositions(p);
            mEnv->GetCharacter()->GetSkeleton()->setVelocities(mEnv->GetCharacter()->GetSkeleton()->getVelocities().setZero());
        }
        break;
        case GLFW_KEY_X:
        {
            Reset();
            Eigen::VectorXd p = mEnv->GetCharacter()->GetSkeleton()->getPositions();
            p.setZero();
            mEnv->GetCharacter()->GetSkeleton()->setPositions(p);
            mEnv->GetCharacter()->GetSkeleton()->setVelocities(mEnv->GetCharacter()->GetSkeleton()->getVelocities().setZero());
        }
        break;
        case GLFW_KEY_KP_0:
        case GLFW_KEY_T:
        {
            Eigen::VectorXd p = mEnv->GetCharacter()->GetSkeleton()->getPositions();
            mEnv->GetCharacter()->GetSkeleton()->setPositions(mEnv->GetCharacter()->GetMirrorPosition(p));
            break;
        }
        case GLFW_KEY_7:
        case GLFW_KEY_KP_7:
            mCameraMoving = 1;
            break;
        case GLFW_KEY_5:
        case GLFW_KEY_KP_5:
            mCameraMoving = 0;
            mTrackball.setQuaternion(Eigen::Quaterniond::Identity());
            break;
        case GLFW_KEY_9:
        case GLFW_KEY_KP_9:
            mCameraMoving = -1;
            break;
        case GLFW_KEY_4:
        case GLFW_KEY_KP_4:
        {
            mCameraMoving = 0;
            Eigen::Quaterniond r = Eigen::Quaterniond(Eigen::AngleAxisd(0.01 * M_PI, Eigen::Vector3d::UnitY())) * mTrackball.getCurrQuat();
            mTrackball.setQuaternion(r);
            break;
        }
        case GLFW_KEY_6:
        case GLFW_KEY_KP_6:
        {
            mCameraMoving = 0;
            Eigen::Quaterniond r = Eigen::Quaterniond(Eigen::AngleAxisd(-0.01 * M_PI, Eigen::Vector3d::UnitY())) * mTrackball.getCurrQuat();
            mTrackball.setQuaternion(r);
            break;
        }
        case GLFW_KEY_8:
        case GLFW_KEY_KP_8:
        {
            mCameraMoving = 0;
            Eigen::Quaterniond r = Eigen::Quaterniond(Eigen::AngleAxisd(0.01 * M_PI, Eigen::Vector3d::UnitX())) * mTrackball.getCurrQuat();
            mTrackball.setQuaternion(r);
            break;
        }
        case GLFW_KEY_2:
        case GLFW_KEY_KP_2:
        {
            mCameraMoving = 0;
            Eigen::Quaterniond r = Eigen::Quaterniond(Eigen::AngleAxisd(-0.01 * M_PI, Eigen::Vector3d::UnitX())) * mTrackball.getCurrQuat();
            mTrackball.setQuaternion(r);
            break;
        }
        case GLFW_KEY_N:
        {
            mEnv->SetParamState(mEnv->GetMinV());
            break;
        }
        case GLFW_KEY_M:
        {
            mEnv->SetParamState(mEnv->GetMaxV());
            break;
        }
        case GLFW_KEY_I:
        {
            mPhysics = !mPhysics;
            mDrawNodeCOM = !mDrawNodeCOM;
            mDrawJoint = !mDrawJoint;
            break;
        }

        default:
            break;
        }
    }
}

void GLFWApp::mouseMove(double xpos, double ypos)
{
    double deltaX = xpos - mMouseX;
    double deltaY = ypos - mMouseY;
    mMouseX = xpos;
    mMouseY = ypos;
    if (mRotate)
    {
        if (deltaX != 0 || deltaY != 0)
        {
            mTrackball.updateBall(xpos, height - ypos);
        }
    }
    if (mTranslate)
    {
        Eigen::Matrix3d rot;
        rot = mTrackball.getRotationMatrix();
        mTrans += (1 / mZoom) * rot.transpose() * Eigen::Vector3d(deltaX, -deltaY, 0.0);
    }
    if (mZooming)
    {
        mZoom = std::max(0.01, mZoom + deltaY * 0.01);
    }
}

void GLFWApp::mousePress(int button, int action, int mods)
{
    mMouseDown = true;
    if (action == GLFW_PRESS)
    {
        if (button == GLFW_MOUSE_BUTTON_LEFT)
        {
            mRotate = true;
            mTrackball.startBall(mMouseX, height - mMouseY);
        }
        else if (button == GLFW_MOUSE_BUTTON_RIGHT)
        {
            mTranslate = true;
        }
    }
    else if (action == GLFW_RELEASE)
    {
        if (button == GLFW_MOUSE_BUTTON_LEFT)
        {
            mRotate = false;
        }
        else if (button == GLFW_MOUSE_BUTTON_RIGHT)
        {
            mTranslate = false;
        }
    }
}

void GLFWApp::mouseScroll(double xoffset, double yoffset)
{
    mZoom += yoffset * 0.01;
}

void GLFWApp::DrawJoint()
{
    for (auto j : mEnv->GetCharacter()->GetSkeleton()->getJoints())
    {
        Eigen::Isometry3d t = j->getChildBodyNode()->getTransform() * j->getTransformFromChildBodyNode();
        Eigen::Vector3d p;
        glPushMatrix();
        if (j->getNumDofs() == 3)
        {
            p = t * Eigen::Vector3d::Zero();
            glTranslated(p[0], p[1], p[2]);
            glColor4f(0.5, 0, 0, 0.5);
            GUI::DrawSphere(0.02, false);
        }
        else if (j->getNumDofs() == 1)
        {
            p = t * Eigen::Vector3d(0, 0, 0);
            glTranslatef(p[0], p[1], p[2]);
            glRotated(90, 0, 1, 0);
            glColor4f(0.5, 0, 0, 0.5);
            GUI::DrawCylinder(0.01, 0.04);
        }
        glPopMatrix();
    }
}

void GLFWApp::DrawNodeCOM()
{
    for (auto bn : mEnv->GetCharacter()->GetSkeleton()->getBodyNodes())
    {
        Eigen::Vector3d p = bn->getCOM();
        glPushMatrix();
        glTranslated(p[0], p[1], p[2]);
        GUI::DrawSphere(0.005);
        glPopMatrix();
    }
}

void GLFWApp::draw()
{
    initGL();
    initLights();
    initFog();
    SetFocusing();

    /* Preprocessing */
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glViewport(0, 0, width, height);
    gluPerspective(mPersp, width / height, 0.1, 10.0);
    gluLookAt(mEye[0], mEye[1], mEye[2], 0.0, 0.0, -1.0, mUp[0], mUp[1], mUp[2]);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    mTrackball.setCenter(Eigen::Vector2d(width * 0.5, height * 0.5));
    mTrackball.setRadius(std::min(width, height) / 2.5);
    mTrackball.applyGLRotation();

    {
        glEnable(GL_DEPTH_TEST);
        glDisable(GL_TEXTURE_2D);
        glDisable(GL_LIGHTING);
        glLineWidth(2.0);
        if (mRotate || mTranslate || mZooming)
        {
            glColor3f(1.0f, 0.0f, 0.0f);
            glBegin(GL_LINES);
            glVertex3f(-0.1f, 0.0f, -0.0f);
            glVertex3f(0.15f, 0.0f, -0.0f);
            glEnd();

            glColor3f(0.0f, 1.0f, 0.0f);
            glBegin(GL_LINES);
            glVertex3f(0.0f, -0.1f, 0.0f);
            glVertex3f(0.0f, 0.15f, 0.0f);
            glEnd();

            glColor3f(0.0f, 0.0f, 1.0f);
            glBegin(GL_LINES);
            glVertex3f(0.0f, 0.0f, -0.1f);
            glVertex3f(0.0f, 0.0f, 0.15f);
            glEnd();
        }
    }

    glScalef(mZoom, mZoom, mZoom);
    glTranslatef(mTrans[0] * 0.001, mTrans[1] * 0.001, mTrans[2] * 0.001);

    GLfloat matrix[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, matrix);
    Eigen::Matrix3d A;
    Eigen::Vector3d b;
    A << matrix[0], matrix[4], matrix[8],
        matrix[1], matrix[5], matrix[9],
        matrix[2], matrix[6], matrix[10];
    b << matrix[12], matrix[13], matrix[14];
    mViewMatrix.linear() = A;
    mViewMatrix.translation() = b;

    auto ground = mEnv->GetGround();
    float y = ground->getBodyNode(0)->getCOM()[1] +
              dynamic_cast<const BoxShape *>(ground->getBodyNode(
                                                       0)
                                                 ->getShapeNodesWith<dart::dynamics::VisualAspect>()[0]
                                                 ->getShape()
                                                 .get())
                      ->getSize()[1] *
                  0.5;
    DrawGround(y);

    if (mDrawMuscleTorque)
        DrawMuscleTorque();

    if (mDrawCollision)
        DrawCollision();

    if (mDrawJoint)
        DrawJoint();

    if (mDrawNodeCOM)
        DrawNodeCOM();

    if (mPhysics)
    {
        if (mDrawOBJ || !mIsSelectedMode)
            DrawSkeleton(mEnv->GetCharacter()->GetSkeleton());

        if (mEnv->GetUseMuscle())
            DrawMuscles(mEnv->GetCharacter()->GetMuscles());
    }

    DrawSkeleton(mEnv->GetBVHSkeleton());

    if (mDrawReference)
    {
        Eigen::VectorXd p = mEnv->GetTargetPositions();
        if (mIncludeAction == 1)
            p.tail(p.rows() - 6) += mDisplacement;
        else if (mIncludeAction == 2)
        {
            if (mEnv->GetActionType() == 0)
            {
                p.head(6) = mEnv->GetCharacter()->GetSkeleton()->getPositions().head(6) + 1.0 / mEnv->GetSimulationHz() * mEnv->GetCharacter()->GetSkeleton()->getVelocities().head(6);

                p.tail(p.rows() - 6) += mEnv->GetAction();
            }
            else if (mEnv->GetActionType() == 1)
            {
                p.tail(p.rows() - 6).setZero();
                if (mEnv->IsArmExist())
                {
                    p[mEnv->GetCharacter()->GetSkeleton()->getJoint("ArmL")->getIndexInSkeleton(2)] = -M_PI / 2;
                    p[mEnv->GetCharacter()->GetSkeleton()->getJoint("ArmR")->getIndexInSkeleton(2)] = M_PI / 2;
                }
                p.tail(p.rows() - 6) += mEnv->GetAction();
            }
            else if (mEnv->GetActionType() == 2)
            {
                p = mEnv->GetCharacter()->GetSkeleton()->getPositions();
                p.tail(p.rows() - 6) += mEnv->GetAction();
            }
        }
        mEnv->GetReferenceSkeleton()->setPositions(p);

        DrawSkeleton(mEnv->GetReferenceSkeleton());
    }

    if (mDrawFootStep)
        DrawFootStep();

    Eigen::Vector3d head_com = mEnv->GetCharacter()->GetSkeleton()->getBodyNode("Head")->getCOM();
    Eigen::Vector3d head_com_vel = mEnv->GetCharacter()->GetSkeleton()->getBodyNode("Head")->getLinearVelocity();

    glEnable(GL_COLOR_MATERIAL);

    for (auto m : mEnv->GetCharacter()->GetMuscles())
    {
        if (m->selected)
        {
            for (auto bn : m->GetRelatedBodyNodes())
                DrawSingleBodyNode(bn);

            for (auto ac : m->GetAnchors())
            {
                Eigen::Vector3d p = ac->GetPoint();
                glPushMatrix();
                glTranslatef(p[0], p[1], p[2]);
                glColor4d(0.8, 0.8, 0.8, 0.8);
                GUI::DrawSphere(0.01);
                glPopMatrix();
            }
        }
    }

    if (mDrawAnchorForce)
        DrawAnchorForce();

    if (mDrawBodyFrame)
        for (auto bn : mEnv->GetCharacter()->GetSkeleton()->getBodyNodes())
        {
            auto tr = bn->getTransform();
            Eigen::Vector3d origin = tr * Eigen::Vector3d::Zero();
            Eigen::Vector3d x = tr * (0.1 * Eigen::Vector3d::UnitX());
            Eigen::Vector3d y = tr * (0.1 * Eigen::Vector3d::UnitY());
            Eigen::Vector3d z = tr * (0.1 * Eigen::Vector3d::UnitZ());

            glColor3f(1, 0, 0);
            glBegin(GL_LINES);
            glVertex3f(origin[0], origin[1], origin[2]);
            glVertex3f(x[0], x[1], x[2]);
            glEnd();

            glColor3f(0, 1, 0);
            glBegin(GL_LINES);
            glVertex3f(origin[0], origin[1], origin[2]);
            glVertex3f(y[0], y[1], y[2]);
            glEnd();

            glColor3f(0, 0, 1);
            glBegin(GL_LINES);
            glVertex3f(origin[0], origin[1], origin[2]);
            glVertex3f(z[0], z[1], z[2]);
            glEnd();
        }

    DrawPhase(mEnv->GetPhase(), mEnv->GetGlobalPhase());
}

void GLFWApp::UpdateMuscleTorque()
{
    if (mEnv->GetUseMuscle())
    {
        auto mt = mEnv->GetMuscleTuple();
        Eigen::VectorXd tmp = mt.L * mEnv->GetActivationLevels() + mt.b;
        mMuscleTorque = Eigen::VectorXd::Zero(tmp.rows() + 6);
        mMuscleTorque.tail(tmp.rows()) = tmp;
    }
    else
        mMuscleTorque = mEnv->GetDesiredTorquesValue();
}

void GLFWApp::initGL()
{
    glClearColor(1.0, 1.0, 1.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);
    glShadeModel(GL_SMOOTH);
    glPolygonMode(GL_FRONT, GL_FILL);
}

void GLFWApp::initFog()
{
    glEnable(GL_FOG);
    GLfloat fogColor[] = {1, 1, 1, 1};
    glFogfv(GL_FOG_COLOR, fogColor);
    glFogi(GL_FOG_MODE, GL_LINEAR);
    glFogf(GL_FOG_START, 0.0);
    glFogf(GL_FOG_END, 8.0);
}

void GLFWApp::initLights()
{
    static float ambient[] = {0.2, 0.2, 0.2, 1.0};
    static float diffuse[] = {0.6, 0.6, 0.6, 1.0};
    static float front_mat_shininess[] = {60.0};
    static float front_mat_specular[] = {0.2, 0.2, 0.2, 1.0};
    static float front_mat_diffuse[] = {0.5, 0.28, 0.38, 1.0};
    static float lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
    static float lmodel_twoside[] = {GL_FALSE};
    GLfloat position[] = {1.0, 0.0, 0.0, 0.0};
    GLfloat position1[] = {-1.0, 0.0, 0.0, 0.0};
    glEnable(GL_LIGHT0);
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse);
    glLightfv(GL_LIGHT0, GL_POSITION, position);
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
    glLightModelfv(GL_LIGHT_MODEL_TWO_SIDE, lmodel_twoside);
    glEnable(GL_LIGHT1);
    glLightfv(GL_LIGHT1, GL_DIFFUSE, diffuse);
    glLightfv(GL_LIGHT1, GL_POSITION, position1);
    glEnable(GL_LIGHTING);
    glEnable(GL_COLOR_MATERIAL);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, front_mat_shininess);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, front_mat_specular);
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, front_mat_diffuse);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glDisable(GL_CULL_FACE);
    glEnable(GL_NORMALIZE);
}

void GLFWApp::update()
{
    int num = mEnv->GetSimulationHz() / mEnv->GetControlHz();
    num /= (mFramerate / mEnv->GetControlHz());
    int generalstate_num = mEnv->GetStateDiffNum();

    if (mSimCount % (mFramerate / mEnv->GetControlHz()) == 0)
    {
        mRefs.clear();
        mWeights.clear();
        mAxis.clear();
        mActions.clear();

        mProjState.clear();

        mDisplacement = Eigen::VectorXd::Zero(mEnv->GetNumAction());

        Eigen::VectorXd action = Eigen::VectorXd::Zero(mEnv->GetNumAction());

        int idx = 0;

        for (auto mNN : mNNs)
        {
            Eigen::VectorXd ProjState = mEnv->GetProjState(mNN.minv, mNN.maxv);
            mProjState.push_back(ProjState);
            Eigen::VectorXd action = (mNN.sim.attr("get_action"))(ProjState).cast<Eigen::VectorXd>();
            if (mNN.cascadingType > 0)
            {
                mAxis.push_back(dart::math::clip(0.25 + 0.05 * action.tail(1)[0], 0.05, 1.0));

                mActions.push_back(action.head(action.rows() - 1));
            }
            else
            {
                mAxis.push_back(-1.0);
                mActions.push_back(action);
            }
            idx++;
        }

        for (int i = 0; i < mNNs.size(); i++)
        {
            auto l = mCascadingMap[i];
            double min_norm = 999999.0;
            double min_w = 1.0;
            int min_idx = 0;
            for (auto l_idx : l)
            {
                Eigen::VectorXd state_diff = mProjState[l_idx].head(generalstate_num) - mProjState[i].head(generalstate_num);

                state_diff.segment(mEnv->getMuscleStateStartIdx() + 24, 26) *= 0.4;
                state_diff.segment(mEnv->getMuscleStateStartIdx() + 74, 26) *= 0.4;
                state_diff.segment(mEnv->getMuscleStateStartIdx() + 124, 26) *= 0.4;
                if (state_diff.norm() < min_norm)
                {

                    min_norm = state_diff.norm();
                    min_w = mEnv->CalculateWeight(state_diff, mAxis[i]);
                    min_idx = l_idx;
                }
            }
            if (min_w <= 1E-6)
                min_w = 0.0;
            mWeights.push_back(min_w);
            action += mWeights[i] * mActions[i] * mUseWeights[i * 2];
            if (mEnv->UseDisplacement())
                mDisplacement += mRefs.back();
        }

        if (mNNs.size() > 0)
            mValues.push_back((mNNs.back().sim.attr("get_value"))(mProjState.back()).cast<double>());

        mEnv->GetReward();
        mEnv->SetAction(action);
        mRewardBuffer.push_back(mEnv->GetRewardMap());

        AddGraphData();

        mEnv->UpdateHeadInfo();
    }

    if (mEnv->GetUseMuscle() && mNNs.size() > 0)
    {
        int inference_per_sim = mEnv->GetInferencePerSim();
        for (int i = 0; i < num; i += inference_per_sim)
        {
            MuscleTuple &tp = mEnv->GetMuscleTuple();
            Eigen::VectorXd mt = tp.JtA;
            Eigen::VectorXd dt = (tp.tau_des - tp.Jtp);
            std::vector<Eigen::VectorXd> prev_unnormalized_activations;
            for (int j = 0; j < mNNs.size() - 1; j++)
            {
                if (mWeights[j] > 1E-3)
                {
                    if (mNNs[j].isLegacy)
                    {
                        prev_unnormalized_activations.push_back((mWeights[j] * mUseWeights[j * 2 + 1]) * mNNs[j].muscle.attr("forward_without_filter_render")(mt, dt).cast<Eigen::VectorXd>());
                    }
                    else
                    {
                        Eigen::VectorXd prev_out = Eigen::VectorXd::Zero(mEnv->GetCharacter()->GetMuscles().size());
                        for (auto l : mCascadingMap[j])
                            prev_out += prev_unnormalized_activations[l];
                        prev_unnormalized_activations.push_back((mWeights[j] * mUseWeights[j * 2 + 1]) * mNNs[j].muscle.attr("forward_with_prev_out_without_filter_render")(mt, dt, prev_out, mWeights[j] * mUseWeights[j * 2 + 1]).cast<Eigen::VectorXd>());
                    }
                }
                else
                {
                    prev_unnormalized_activations.push_back(Eigen::VectorXd::Zero(mEnv->GetCharacter()->GetMuscles().size()));
                }
            }
            Eigen::VectorXd prev_out = Eigen::VectorXd::Zero(mEnv->GetCharacter()->GetMuscles().size());
            for (auto l : mCascadingMap.back())
                prev_out += prev_unnormalized_activations[l];
            Eigen::VectorXd activation = (mNNs.back().muscle).attr("forward_with_prev_out_render")(mt, dt, prev_out, mWeights.back() * mUseWeights.back()).cast<Eigen::VectorXd>();
            mEnv->SetMuscleAction(activation);
            for (int j = 0; j < inference_per_sim; j++)
                mEnv->Step();
        }
    }
    else
        for (int i = 0; i < num; i++)
            mEnv->Step();

    mCameraPos[2] += -mEnv->GetTargetVelocity() * (1.0 / mFramerate);

    UpdateMuscleTorque();

    mSimCount++;

    for (int i = 0; i < mNNs.size(); i++)
    {
        if (mNNs[i].cascadingType == 2)
        {
            for (int j = 0; j < i; j++)
                mWeights[j] *= (1.0 - mWeights[i]);
        }
    }
}

void GLFWApp::Reset()
{
    mEnv->Reset();
    mCameraPos = -mEnv->GetWorld()->getSkeleton("Human")->getRootBodyNode()->getCOM();
    mCameraPos[1] = -1;

    mSimCount = 0;

    mRewardBuffer.clear();
    mValues.clear();

    for (auto &gd : mGraphData)
    {
        gd.r_left.clear();
        gd.r_right.clear();
        gd.r_ref.clear();
    }
    mMuscleTorque.setZero();
    AddGraphData();
}

void GLFWApp::SetFocusing()
{
    if (mFocus)
    {
        mTrans = mCameraPos;
        mTrans *= 1000;
    }

    Eigen::Quaterniond origin_r = mTrackball.getCurrQuat();
    if (mCameraMoving == 1 && Eigen::AngleAxisd(origin_r).angle() < 0.5 * M_PI)
    {
        Eigen::Quaterniond r = Eigen::Quaterniond(Eigen::AngleAxisd(mCameraMoving * 0.01 * M_PI, Eigen::Vector3d::UnitY())) * origin_r;
        mTrackball.setQuaternion(r);
    }
    else if (mCameraMoving == -1 && Eigen::AngleAxisd(origin_r).axis()[1] > 0)
    {
        Eigen::Quaterniond r = Eigen::Quaterniond(Eigen::AngleAxisd(mCameraMoving * 0.01 * M_PI, Eigen::Vector3d::UnitY())) * origin_r;
        mTrackball.setQuaternion(r);
    }
}

void GLFWApp::DrawEntity(const Entity *entity)
{
    if (!entity)
        return;
    const auto &bn = dynamic_cast<const BodyNode *>(entity);
    if (bn)
    {
        DrawBodyNode(bn);
        return;
    }
    const auto &sf = dynamic_cast<const ShapeFrame *>(entity);
    if (sf)
    {
        DrawShapeFrame(sf);
        return;
    }
}

void GLFWApp::DrawSingleBodyNode(const BodyNode *bn)
{
    if (!bn)
        return;

    glPushMatrix();

    glMultMatrixd(bn->getTransform().data());

    auto sns = bn->getShapeNodesWith<VisualAspect>();
    for (const auto &sn : sns)
    {

        if (!sn)
            return;

        const auto &va = sn->getVisualAspect();

        if (!va || va->isHidden())
            return;

        glPushMatrix();
        Eigen::Affine3d tmp = sn->getRelativeTransform();
        glMultMatrixd(tmp.data());
        Eigen::Vector4d c = va->getRGBA();
        c[3] *= 0.5;
        DrawShape(sn->getShape().get(), c);

        glPopMatrix();
    }
    glPopMatrix();
}

void GLFWApp::DrawBodyNode(const BodyNode *bn)
{
    if (!bn)
        return;

    glPushMatrix();
    Eigen::Affine3d tmp = bn->getRelativeTransform();
    glMultMatrixd(tmp.data());

    auto sns = bn->getShapeNodesWith<VisualAspect>();
    for (const auto &sn : sns)
        DrawShapeFrame(sn);

    for (const auto &et : bn->getChildEntities())
        DrawEntity(et);

    glPopMatrix();
}

void GLFWApp::DrawSkeleton(const SkeletonPtr &skel)
{
    DrawBodyNode(skel->getRootBodyNode());
}

void GLFWApp::DrawShapeFrame(const ShapeFrame *sf)
{
    if (!sf)
        return;

    const auto &va = sf->getVisualAspect();

    if (!va || va->isHidden())
        return;

    glPushMatrix();
    Eigen::Affine3d tmp = sf->getRelativeTransform();
    glMultMatrixd(tmp.data());

    DrawShape(sf->getShape().get(), va->getRGBA());

    glPopMatrix();
}

void GLFWApp::DrawShape(const Shape *shape, const Eigen::Vector4d &color)
{
    if (!shape)
        return;

    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_DEPTH_TEST);
    glColor4d(color[0], color[1], color[2], color[3]);
    if (mDrawOBJ == false)
    {
        glColor4dv(color.data());
        if (shape->is<SphereShape>())
        {
            const auto *sphere = dynamic_cast<const SphereShape *>(shape);
            GUI::DrawSphere(sphere->getRadius());
        }
        else if (shape->is<BoxShape>())
        {
            const auto *box = dynamic_cast<const BoxShape *>(shape);
            GUI::DrawCube(box->getSize());
        }
        else if (shape->is<CapsuleShape>())
        {
            const auto *capsule = dynamic_cast<const CapsuleShape *>(shape);
            GUI::DrawCapsule(capsule->getRadius(), capsule->getHeight());
        }
        else if (shape->is<CylinderShape>())
        {
            const auto *cylinder = dynamic_cast<const CylinderShape *>(shape);
            GUI::DrawCylinder(cylinder->getRadius(), cylinder->getHeight());
        }
    }
    else
    {
        if (shape->is<MeshShape>())
        {
            const auto &mesh = dynamic_cast<const MeshShape *>(shape);
            float y = mEnv->GetGround()->getBodyNode(0)->getTransform().translation()[1] + dynamic_cast<const BoxShape *>(mEnv->GetGround()->getBodyNode(0)->getShapeNodesWith<dart::dynamics::VisualAspect>()[0]->getShape().get())->getSize()[1] * 0.5;
            mShapeRenderer.renderMesh(mesh, false, y, Eigen::Vector4d(0.6, 0.6, 0.6, 0.8));
            // DrawShadow(mesh->getScale(), mesh->getMesh(), y);
        }
    }

    glDisable(GL_COLOR_MATERIAL);
}
void GLFWApp::DrawMuscles(const std::vector<Muscle *> &muscles)
{
    int count = 0;
    glEnable(GL_LIGHTING);
    glEnable(GL_DEPTH_TEST);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_COLOR_MATERIAL);

    for (auto muscle : muscles)
    {
        muscle->Update();

        if (!mIsSelectedMode || muscle->selected)
        {
            double a = muscle->activation;
            Eigen::Vector4d color;
            if (mDrawActivation)
                color = Eigen::Vector4d(1.0, 0.2, 0.2, a * 0.9 + 0.1);
            else if (mDrawPassiveForce)
            {
                double p_f = muscle->Getf_p() * (1.0 / mResolution);
                color = Eigen::Vector4d(0.2, 0.2, 1.0, p_f);
                if (p_f > 1 - (1E-4))
                    std::cout << "[DEBUG] Muscle Name : " << muscle->name << std::endl;
            }
            else if (mDrawActiveForce)
            {
                double a_f = muscle->Getf_A() * a * (1.0 / mResolution);
                color = Eigen::Vector4d(0.2, 1.0, 0.2, a_f);
            }
            else
                color = Eigen::Vector4d(0.4 + (2.0 * a), 0.4, 0.4, 0.1 + 0.9 * a);
            glColor4dv(color.data());
            mShapeRenderer.renderMuscle(muscle);
        }
    }
    glEnable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);
}

void GLFWApp::DrawGround(double y)
{
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDisable(GL_LIGHTING);
    double width = 0.005;
    int count = 0;
    glBegin(GL_QUADS);
    for (double x = -100.0; x < 100.01; x += 1.0)
    {
        for (double z = -100.0; z < 100.01; z += 1.0)
        {
            if (count % 2 == 0)
                glColor3f(216.0 / 255.0, 211.0 / 255.0, 204.0 / 255.0);
            else
                glColor3f(216.0 / 255.0 - 0.1, 211.0 / 255.0 - 0.1, 204.0 / 255.0 - 0.1);
            count++;
            glVertex3f(x, y, z);
            glVertex3f(x + 1.0, y, z);
            glVertex3f(x + 1.0, y, z + 1.0);
            glVertex3f(x, y, z + 1.0);
        }
    }
    glEnd();
    glEnable(GL_LIGHTING);
}

void GLFWApp::DrawCollision()
{
    const auto result = mEnv->GetWorld()->getConstraintSolver()->getLastCollisionResult();
    for (const auto &contact : result.getContacts())
    {
        Eigen::Vector3d v = contact.point;
        Eigen::Vector3d f = contact.force / 1000.0;
        glLineWidth(2.0);
        glColor3f(0.8, 0.8, 0.2);
        glBegin(GL_LINES);
        glVertex3f(v[0], v[1], v[2]);
        glVertex3f(v[0] + f[0], v[1] + f[1], v[2] + f[2]);
        glEnd();
        glColor3f(0.8, 0.8, 0.2);
        glPushMatrix();
        glTranslated(v[0], v[1], v[2]);
        GUI::DrawSphere(0.01);
        glPopMatrix();
    }
}

void GLFWApp::DrawPhase(double phase, double global_phase)
{

    glDisable(GL_LIGHTING);
    glDisable(GL_TEXTURE_2D);

    glPushMatrix();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glViewport(0, 0, width, height);
    gluOrtho2D(0.0, (GLdouble)width, 0.0, (GLdouble)height);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glLineWidth(1.0);
    glColor3f(0.0f, 0.0f, 0.0f);
    glTranslatef(height * 0.05, height * 0.05, 0.0f);
    glBegin(GL_LINE_LOOP);
    for (int i = 0; i < 360; i++)
    {
        double theta = i / 180.0 * M_PI;
        double x = height * 0.04 * cos(theta);
        double y = height * 0.04 * sin(theta);
        glVertex2d(x, y);
    }
    glEnd();

    glColor3f(1, 0, 0);
    glBegin(GL_LINES);
    glVertex2d(0, 0);
    glVertex2d(height * 0.04 * sin(global_phase * 2 * M_PI), height * 0.04 * cos(global_phase * M_PI * 2));
    glEnd();

    glColor3f(0, 0, 0);
    glLineWidth(2.0);

    glBegin(GL_LINES);
    glVertex2d(0, 0);
    glVertex2d(height * 0.04 * sin(phase * 2 * M_PI), height * 0.04 * cos(phase * M_PI * 2));
    glEnd();

    glPushMatrix();
    glPointSize(2.0);
    glBegin(GL_POINTS);
    glVertex2d(height * 0.04 * sin(phase * 2 * M_PI), height * 0.04 * cos(phase * M_PI * 2));
    glEnd();
    glPopMatrix();

    glPopMatrix();
}

void GLFWApp::DrawFootStep()
{
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_DEPTH_TEST);

    Eigen::Vector3d current_target = mEnv->GetCurrentTargetFoot();
    glColor4d(0.2, 0.2, 0.8, 0.5);
    glPushMatrix();
    glTranslated(0, current_target[1], current_target[2]);
    GUI::DrawCube(Eigen::Vector3d(1.0, 0.15, 0.15));
    glPopMatrix();

    glColor4d(0.2, 0.8, 0.2, 0.5);
    glPushMatrix();
    glTranslated(0, current_target[1], current_target[2] + mEnv->GetStepDisplacement());
    GUI::DrawCube(Eigen::Vector3d(1.0, 0.15, 0.15));
    glPopMatrix();

    Eigen::Vector3d next_target = mEnv->GetNextTargetFoot();
    glColor4d(0.8, 0.2, 0.2, 0.5);
    glPushMatrix();
    glTranslated(0, next_target[1], next_target[2]);
    GUI::DrawCube(Eigen::Vector3d(1.0, 0.15, 0.15));
    glPopMatrix();
}

void GLFWApp::SetMuscleColor()
{
    int idx = 0;
    auto m_l = mEnv->getMuscleLengthParams();
    auto m_f = mEnv->getMuscleForceParams();

    for (auto m : mEnv->GetCharacter()->GetMuscles())
        m->selected = false;

    for (int i = 0; i < mEnv->GetMuscleLengthParamNum() + mEnv->GetMuscleForceParamNum(); i++)
    {
        if (i < mEnv->GetMuscleLengthParamNum())
            for (auto m_e : m_l[i].muscle)
                m_e->selected = m_e->selected || mSelectedParameter[i];
        else if (i < mEnv->GetMuscleLengthParamNum() + mEnv->GetMuscleForceParamNum())
            for (auto m_e : m_f[i - mEnv->GetMuscleLengthParamNum()].muscle)
                m_e->selected = m_e->selected || mSelectedParameter[i];
    }
    mIsSelectedMode = false;
    for (auto m : mEnv->GetCharacter()->GetMuscles())
        mIsSelectedMode = mIsSelectedMode || m->selected;
}

void GLFWApp::CreateGraphData()
{
    mGraphData.clear();

    {
        GraphElem hip_flexor;
        hip_flexor.name = "Hip_Flexion";
        hip_flexor.r_left.clear();
        hip_flexor.r_right.clear();
        hip_flexor.r_ref.clear();
        mGraphData.push_back(hip_flexor);

        GraphElem knee_flexor;
        knee_flexor.name = "Knee_Flexion";
        knee_flexor.r_left.clear();
        knee_flexor.r_right.clear();
        knee_flexor.r_ref.clear();
        mGraphData.push_back(knee_flexor);

        GraphElem ankle_flexor;
        ankle_flexor.name = "Ankle_Flexion";
        ankle_flexor.r_left.clear();
        ankle_flexor.r_right.clear();
        ankle_flexor.r_ref.clear();
        mGraphData.push_back(ankle_flexor);

        GraphElem pelvic_x;
        pelvic_x.name = "Pelvic_Tilt";
        pelvic_x.r_left.clear();
        pelvic_x.r_right.clear();
        pelvic_x.r_ref.clear();
        mGraphData.push_back(pelvic_x);

        GraphElem pelvic_y;
        pelvic_y.name = "Pelvic_Rotation";
        pelvic_y.r_left.clear();
        pelvic_y.r_right.clear();
        pelvic_y.r_ref.clear();
        mGraphData.push_back(pelvic_y);

        GraphElem pelvic_z;
        pelvic_z.name = "Pelvic_Obliquity";
        pelvic_z.r_left.clear();
        pelvic_z.r_right.clear();
        pelvic_z.r_ref.clear();
        mGraphData.push_back(pelvic_z);
    }

    {
        GraphElem contact;
        contact.name = "Contact";
        contact.r_left.clear();
        contact.r_right.clear();
        contact.r_ref.clear();
        mGraphData.push_back(contact);
    }

    mContactData.name = "Contact";
    mContactData.r_left.clear();
    mContactData.r_right.clear();
    mContactData.r_ref.clear();
}
double GLFWApp::GetProjectedAngle(Eigen::Vector3d pos, int axis)
{
    if (axis == 0)
    {
        Eigen::Vector3d res = BallJoint::convertToRotation(pos) * Eigen::Vector3d::UnitY();
        return atan2(res[2], res[1]);
    }
    else if (axis == 1)
    {
        Eigen::Vector3d res = BallJoint::convertToRotation(pos) * Eigen::Vector3d::UnitZ();
        return atan2(res[0], res[2]);
    }
    else if (axis == 2)
    {
        Eigen::Vector3d res = BallJoint::convertToRotation(pos) * Eigen::Vector3d::UnitX();
        return atan2(res[1], res[0]);
    }
    else
    {
        exit(-1);
        return 0;
    }
}

void GLFWApp::DrawAnchorForce()
{
    for (auto m : mEnv->GetCharacter()->GetMuscles())
    {
        double f = m->GetForce();

        std::vector<Eigen::Vector3d> ps;
        std::vector<Eigen::Vector3d> ps_f;

        for (auto ac : m->GetAnchors())
        {
            ps.push_back(ac->GetPoint());
            ps_f.push_back(Eigen::Vector3d::Zero());
        }

        for (int i = 0; i < m->GetAnchors().size(); i++)
        {
            if (i == 0)
                ps_f[i] = (ps[i + 1] - ps[i]).normalized() * f;
            else if (i == m->GetAnchors().size() - 1)
                ps_f[i] = (ps[i - 1] - ps[i]).normalized() * f;
            else
            {
                ps_f[i] = (ps[i + 1] - ps[i]).normalized() * f;
                ps_f[i] += (ps[i - 1] - ps[i]).normalized() * f;
            }
        }

        for (int i = 0; i < ps.size(); i++)
        {
            GUI::DrawArrow3D(ps[i], ps_f[i].normalized(), ps_f[i].norm() * 0.001, 0.001, Eigen::Vector3d(0.4, 0, 0), 0.002);
        }
    }
}

std::vector<std::pair<double, double>>
GLFWApp::Align(std::vector<std::pair<double, double>> l, double offset)
{
    for (auto &elem : l)
    {
        elem.first = fmod(elem.first + offset, 1.0);
    }
    return l;
}

void GLFWApp::
    DrawAiMesh(const struct aiScene *sc, const struct aiNode *nd, const Eigen::Affine3d &M, double y)
{
    unsigned int i;
    unsigned int n = 0, t;
    Eigen::Vector3d v;
    Eigen::Vector3d dir(0.4, 0, -0.4);
    glColor3f(0.3, 0.3, 0.3);

    for (; n < nd->mNumMeshes; ++n)
    {
        const struct aiMesh *mesh = sc->mMeshes[nd->mMeshes[n]];

        for (t = 0; t < mesh->mNumFaces; ++t)
        {
            const struct aiFace *face = &mesh->mFaces[t];
            GLenum face_mode;

            switch (face->mNumIndices)
            {
            case 1:
                face_mode = GL_POINTS;
                break;
            case 2:
                face_mode = GL_LINES;
                break;
            case 3:
                face_mode = GL_TRIANGLES;
                break;
            default:
                face_mode = GL_POLYGON;
                break;
            }
            glBegin(face_mode);
            for (i = 0; i < face->mNumIndices; i++)
            {
                int index = face->mIndices[i];

                v[0] = (&mesh->mVertices[index].x)[0];
                v[1] = (&mesh->mVertices[index].x)[1];
                v[2] = (&mesh->mVertices[index].x)[2];
                v = M * v;
                double h = v[1] - y;

                v += h * dir;

                v[1] = y + 0.001;
                glVertex3f(v[0], v[1], v[2]);
            }
            glEnd();
        }
    }

    for (n = 0; n < nd->mNumChildren; ++n)
    {
        DrawAiMesh(sc, nd->mChildren[n], M, y);
    }
}

void GLFWApp::DrawMuscleTorque()
{

    for (auto j : mEnv->GetCharacter()->GetSkeleton()->getJoints())
    {

        Eigen::Isometry3d t = j->getChildBodyNode()->getTransform() * j->getTransformFromChildBodyNode();
        Eigen::Vector3d p;
        glPushMatrix();
        if (j->getNumDofs() == 3)
        {
            p = t * Eigen::Vector3d::Zero();
            glTranslated(p[0], p[1], p[2]);
            GUI::DrawSphere(0.005);
            {

                Eigen::Vector3d f = 0.01 * mMuscleTorque.segment(j->getIndexInSkeleton(0), 3);
                GUI::DrawLine(Eigen::Vector3d::Zero(), f, Eigen::Vector3d(1, 0, 0));
            }
        }
        else if (j->getNumDofs() == 1)
        {
            p = t * Eigen::Vector3d(0, 0, 0);
            glTranslatef(p[0], p[1], p[2]);
            {
                double f = 0.01 * mMuscleTorque[j->getIndexInSkeleton(0)];
                GUI::DrawLine(Eigen::Vector3d::Zero(), Eigen::Vector3d(f, 0, 0), Eigen::Vector3d(1, 0, 0));
            }
            glRotated(90, 0, 1, 0);
            GUI::DrawCylinder(0.005, 0.005);
        }
        glPopMatrix();
    }
}

void GLFWApp::DrawShadow(const Eigen::Vector3d &scale, const aiScene *mesh, double y)
{
    glDisable(GL_LIGHTING);
    glPushMatrix();
    glScalef(scale[0], scale[1], scale[2]);
    GLfloat matrix[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, matrix);
    Eigen::Matrix3d A;
    Eigen::Vector3d b;
    A << matrix[0], matrix[4], matrix[8],
        matrix[1], matrix[5], matrix[9],
        matrix[2], matrix[6], matrix[10];
    b << matrix[12], matrix[13], matrix[14];

    Eigen::Affine3d M;
    M.linear() = A;
    M.translation() = b;
    M = (mViewMatrix.inverse()) * M;

    glPushMatrix();
    glLoadIdentity();
    glMultMatrixd(mViewMatrix.data());
    DrawAiMesh(mesh, mesh->mRootNode, M, y);
    glPopMatrix();
    glPopMatrix();
    glEnable(GL_LIGHTING);
}

void GLFWApp::SaveGraphData()
{
    std::cout << "[DEBUG] Saving Graph Data.... " << std::endl;
    mSavedGraphData.clear();
    for (auto m : mGraphData)
        mSavedGraphData.push_back(m);
}

void GLFWApp::AddGraphData()
{

    int idx = 0;
    {
        mGraphData[idx].r_left.push_back(std::pair(mEnv->GetGlobalPhase(), -180.0 / M_PI * GetProjectedAngle(mEnv->GetCharacter()->GetSkeleton()->getJoint("FemurL")->getPositions(), 0)));
        mGraphData[idx].r_right.push_back(std::pair(mEnv->GetGlobalPhase(), -180.0 / M_PI * GetProjectedAngle(mEnv->GetCharacter()->GetSkeleton()->getJoint("FemurR")->getPositions(), 0)));
        mGraphData[idx].r_ref.push_back(std::pair(mEnv->GetGlobalPhase(), -180.0 / M_PI * GetProjectedAngle(mEnv->GetBVHSkeleton()->getJoint("FemurR")->getPositions(), 0)));

        idx++;
        mGraphData[idx].r_left.push_back(std::pair(mEnv->GetGlobalPhase(), 180.0 / M_PI * mEnv->GetCharacter()->GetSkeleton()->getJoint("TibiaL")->getPosition(0)));
        mGraphData[idx].r_right.push_back(std::pair(mEnv->GetGlobalPhase(), 180.0 / M_PI * mEnv->GetCharacter()->GetSkeleton()->getJoint("TibiaR")->getPosition(0)));
        mGraphData[idx].r_ref.push_back(std::pair(mEnv->GetGlobalPhase(), 180.0 / M_PI * GetProjectedAngle(mEnv->GetBVHSkeleton()->getJoint("TibiaR")->getPositions(), 0)));

        idx++;

        auto skel = mEnv->GetCharacter()->GetSkeleton();

        if (!mCalibrateGraph)
        {
            mGraphData[idx].r_left.push_back(std::pair(mEnv->GetGlobalPhase(), -180.0 / M_PI * GetProjectedAngle(mEnv->GetCharacter()->GetSkeleton()->getJoint("TalusL")->getPositions(), 0)));
            mGraphData[idx].r_right.push_back(std::pair(mEnv->GetGlobalPhase(), -180.0 / M_PI * GetProjectedAngle(mEnv->GetCharacter()->GetSkeleton()->getJoint("TalusR")->getPositions(), 0)));
        }
        else
        {
            mGraphData[idx].r_left.push_back(std::pair(mEnv->GetGlobalPhase(), -180.0 / M_PI * GetProjectedAngle(BallJoint::convertToPositions((skel->getBodyNode("TibiaL")->getTransform().inverse() * skel->getBodyNode("TalusL")->getTransform()).linear()), 0)));
            mGraphData[idx].r_right.push_back(std::pair(mEnv->GetGlobalPhase(), -180.0 / M_PI * GetProjectedAngle(BallJoint::convertToPositions((skel->getBodyNode("TibiaR")->getTransform().inverse() * skel->getBodyNode("TalusR")->getTransform()).linear()), 0)));
        }
        mGraphData[idx].r_ref.push_back(std::pair(mEnv->GetGlobalPhase(), -180.0 / M_PI * GetProjectedAngle(mEnv->GetBVHSkeleton()->getJoint("TalusR")->getPositions(), 0)));

        idx++;
        mGraphData[idx].r_left.push_back(std::pair(mEnv->GetGlobalPhase(), -180.0 / M_PI * GetProjectedAngle(mEnv->GetCharacter()->GetSkeleton()->getJoint("Pelvis")->getPositions().head(3), 0)));
        idx++;
        mGraphData[idx].r_left.push_back(std::pair(mEnv->GetGlobalPhase(), -180.0 / M_PI * GetProjectedAngle(mEnv->GetCharacter()->GetSkeleton()->getJoint("Pelvis")->getPositions().head(3), 1)));
        idx++;
        mGraphData[idx].r_left.push_back(std::pair(mEnv->GetGlobalPhase(), 180.0 / M_PI * GetProjectedAngle(mEnv->GetCharacter()->GetSkeleton()->getJoint("Pelvis")->getPositions().head(3), 2)));
        idx++;
    }

    Eigen::Vector2i c = mEnv->GetIsContact();
    mContactData.r_left.push_back(std::pair(mEnv->GetGlobalPhase(), c[0]));
    mContactData.r_right.push_back(std::pair(mEnv->GetGlobalPhase(), c[1]));

    {
        Eigen::Vector2i c = mEnv->GetIsContact();
        mGraphData[idx].r_left.push_back(std::pair(mEnv->GetGlobalPhase(), c[0]));
        mGraphData[idx].r_right.push_back(std::pair(mEnv->GetGlobalPhase(), c[1]));
    }
}
