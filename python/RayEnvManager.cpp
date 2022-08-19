#include "RayEnvManager.h"

RayEnvManager::
    RayEnvManager(std::string meta_file)
{
    dart::math::seedRand();
    mEnv = new MASS::Environment(false);
    if (!mEnv->Initialize_from_path(std::string(meta_file)))
        mEnv->Initialize_from_text(meta_file);
}
void RayEnvManager::
    SetParamState(py::array_t<float> np_array)
{
    mEnv->SetParamState(toEigenVector(np_array));
    if (mEnv->GetUseOptimization())
        mEnv->NaiveMotionOptimization();
}

void RayEnvManager::
    StepsAtOnce()
{
    int num = GetNumSteps();
    for (int j = 0; j < num; j++)
        mEnv->Step();
}
py::list
RayEnvManager::
    GetMuscleTuple(bool isRender)
{
    Eigen::VectorXd JtA;
    Eigen::VectorXd Jtp;
    Eigen::VectorXd tau_des;
    Eigen::MatrixXd L;
    Eigen::VectorXd b;

    auto &tp = mEnv->GetMuscleTuple(isRender);
    JtA = tp.JtA;
    Jtp = tp.Jtp;
    tau_des = tp.tau_des;
    L = tp.L;
    b = tp.b;

    py::list t;
    t.append(toNumPyArray(JtA));
    t.append(toNumPyArray(Jtp));
    t.append(toNumPyArray(tau_des));
    t.append(toNumPyArray(L));
    t.append(toNumPyArray(b));
    return t;
}
py::list
RayEnvManager::
    GetSpace(py::str metadata)
{
    auto space = mEnv->GetSpace(metadata.cast<std::string>());
    py::list t;
    t.append(toNumPyArray(space.first));
    t.append(toNumPyArray(space.second));
    return t;
}

PYBIND11_MODULE(pymss, m)
{
    py::class_<RayEnvManager>(m, "RayEnvManager")
        .def(py::init<std::string>())
        .def("GetNumState", &RayEnvManager::GetNumState)
        .def("GetNumAction", &RayEnvManager::GetNumAction)
        .def("GetSimulationHz", &RayEnvManager::GetSimulationHz)
        .def("GetControlHz", &RayEnvManager::GetControlHz)
        .def("GetNumSteps", &RayEnvManager::GetNumSteps)
        .def("UseTimeWarp", &RayEnvManager::UseTimeWarp)
        .def("UseMuscle", &RayEnvManager::UseMuscle)
        .def("UseAdaptiveSampling", &RayEnvManager::UseAdaptiveSampling)
        .def("Step", &RayEnvManager::Step)
        .def("Reset", &RayEnvManager::Reset)
        .def("IsEndOfEpisode", &RayEnvManager::IsEndOfEpisode)
        .def("IsComplete", &RayEnvManager::IsComplete)
        .def("SetEoeTime", &RayEnvManager::SetEoeTime)
        .def("GetState", &RayEnvManager::GetState)
        .def("GetStateType", &RayEnvManager::GetStateType)
        .def("GetProjState", &RayEnvManager::GetProjState)
        .def("GetParamState", &RayEnvManager::GetParamState)
        .def("GetInferencePerSim", &RayEnvManager::GetInferencePerSim)
        .def("SetParamState", &RayEnvManager::SetParamState)
        .def("SetAction", &RayEnvManager::SetAction)
        .def("GetReward", &RayEnvManager::GetReward)
        .def("StepsAtOnce", &RayEnvManager::StepsAtOnce)
        .def("GetNumTotalMuscleRelatedDofs", &RayEnvManager::GetNumTotalMuscleRelatedDofs)
        .def("GetNumMuscles", &RayEnvManager::GetNumMuscles)
        .def("SetMuscleAction", &RayEnvManager::SetMuscleAction)
        .def("GetMuscleTuple", &RayEnvManager::GetMuscleTuple)
        .def("GetNumParamState", &RayEnvManager::GetNumParamState)
        .def("GetMinV", &RayEnvManager::GetMinV)
        .def("GetMaxV", &RayEnvManager::GetMaxV)
        .def("GetMetadata", &RayEnvManager::GetMetadata)
        .def("GetSpace", &RayEnvManager::GetSpace)
        .def("GetWeight", &RayEnvManager::GetWeight)
        .def("SetWeight", &RayEnvManager::SetWeight)
        .def("CalculateWeight", &RayEnvManager::CalculateWeight)
        .def("GetNumActiveDof", &RayEnvManager::GetNumActiveDof)
        .def("GetPhase", &RayEnvManager::GetPhase)
        .def("GetDisplacement", &RayEnvManager::GetDisplacement)
        .def("UseDisplacement", &RayEnvManager::UseDisplacement)
        .def("GetParamSamplingPolicy", &RayEnvManager::GetParamSamplingPolicy)
        .def("UpdateHeadInfo", &RayEnvManager::UpdateHeadInfo)
        .def("GetCascadingType", &RayEnvManager::GetCascadingType)
        .def("GetStateDiffNum", &RayEnvManager::GetStateDiffNum);
}
