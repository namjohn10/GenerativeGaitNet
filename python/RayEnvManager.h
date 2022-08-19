#ifndef _RAY_ENV_MANAGER_H_
#define _RAY_ENV_MANAGER_H_

#include "Environment.h"
#include "NumPyHelper.h"

namespace py = pybind11;

class RayEnvManager
{
private:
	MASS::Environment *mEnv;

public:
	RayEnvManager(std::string meta_file);
	void Initialize_from_text(std::string metadata) { mEnv->Initialize_from_text(metadata); }
	int GetNumState() { return mEnv->GetNumState(); }
	int GetNumAction() { return mEnv->GetNumAction(); }
	int GetNumActiveDof() { return mEnv->GetNumActiveDof(); }
	int GetSimulationHz() { return mEnv->GetSimulationHz(); }
	int GetControlHz() { return mEnv->GetControlHz(); }
	int GetNumSteps() { return mEnv->GetNumSteps(); }
	bool UseMuscle() { return mEnv->GetUseMuscle(); }
	void Step() { mEnv->Step(); }
	void Reset() { mEnv->Reset(); }
	int IsEndOfEpisode() { return mEnv->IsEndOfEpisode(); }
	bool IsComplete() { return mEnv->GetIsComplete(); }
	void SetEoeTime() { mEnv->setEoeTime(); }
	double GetReward() { return mEnv->GetReward(); }
	void StepsAtOnce();
	int GetNumTotalMuscleRelatedDofs() { return mEnv->GetNumTotalRelatedDofs(); }
	int GetNumMuscles() { return mEnv->GetCharacter()->GetMuscles().size(); }
	int GetInferencePerSim() { return mEnv->GetInferencePerSim(); }
	int GetNumParamState() { return mEnv->GetNumParamState(); }
	bool UseAdaptiveSampling() { return mEnv->GetUseAdaptiveSampling(); }
	bool UseTimeWarp() { return mEnv->GetUseTimeWarping(); }
	void SetParamState(py::array_t<float> np_array);
	py::array_t<float> GetState() { return toNumPyArray(mEnv->GetState()); }
	py::array_t<float> GetProjState(py::array_t<float> minv, py::array_t<float> maxv) { return toNumPyArray(mEnv->GetProjState(toEigenVector(minv), toEigenVector(maxv))); }
	py::array_t<float> GetParamState() { return toNumPyArray(mEnv->GetParamState()); }

	void SetAction(py::array_t<float> np_array) { mEnv->SetAction(toEigenVector(np_array)); }
	void SetMuscleAction(py::array_t<float> np_array) { mEnv->SetMuscleAction(toEigenVector(np_array)); }

	py::list GetMuscleTuple(bool isReuse);
	py::array_t<float> GetMinV() { return toNumPyArray(mEnv->GetMinV()); }
	py::array_t<float> GetMaxV() { return toNumPyArray(mEnv->GetMaxV()); }
	py::str GetMetadata() { return mEnv->GetMetadata(); }
	py::list GetSpace(py::str metadata);

	double GetWeight() { return mEnv->GetWeight(); }
	void SetWeight(double w) { mEnv->SetWeight(w); }
	double CalculateWeight(py::array_t<float> np_array, double axis) { return mEnv->CalculateWeight(toEigenVector(np_array), axis); }

	double GetPhase() { return mEnv->GetPhase(); }
	py::array_t<float> GetDisplacement() { return toNumPyArray(mEnv->GetDisplacement()); }
	bool UseDisplacement() { return mEnv->UseDisplacement(); }

	py::array_t<float> GetParamSamplingPolicy() { return toNumPyArray(mEnv->GetParamSamplingPolicy()); }

	void UpdateHeadInfo() { mEnv->UpdateHeadInfo(); }
	int GetCascadingType() { return mEnv->GetCascadingType(); }
	int GetStateType() { return mEnv->GetStateType(); }
	int GetStateDiffNum() { return mEnv->GetStateDiffNum(); }
};

#endif // MSS_RayEnvManager_H
