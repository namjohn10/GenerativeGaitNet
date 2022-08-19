#ifndef __MASS_ENVIRONMENT_H__
#define __MASS_ENVIRONMENT_H__
#include "dart/dart.hpp"
#include "Character.h"
#include "Muscle.h"
#include "BVH.h"

using namespace dart::dynamics;

static std::map<std::string, std::vector<std::string>> MuscleGroups = {
	{"Group_L_Hip_Extensor", {"L_Gluteus_Maximus", "L_Gluteus_Medius", "L_Gluteus_Minimus"}},
	{"Group_R_Hip_Extensor", {"R_Gluteus_Maximus", "R_Gluteus_Medius", "R_Gluteus_Minimus"}},
	{"Group_L_Adductor_Magnus_Group", {"L_Adductor_Longus", "L_Adductor_Brevis", "L_Adductor_Magnus"}},
	{"Group_R_Adductor_Magnus_Group", {"R_Adductor_Longus", "R_Adductor_Brevis", "R_Adductor_Magnus"}},
	{"Group_L_Quadriceps_Femoris", {"L_Rectus_Femoris", "L_Vastus_Lateralis", "L_Vastus_Medialis", "L_Vastus_Intermedius"}},
	{"Group_R_Quadriceps_Femoris", {"R_Rectus_Femoris", "R_Vastus_Lateralis", "R_Vastus_Medialis", "R_Vastus_Intermedius"}},
	{"Group_L_Hamstring", {"L_Semitendinosus", "L_Semimembranosus", "L_Bicep_Femoris"}},
	{"Group_R_Hamstring", {"R_Semitendinosus", "R_Semimembranosus", "R_Bicep_Femoris"}},
	{"Group_L_Triceps_Surae", {"L_Gastrocnemius", "L_Soleus"}},
	{"Group_R_Triceps_Surae", {"R_Gastrocnemius", "R_Soleus"}},
	{"Group_L_Ankle_Eversion", {"L_Peroneus_Longus"}},
	{"Group_R_Ankle_Eversion", {"R_Peroneus_Longus"}},
	{"Group_L_Ankle_Inversion", {"L_Tibialis_Posterior", "L_Extensor_Hallucis_Longus", "L_Flexor_Digitorum_Longus"}},
	{"Group_R_Ankle_Inversion", {"R_Tibialis_Posterior", "R_Extensor_Hallucis_Longus", "R_Flexor_Digitorum_Longus"}},
};

namespace MASS
{
	class PATH;

	struct ParamElem
	{
		std::string name;
		double min_v;
		double max_v;
		double cur_v;
	};

	struct ParamCategory
	{
		std::string name;
		std::vector<ParamElem> params;
	};

	struct MuscleParam
	{
		double min_ratio;
		double max_ratio;
		double current_ratio;
		std::string name;
		std::vector<Muscle *> muscle;
		bool isGroup;
		bool Compare(std::string m_name)
		{
			if (isGroup)
			{
				if (MuscleGroups.count(name) > 0)
				{
					for (std::string &mc_name : MuscleGroups[name])
					{
						if (m_name.find(mc_name) != std::string::npos)
							return true;
					}
					return false;
				}
				else
					return (m_name.find(name) != std::string::npos);
			}
			else
				return name == m_name;
		}
	};

	struct SkelParam
	{
		double min_ratio;
		double max_ratio;
		double current_ratio;
		std::string name;
		BodyNode *body_node;
		bool Compare(std::string s_name) { return name == s_name; }
	};

	struct MuscleTuple
	{
		Eigen::VectorXd JtA;
		Eigen::VectorXd Jtp;
		Eigen::MatrixXd L;
		Eigen::VectorXd b;
		Eigen::VectorXd tau_des;
	};

	class Environment
	{
	public:
		Environment(bool isRender = false);

		void SetUseAdaptiveSampling(bool use_adaptivesampling) { mUseAdaptiveSampling = use_adaptivesampling; }
		void SetUseMuscle(bool use_muscle) { mUseMuscle = use_muscle; }
		void SetUseExcitation(bool use_excitation) { mUseExcitation = use_excitation; }

		void SetControlHz(int con_hz) { mControlHz = con_hz; }
		void SetSimulationHz(int sim_hz) { mSimulationHz = sim_hz; }
		void SetCharacter(Character *character) { mCharacter = character; }
		void SetGround(const dart::dynamics::SkeletonPtr &ground) { mGround = ground; }

		void SetMuscleAction(const Eigen::VectorXd &a);

		void SetIsRender(bool isRender) { mIsRender = isRender; }

		bool Initialize_from_path(const std::string &path);
		void Initialize();
		void Initialize_from_text(const std::string &metadata, bool load_obj = true);

		double GetReward();
		double GetAvgVelocityReward();
		std::map<std::string, double> GetRewardMap();
		double GetLocoPrinReward();
		double GetImitationReward();
		double GetOverTorqueReward();
		double GetMetabolicReward();
		double GetCOMReward();
		double GetStepReward();
		double GetPassiveForceReward();

		Eigen::Vector3d GetAngularMomentum();

		int GetCascadingType(std::string metadata);

		int &GetMetabolicType() { return mMetabolicType; }

		void SetParamState(Eigen::VectorXd ParamState);

		int GetNumParamState() { return mNumParamState; }
		Eigen::VectorXd GetMuscleLengthParamState();
		void SetMuscleLengthParamState(Eigen::VectorXd ParamState);
		Eigen::VectorXd GetMuscleForceParamState();
		void SetMuscleForceParamState(Eigen::VectorXd ParamState);
		Eigen::VectorXd GetSkelLengthParamState();
		void SetSkelLengthParamState(Eigen::VectorXd ParamState);

		Eigen::VectorXd GetState();
		Eigen::VectorXd GetParamState();
		Eigen::VectorXd GetProjState(Eigen::VectorXd minv, Eigen::VectorXd maxv);
		Eigen::VectorXd GetMuscleState(bool isRender = false);

		Eigen::VectorXd GetMirrorState(Eigen::VectorXd state);
		Eigen::VectorXd GetMirrorAction(Eigen::VectorXd action);
		Eigen::VectorXd GetMirrorActivation(Eigen::VectorXd activation);

		void SetAction(const Eigen::VectorXd &a);
		Eigen::VectorXd GetDesiredTorques();
		Eigen::VectorXd GetDesiredTorquesValue() { return mDesiredTorque; }
		Eigen::VectorXd GetCurTorques() { return 0.01 * mCharacter->GetSkeleton()->getAccelerations(); }
		Eigen::VectorXd GetMuscleTorques();

		Eigen::VectorXd GetDerivationOfActivation(Eigen::VectorXd a, Eigen::VectorXd e);

		int GetStateType() { return mStateType; }

		double CalculateWeight(Eigen::VectorXd state_diff, double a);
		void Step();

		void Reset();
		int IsEndOfEpisode();

		const dart::simulation::WorldPtr &GetWorld() { return mWorld; }
		Character *GetCharacter() { return mCharacter; }
		const dart::dynamics::SkeletonPtr &GetGround() { return mGround; }
		int GetControlHz() { return mControlHz; }
		int GetSimulationHz() { return mSimulationHz; }
		int GetNumTotalRelatedDofs() { return mCurrentMuscleTuple.JtA.rows(); }

		MuscleTuple &GetMuscleTuple(bool isRender = false);

		int GetNumState() { return mNumState; }
		int GetNumMuscleState() { return mNumMuscleState; }
		int GetNumAction() { return mNumActiveDof + mUseTimeWarping + mUseVelocity + mUseStepWarping; }
		int GetNumActiveDof() { return mNumActiveDof; }
		int GetNumSteps() { return mSimulationHz / mControlHz; }
		int GetInferencePerSim() { return mInferencePerSim; }

		const Eigen::VectorXd &GetAction() { return mAction; }
		const Eigen::VectorXd &GetActivationLevels() { return mActivationLevels; }
		const Eigen::VectorXd &GetExcitationLevels() { return mExcitationLevels; }
		const Eigen::VectorXd &GetAverageActivationLevels() { return mAverageActivationLevels; }
		const Eigen::VectorXd &GetTargetPositions() { return mTargetPositions; }

		bool GetUseMuscle() { return mUseMuscle; }
		bool &GetUseExcitation() { return mUseExcitation; }
		bool GetUseContractileState() { return mUseContractileState; }
		bool GetUseAdaptiveSampling() { return mUseAdaptiveSampling; }
		bool GetUseTimeWarping() { return mUseTimeWarping; }
		bool GetUseStepWarping() { return mUseStepWarping; }

		Eigen::VectorXd GetPassiveMuscleTorques();
		double GetPhase();
		double GetGlobalPhase();

		std::vector<MuscleParam> &getMuscleLengthParams() { return mMuscleLengthParams; }
		int GetMuscleLengthParamNum() { return mMuscleLengthParamNum; }
		std::vector<std::string> GetMuscleLengthParamName();

		std::vector<MuscleParam> &getMuscleForceParams() { return mMuscleForceParams; }
		int GetMuscleForceParamNum() { return mMuscleForceParamNum; }
		std::vector<std::string> GetMuscleForceParamName();

		std::vector<SkelParam> &getSkelLengthParams() { return mSkelLengthParams; }
		int GetSkelLengthParamNum() { return mSkelLengthParamNum; }
		std::vector<std::string> GetSkelLengthParamName();

		const dart::dynamics::SkeletonPtr &GetReferenceSkeleton() { return mReferenceSkeleton; }
		const dart::dynamics::SkeletonPtr &GetBVHSkeleton() { return mBVHSkeleton; }

		bool GetIsRender() { return mIsRender; }

		void ApplyMuscleParameter();
		void ApplySkelParameter();

		bool GetIsComplete() { return mIsComplete; }

		Eigen::VectorXd mTargetPositions, mTargetVelocities;
		Eigen::VectorXd mBVHPositions, mBVHVelocities;

		Eigen::VectorXd GetMinV();
		Eigen::VectorXd GetMaxV();
		Eigen::VectorXd GetNormalV();

		std::vector<std::string> GetParamName();
		Eigen::VectorXd getPositionDifferences(Eigen::VectorXd p1, Eigen::VectorXd p2);

		void setEoeTime() { mEoeTime = mWorld->getTime(); }

		std::string GetMetadata() { return metadata; }

		std::pair<Eigen::VectorXd, Eigen::VectorXd> GetSpace(std::string metadata);

		bool &GetIsTorqueClip() { return mIsTorqueClip; }
		void SetIsTorqueClip(bool IsTorqueClip) { mIsTorqueClip = IsTorqueClip; }

		void NaivePoseOptimization(int max_iter = 1000);
		void NaiveMotionOptimization();
		void SetUseOptimization(bool isUseOptimization) { mIsUseOptimization = isUseOptimization; }
		bool GetUseOptimization() { return mIsUseOptimization; }

		void SetWeight(double w) { mWeight = w; }
		double GetWeight() { return mWeight; }
		std::vector<BoneInfo> &GetSkelInfo() { return mSkelInfos; }

		bool GetUseVelocity() { return mUseVelocity; }
		bool GetUsePhase() { return mUsePhase; }
		bool GetUseStride() { return mUseStride; }

		double GetPhaseRatio() { return mPhaseRatio * sqrt(1 / mGlobalRatio); }
		double GetTargetVelocity();

		double GetAvgVelocity()
		{
			int horizon = mCharacter->GetBVH()->GetMaxTime() * mSimulationHz / (mPhaseRatio * sqrt(1 / mGlobalRatio));
			Eigen::Vector3d avg_vel;
			if (mComTrajectory.size() > horizon)
				avg_vel = ((mComTrajectory.back() - mComTrajectory[mComTrajectory.size() - horizon]) / horizon) * mSimulationHz;
			else if (mComTrajectory.size() <= 1)
				avg_vel = mCharacter->GetSkeleton()->getCOMLinearVelocity();
			else
				avg_vel = ((mComTrajectory.back() - mComTrajectory.front()) / (mComTrajectory.size() - 1)) * mSimulationHz;
			return avg_vel[2];
		}
		double GetInsVelocity() { return mCharacter->GetSkeleton()->getCOMLinearVelocity()[2]; }

		Eigen::Vector3d GetNextTargetFoot() { return mNextTargetFoot; }
		Eigen::Vector3d GetCurrentTargetFoot() { return mCurrentTargetFoot; }
		Eigen::Vector3d GetCurrentFoot() { return mCurrentFoot; }

		Eigen::VectorXd GetDisplacement();
		bool UseDisplacement() { return mUseDisplacement; }

		void CreateTotalParams();
		std::vector<ParamCategory> GetTotalParams() { return mTotalParams; }

		Eigen::VectorXd GetParamSamplingPolicy();
		double GetStepDisplacement() { return mStepDisplacement; }
		double GetPhasepDisplacement() { return mPhaseDisplacement; }
		double GetActionType() { return mActionType; }
		bool IsArmExist();
		void updateCharacterInLimit();
		void UpdateHeadInfo();
		double GetHeight()
		{
			double head_ratio = 1.0;
			for (auto skel_elem : mSkelLengthParams)
			{
				if (skel_elem.name == "Head")
					head_ratio = skel_elem.current_ratio;
			}
			return mBodyHeight * mGlobalRatio + mHeadHeight * head_ratio;
		}

		double UpdatePDParameter(Joint *jn, bool isFirst = false);

		bool &GetIsTorqueSymMode() { return mIsTorqueSymMode; }
		bool &GetIsMuscleSymMode() { return mIsMuscleSymMode; }

		Eigen::Vector2i GetIsContact();
		int GetCascadingType() { return mCascadingType; }
		int GetStateDiffNum() { return (mStateType == 11 ? GetNumState() - getMuscleLengthParams().size() - getMuscleForceParams().size() : GetNumState()); }

		Eigen::VectorXd getAllMuscleParam()
		{
			Eigen::VectorXd v = Eigen::VectorXd::Ones(2 * mCharacter->GetMuscles().size());
			int idx = 0;
			for (auto m : mCharacter->GetMuscles())
				v[idx++] = m->get_f();
			for (auto m : mCharacter->GetMuscles())
				v[idx++] = m->get_l();
			return v;
		}
		void UpdateParamState();
		void setAllMuscleParam(Eigen::VectorXd v)
		{
			int idx = 0;
			for (auto m : mCharacter->GetMuscles())
				m->change_f(v[idx++]);
			for (auto m : mCharacter->GetMuscles())
				m->change_l(v[idx++]);
			UpdateParamState();
		}
		void applyProjAllParam(Eigen::VectorXd v)
		{

			int idx = 0;
			for (auto m : mCharacter->GetMuscles())
			{
				if (v[idx] > m->get_f())
					;
				m->change_f(v[idx]);
				idx++;
			}
			for (auto m : mCharacter->GetMuscles())
			{
				if (v[idx] > m->get_l())
					m->change_l(v[idx]);
				idx++;
			}
		}
		int getMuscleStateStartIdx() { return mMuscleStateStartIdx; }

	private:
		int mMuscleStateStartIdx;
		int mCurrentStance;

		bool mIsTorqueSymMode;
		bool mIsMuscleSymMode;

		Eigen::Vector3d mNextTargetFoot;
		Eigen::Vector3d mCurrentTargetFoot;
		Eigen::Vector3d mCurrentFoot;

		int mStateType;
		int mPhaseType;
		int mMetabolicType;

		bool mIsUseOptimization;
		bool mIsTorqueClip;
		std::string metadata;

		dart::simulation::WorldPtr mWorld;
		dart::dynamics::SkeletonPtr mGround;

		int mControlHz, mSimulationHz;
		bool mUseMuscle;
		bool mUseNewSPD;

		bool mUseSym;
		bool mUseExcitation;
		Character *mCharacter;
		Eigen::VectorXd mAction;

		int mNumState;
		int mNumMuscleState;
		int mNumActiveDof;
		int mRootJointDof;

		Eigen::VectorXd mExcitationLevels;
		Eigen::VectorXd mActivationLevels;
		Eigen::VectorXd mAverageActivationLevels;

		Eigen::VectorXd mDesiredTorque;
		Eigen::VectorXd mNetDesiredTorque;
		Eigen::VectorXd mClipedDesiredTorque;

		MuscleTuple mCurrentMuscleTuple;

		int mMuscleLengthParamNum;
		int mMuscleForceParamNum;

		std::vector<MuscleParam> mMuscleLengthParams;
		std::vector<MuscleParam> mMuscleForceParams;

		int mSkelLengthParamNum;
		std::vector<SkelParam> mSkelLengthParams;

		int mNumParamState;
		bool mIsComplete;

		bool mUseAdaptiveSampling;
		bool mUseContractileState;
		bool mUseMuscleRegularization;

		dart::dynamics::SkeletonPtr mReferenceSkeleton;
		dart::dynamics::SkeletonPtr mBVHSkeleton;

		bool mIsRender;

		double mAverageReward;
		int mInferencePerSim;

		double max_r;
		bool mUseConstraint;
		double mEoeTime;

		std::vector<Eigen::VectorXd> mActivationBuf;
		std::map<std::string, double> mValues;

		double mActiveForce;
		double mPassiveForce;

		double mOverTorque;

		std::vector<BoneInfo> mSkelInfos;
		double mWeight;

		bool mUseLocoPrinReward;

		bool mUseVelocity;
		double mMinCOMVelocity;
		double mMaxCOMVelocity;
		double mTargetCOMVelocity;

		double mGlobalTime;

		double mLocalTime;
		bool mUseTimeWarping;
		double mPhaseDisplacement;

		bool mUseStepWarping;
		double mStepDisplacement;

		double mUsedTorque;
		double mNetUsedTorque;

		double mSimUsedTorque;
		double mNetSimUsedTorque;

		double mTotalAcc;

		int mStepCount;
		int mCycleCount;

		bool mUsePhase;
		double mMinPhase;
		double mMaxPhase;
		double mPhaseRatio;

		double mStrideRatio;
		double mRefStride;

		bool mUseStride;
		double mMinStride;
		double mMaxStride;

		int mGlobalPrevIdx;
		double mPrevTime;

		std::vector<Eigen::Vector3d> mComTrajectory;
		bool mUseDisplacement;

		std::vector<ParamCategory> mTotalParams;

		bool mSelfCollision;

		double mGlobalRatio;

		int mActionType;

		double mMetabolicEnergy;
		double mMass;

		double mBodyHeight;
		double mHeadHeight;

		bool mIsConstantPDParameter;

		Eigen::Vector3d mPrevCOM;
		Eigen::Vector3d mCurCOM;

		Eigen::Vector3d mHeadPrevLinearVel;
		Eigen::Vector3d mHeadPrevAngularVel;

		Eigen::VectorXd mKv;
		Eigen::VectorXd mKp;

		Eigen::VectorXd mRefMass;
		double mOriginalKp;

		double mActionScale;
		double mPhaseScale;
		bool mUseImitation;
		bool mUseCriticalDamping;
		double mStartPhase;
		bool mIsNewPhase;

		double mMetabolicWeight;
		double mRotDiffWeight;
		double mLinearAccWeight;
		double mAngularAccWeight;
		std::default_random_engine generator;
		std::normal_distribution<double> distribution;
		bool mIsNew;
		bool mUseInitNegative;
		bool mUseAbsStep;

		double mPrevPhase;
		Eigen::VectorXd mOptWeight;
		double mUseAdaptiveKp;
		double mCascadingType;
	};
};

#endif