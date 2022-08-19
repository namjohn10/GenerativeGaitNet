#include "Environment.h"
#include "DARTHelper.h"
#include "Character.h"
#include "BVH.h"
#include "Muscle.h"
#include "dart/collision/bullet/bullet.hpp"

#include <initializer_list>
#include <chrono>

#include <random>

using namespace dart;
using namespace dart::simulation;
using namespace MASS;

double clamp(double v, double min_v, double max_v)
{
	if (max_v < v)
		return max_v;
	else if (min_v > v)
		return min_v;
	else
		return v;
}
void Environment::
	UpdateParamState()
{
	for (auto &m_l : mMuscleLengthParams)
	{
		double current_ratio = 1.0;
		for (auto m : m_l.muscle)
		{
			if (current_ratio > m->get_l())
				current_ratio = m->get_l();
		}
		m_l.current_ratio = current_ratio;
	}
	for (auto &m_f : mMuscleForceParams)
	{
		double current_ratio = 1.0;
		for (auto m : m_f.muscle)
			if (current_ratio > m->get_f())
				current_ratio = m->get_f();
		m_f.current_ratio = current_ratio;
	}
}

Environment::
	Environment(bool isRender)
	: mUseContractileState(false), mIsUseOptimization(false), mIsTorqueClip(true), mIsComplete(false), mUseConstraint(false), mUseAdaptiveSampling(false),
	  mInferencePerSim(1), mEoeTime(0), mControlHz(30), mSimulationHz(900), mNumMuscleState(0), mUseVelocity(false),
	  mWorld(std::make_shared<World>()), mUseMuscle(true),
	  mWeight(1.0), mLocalTime(0.0), mUseTimeWarping(false),

	  mIsTorqueSymMode(false),
	  mIsMuscleSymMode(false),
	  mGlobalTime(0.0), mUseStepWarping(false),
	  mMinCOMVelocity(1.0), mMaxCOMVelocity(1.0), mTargetCOMVelocity(1.0),
	  mIsRender(isRender), mUsePhase(false), mPhaseRatio(1.0), mStrideRatio(1), mUseStride(false), mUseDisplacement(false), mStateType(0), mMetabolicType(0), mPhaseType(0),
	  mSelfCollision(false), mGlobalRatio(1.0), mActionType(0), mUseAdaptiveKp(true),
	  mRefStride(1.34), mIsConstantPDParameter(true), mOriginalKp(300), mUseLocoPrinReward(true), mUseCriticalDamping(false), mCascadingType(0),
	  mActionScale(0.04), mUseImitation(true), mPhaseScale(0.01), mStartPhase(0.0), mIsNewPhase(false), mMetabolicWeight(0.2), mRotDiffWeight(2.0), mLinearAccWeight(1.0), mAngularAccWeight(0.5), mUseExcitation(false), mUseNewSPD(false), mUseInitNegative(false), mUseAbsStep(true)
{
	distribution = std::normal_distribution<double>(0, 0.05);
	mComTrajectory.clear();
}

bool Environment::
	Initialize_from_path(const std::string &path)
{
	std::ifstream ifs(path);
	if (!(ifs.is_open()))
		return false;
	std::stringstream ss;
	ss << ifs.rdbuf();
	ifs.close();
	Initialize_from_text(ss.str());
	return true;
}

void Environment::
	Initialize_from_text(const std::string &_metadata, bool load_obj)
{
	metadata = _metadata;
	std::stringstream ifs;
	ifs.str(metadata);

	std::string str;
	std::string index;
	std::stringstream ss;
	mCharacter = new MASS::Character();

	while (!ifs.eof())
	{
		str.clear();
		index.clear();
		ss.clear();

		std::getline(ifs, str);
		ss.str(str);
		ss >> index;
		if (!index.compare("use_adaptivesampling"))
		{
			std::string str2;
			ss >> str2;
			if (!str2.compare("true"))
				this->SetUseAdaptiveSampling(true);
			else
				this->SetUseAdaptiveSampling(false);
		}
		else if (!index.compare("use_contractilestate"))
		{
			std::string str2;
			ss >> str2;
			if (!str2.compare("true"))
				mUseContractileState = true;
			else
				mUseContractileState = false;
		}
		else if (!index.compare("use_locoprin"))
		{
			std::string str2;
			ss >> str2;
			if (!str2.compare("true"))
				mUseLocoPrinReward = true;
			else
				mUseLocoPrinReward = false;
		}
		else if (!index.compare("use_adaptivekp"))
		{
			std::string str2;
			ss >> str2;
			if (!str2.compare("true"))
				mUseAdaptiveKp = true;
			else
				mUseAdaptiveKp = false;
		}
		else if (!index.compare("is_torqueclip"))
		{
			std::string str2;
			ss >> str2;
			if (!str2.compare("true"))
				mIsTorqueClip = true;
			else
				mIsTorqueClip = false;
		}
		else if (!index.compare("use_velocity"))
		{
			mUseVelocity = true;
			ss >> mMinCOMVelocity;
			ss >> mMaxCOMVelocity;
		}
		else if (!index.compare("kp"))
			ss >> mOriginalKp;
		else if (!index.compare("metabolic_weight"))
			ss >> mMetabolicWeight;
		else if (!index.compare("rot_diff_weight"))
			ss >> mRotDiffWeight;
		else if (!index.compare("linear_acc_weight"))
			ss >> mLinearAccWeight;
		else if (!index.compare("angular_acc_weight"))
			ss >> mAngularAccWeight;
		else if (!index.compare("phase_type"))
		{
			int phase_type;
			ss >> phase_type;
			mPhaseType = phase_type;
		}
		else if (!index.compare("action_type"))
		{
			int action_type;
			ss >> action_type;
			mActionType = action_type;
		}
		else if (!index.compare("state_type"))
		{
			int state_type;
			ss >> state_type;
			mStateType = state_type;
		}
		else if (!index.compare("metabolic_type"))
		{
			int metabolic_type;
			ss >> metabolic_type;
			mMetabolicType = metabolic_type;
		}
		else if (!index.compare("use_muscle"))
		{
			std::string str2;
			ss >> str2;
			if (!str2.compare("true"))
				this->SetUseMuscle(true);
			else
				this->SetUseMuscle(false);
		}
		else if (!index.compare("use_init_negative"))
		{
			std::string str2;
			ss >> str2;
			if (!str2.compare("true"))
				mUseInitNegative = true;
			else
				mUseInitNegative = false;
		}
		else if (!index.compare("use_newspd"))
		{
			std::string str2;
			ss >> str2;
			if (!str2.compare("true"))
				mUseNewSPD = true;
			else
				mUseNewSPD = false;
			mCharacter->SetUseNewSPD(mUseNewSPD);
		}
		else if (!index.compare("use_excitation"))
		{
			std::string str2;
			ss >> str2;
			if (!str2.compare("true"))
				this->SetUseExcitation(true);
			else
				this->SetUseExcitation(false);
		}
		else if (!index.compare("selfcollision"))
		{
			std::string str2;
			ss >> str2;
			if (!str2.compare("true"))
				mSelfCollision = true;
			else
				mSelfCollision = false;
		}
		else if (!index.compare("use_motionoptimization"))
		{
			std::string str2;
			ss >> str2;
			if (!str2.compare("true"))
				this->SetUseOptimization(true);
			else
				this->SetUseOptimization(false);
		}
		else if (!index.compare("use_abs_step"))
		{
			std::string str2;
			ss >> str2;
			if (!str2.compare("true"))
				mUseAbsStep = true;
			else
				mUseAbsStep = false;
		}
		else if (!index.compare("use_timewarping"))
		{
			std::string str2;
			ss >> str2;
			if (!str2.compare("true"))
				mUseTimeWarping = true;
			else
				mUseTimeWarping = false;
		}
		else if (!index.compare("use_stepwarping"))
		{
			std::string str2;
			ss >> str2;
			if (!str2.compare("true"))
				mUseStepWarping = true;
			else
				mUseStepWarping = false;
		}
		else if (!index.compare("use_displacement"))
		{
			std::string str2;
			ss >> str2;
			if (!str2.compare("true"))
				mUseDisplacement = true;
			else
				mUseDisplacement = false;
		}
		else if (!index.compare("use_phase") && !mUseVelocity)
		{
			mUsePhase = true;
			ss >> mMinPhase;
			ss >> mMaxPhase;
		}
		else if (!index.compare("use_stride") && !mUseVelocity)
		{
			mUseStride = true;
			ss >> mMinStride;
			ss >> mMaxStride;
		}
		else if (!index.compare("use_sym"))
		{
			std::string str2;
			ss >> str2;
			if (!str2.compare("true"))
			{
				mIsMuscleSymMode = true;
				mIsTorqueSymMode = true;
			}
			else
			{
				mIsMuscleSymMode = false;
				mIsTorqueSymMode = false;
			}
		}
		else if (!index.compare("con_hz"))
		{
			int hz;
			ss >> hz;
			this->SetControlHz(hz);
		}
		else if (!index.compare("sim_hz"))
		{
			int hz;
			ss >> hz;
			this->SetSimulationHz(hz);
		}
		else if (!index.compare("skel_file"))
		{
			std::string str2;
			ss >> str2;
			mCharacter->LoadSkeleton(std::string(MASS_ROOT_DIR) + str2, mOriginalKp, load_obj);

			if (mIsRender)
			{
				mReferenceSkeleton = BuildFromFile(std::string(MASS_ROOT_DIR) + str2, true, Eigen::Vector4d(1, 0, 0, 0.1), false);
				mReferenceSkeleton->setName("ReferenceHuman");

				mBVHSkeleton = BuildFromFile(std::string(MASS_ROOT_DIR) + str2, false, Eigen::Vector4d(0, 1, 0, 0.1), false, true);
				mBVHSkeleton->setName("BVHHuman");
			}
			mCharacter->CheckMirrorPair(mCharacter->GetSkeleton());
		}
		else if (!index.compare("muscle_file"))
		{
			std::string str2;
			ss >> str2;
			mCharacter->LoadMuscles(std::string(MASS_ROOT_DIR) + str2);
		}
		else if (!index.compare("bvh_file"))
		{
			std::string str2, str3;

			ss >> str2 >> str3;
			bool cyclic = false;
			if (!str3.compare("true"))
				cyclic = true;
			mCharacter->LoadBVH(std::string(MASS_ROOT_DIR) + str2, cyclic);
		}
		else if (!index.compare("inference_per_sim"))
		{
			double a;
			ss >> a;
			mInferencePerSim = a;
		}
		else if (!index.compare("cascading_type"))
		{
			double a;
			ss >> a;
			mCascadingType = a;
		}
		else if (!index.compare("constant_pd_parameter"))
		{
			std::string str1;
			ss >> str1;
			if (!str1.compare("true"))
				mIsConstantPDParameter = true;
			else
				mIsConstantPDParameter = false;
		}
		else if (!index.compare("use_critical_damping"))
		{
			std::string str1;
			ss >> str1;
			if (!str1.compare("true"))
				mUseCriticalDamping = true;
			else
				mUseCriticalDamping = false;
		}
		else if (!index.compare("muscle_param") || !index.compare("muscle_length_param"))
		{
			while (!ifs.eof())
			{
				MuscleParam muscleparam_elem;

				std::string muscle_name;
				double min_ratio;
				double max_ratio;

				str.clear();
				ss.clear();

				std::getline(ifs, str);
				ss.str(str);

				ss >> muscle_name;
				if (!muscle_name.compare("group"))
				{
					muscleparam_elem.isGroup = true;
					ss >> muscle_name;
					muscleparam_elem.name = muscle_name;
				}
				else
					muscleparam_elem.isGroup = false;

				if (!muscle_name.compare("muscle_end") || !muscle_name.compare("muscle_length_end"))
					break;

				if (!muscle_name.compare("#"))
					continue;

				muscleparam_elem.name = muscle_name;

				ss >> min_ratio;
				ss >> max_ratio;

				muscleparam_elem.min_ratio = min_ratio;
				muscleparam_elem.max_ratio = max_ratio;
				muscleparam_elem.current_ratio = 1.0;

				bool isValid = false;

				for (int i = 0; i < mCharacter->GetMuscles().size() && mUseMuscle; i++)
				{
					auto &m = mCharacter->GetMuscles()[i];

					if (muscleparam_elem.Compare(m->name))
					{
						muscleparam_elem.muscle.push_back(m);
						mCharacter->AddChangedMuscleIdx(i);
						mCharacter->AddChangedMuscle(m);
						isValid = true;
						if (!muscleparam_elem.isGroup)
							break;
					}
				}
				if (isValid)
					mMuscleLengthParams.push_back(muscleparam_elem);
				else
					std::cout << "[Warning] WRONG INPUT AT MUSCLE PARAM" << muscleparam_elem.name << std::endl;
			}
		}
		else if (!index.compare("muscle_force_param"))
		{
			while (!ifs.eof())
			{
				MuscleParam muscleparam_elem;

				std::string muscle_name;
				double min_ratio;
				double max_ratio;

				str.clear();
				ss.clear();

				std::getline(ifs, str);
				ss.str(str);

				ss >> muscle_name;
				if (!muscle_name.compare("group"))
				{
					muscleparam_elem.isGroup = true;
					ss >> muscle_name;
					muscleparam_elem.name = muscle_name;
				}
				else
					muscleparam_elem.isGroup = false;

				if (!muscle_name.compare("muscle_force_end"))
					break;

				if (!muscle_name.compare("#"))
					continue;

				muscleparam_elem.name = muscle_name;

				ss >> min_ratio;
				ss >> max_ratio;

				muscleparam_elem.min_ratio = 0.05;
				muscleparam_elem.max_ratio = max_ratio;
				muscleparam_elem.current_ratio = 1.0;

				bool isValid = false;

				for (int i = 0; i < mCharacter->GetMuscles().size() && mUseMuscle; i++)
				{
					auto &m = mCharacter->GetMuscles()[i];

					if (muscleparam_elem.Compare(m->name))
					{
						muscleparam_elem.muscle.push_back(m);
						mCharacter->AddChangedMuscleIdx(i);
						mCharacter->AddChangedMuscle(m);
						isValid = true;
						if (!muscleparam_elem.isGroup)
							break;
					}
				}
				if (isValid)
					mMuscleForceParams.push_back(muscleparam_elem);
				else
					std::cout << "[Warning] WRONG INPUT AT MUSCLE FORCE" << std::endl;
			}
		}
		else if (!index.compare("skel_length_param"))
		{
			while (!ifs.eof())
			{
				SkelParam skelparam_elem;

				std::string skel_name;
				double min_ratio;
				double max_ratio;

				str.clear();
				ss.clear();

				std::getline(ifs, str);
				ss.str(str);

				ss >> skel_name;

				if (!skel_name.compare("skel_length_end"))
					break;

				if (!skel_name.compare("#"))
					continue;

				skelparam_elem.name = skel_name;

				ss >> min_ratio;
				ss >> max_ratio;

				skelparam_elem.min_ratio = min_ratio;
				skelparam_elem.max_ratio = max_ratio;
				skelparam_elem.current_ratio = 1.0;

				mSkelLengthParams.push_back(skelparam_elem);
			}
		}
		else if (!index.compare("use_constraint"))
		{
			std::string str1;
			ss >> str1;
			if (!str1.compare("true"))
				mUseConstraint = true;
			else
				mUseConstraint = false;
		}
		else if (!index.compare("skeleton_parameter_file"))
		{
			std::string str1;
			ss >> str1;
			mSkelInfos = Character::LoadSkelParamFile(std::string(MASS_ROOT_DIR) + str1);
			mCharacter->ModifySkeletonLength(mSkelInfos);
		}
		else if (!index.compare("simple_motion_file"))
		{
			std::string str1, str2;
			ss >> str1 >> str2;
			mCharacter->SetSimpleMotion(std::string(MASS_ROOT_DIR) + str1, std::string(MASS_ROOT_DIR) + str2);
		}
		else if (!index.compare("action_scale"))
		{
			double a;
			ss >> a;
			mActionScale = a;
		}
		else if (!index.compare("phase_scale"))
		{
			double a;
			ss >> a;
			mPhaseScale = a;
		}
		else if (!index.compare("use_imitation"))
		{
			std::string str1;
			ss >> str1;
			if (!str1.compare("true"))
				mUseImitation = true;
			else
				mUseImitation = false;
		}
		else if (!index.compare("use_newphase"))
		{
			std::string str1;
			ss >> str1;
			if (!str1.compare("true"))
				mIsNewPhase = true;
			else
				mIsNewPhase = false;
		}
	}

	if (mUseAdaptiveSampling && (GetParamState().size() == 0))
	{
		std::cout << "[Error] UseAdaptiveSampling needs parameters" << std::endl;
		exit(-1);
	}

	mGround = MASS::BuildFromFile(std::string(MASS_ROOT_DIR) + std::string("/data/ground.xml"));

	Initialize();

	mCharacter->GetSkeleton()->setPositions(mCharacter->GetSkeleton()->getPositions().setZero());
	mBodyHeight = (mCharacter->GetSkeleton()->getBodyNode("Head")->getCOM()[1] - 0.05) -
				  (mCharacter->GetSkeleton()->getBodyNode("TalusL")->getCOM()[1] - 0.025);
	mHeadHeight = 0.1;

	mOptWeight = mCharacter->GetSkeleton()->getPositions();
	mOptWeight.setOnes();

	for (auto j : mCharacter->GetSkeleton()->getJoints())
	{
		if (j->getNumDofs() == 3)
		{
			mOptWeight[j->getIndexInSkeleton(2)] *= 0.5;
			mOptWeight[j->getIndexInSkeleton(1)] *= 0.5;
		}
	}

	Reset();

	for (int i = 0; i < mCharacter->GetMuscles().size(); i += 2)
	{
		auto m1 = mCharacter->GetMuscles()[i];
		auto m2 = mCharacter->GetMuscles()[i + 1];

		if (std::abs(m1->f0_original - m2->f0_original) > 1E-4)
		{
			std::cout << m1->name << " " << m2->name << std::endl;
			exit(-1);
		}

		if (std::abs(m1->l_mt0_original - m2->l_mt0_original) > 1E-4)
		{
			std::cout << m1->name << " " << m2->name << std::endl;
			exit(-1);
		}
	}
}

double Environment::
	CalculateWeight(Eigen::VectorXd state_diff, double a)
{
	return 1.0 / (1.0 + exp(-(state_diff.norm() - a) * 200));
}

double Environment::
	UpdatePDParameter(Joint *jn, bool isFirst)
{
	BodyNode *bn = jn->getChildBodyNode();
	int bn_num = bn->getNumChildBodyNodes();
	double mass = jn->getChildBodyNode()->getMass();

	if (bn_num > 0)
		for (int i = 0; i < bn_num; i++)
			mass += UpdatePDParameter(bn->getChildBodyNode(i)->getParentJoint(), isFirst);

	if (isFirst)
		mRefMass[jn->getJointIndexInTree()] = mass;
	else
	{
		for (int i = 0; i < jn->getNumDofs(); i++)
		{
			mKp[jn->getIndexInSkeleton(0) + i] = mOriginalKp * mass / mRefMass[jn->getJointIndexInTree()];
			mKv[jn->getIndexInSkeleton(0) + i] = sqrt(2 * mKp[jn->getIndexInSkeleton(0) + i]);
		}
	}

	return mass;
}

void Environment::
	Initialize()
{

	mDesiredTorque.resize(mCharacter->GetSkeleton()->getNumDofs());
	mNetDesiredTorque.resize(mCharacter->GetSkeleton()->getNumDofs());
	mClipedDesiredTorque.resize(mCharacter->GetSkeleton()->getNumDofs());

	mNetDesiredTorque.setZero();
	mClipedDesiredTorque.setZero();

	mNumParamState = GetParamState().size();

	mCharacter->GetBVH()->SetPureMotions();
	mCharacter->SetMirrorMotion();
	mCharacter->GetBVH()->ResetModifiedMotions();

	mRootJointDof = mCharacter->GetSkeleton()->getRootBodyNode()->getParentJoint()->getNumDofs();
	mNumActiveDof = mCharacter->GetSkeleton()->getNumDofs() - mRootJointDof;
	mMuscleLengthParamNum = mMuscleLengthParams.size();
	mMuscleForceParamNum = mMuscleForceParams.size();
	mSkelLengthParamNum = mSkelLengthParams.size();

	if (mUseMuscle)
	{
		int num_total_related_dofs = 0;
		for (auto m : mCharacter->GetMuscles())
		{
			m->Update();
			num_total_related_dofs += m->GetNumRelatedDofs();
		}

		mCurrentMuscleTuple.JtA = Eigen::VectorXd::Zero(num_total_related_dofs);
		mCurrentMuscleTuple.Jtp = Eigen::VectorXd::Zero(mNumActiveDof);
		mCurrentMuscleTuple.L = Eigen::MatrixXd::Zero(mNumActiveDof, mCharacter->GetMuscles().size());
		mCurrentMuscleTuple.b = Eigen::VectorXd::Zero(mNumActiveDof);
		mCurrentMuscleTuple.tau_des = Eigen::VectorXd::Zero(mNumActiveDof);
		mActivationLevels = Eigen::VectorXd::Zero(mCharacter->GetMuscles().size());
		mExcitationLevels = Eigen::VectorXd::Zero(mCharacter->GetMuscles().size());

		if (mUseConstraint)
		{
			for (int i = 0; i < mCharacter->GetChangedMuscles().size(); i++)
			{
				Muscle *m = mCharacter->GetChangedMuscles()[i];
				m->set_l_mt_max(1.1);
				mCharacter->AddMuscleLimitConstraint(std::make_shared<MuscleLimitConstraint>(m));
			}
		}
	}
	mWorld->getConstraintSolver()->setCollisionDetector(dart::collision::BulletCollisionDetector::create());

	mWorld->setGravity(Eigen::Vector3d(0, -9.8, 0.0));
	mWorld->setTimeStep(1.0 / mSimulationHz);
	mWorld->addSkeleton(mCharacter->GetSkeleton());
	mWorld->addSkeleton(mGround);
	if (mIsRender)
	{
		mWorld->addSkeleton(mReferenceSkeleton);
		mWorld->addSkeleton(mBVHSkeleton);
	}
	mAction = Eigen::VectorXd::Zero(mNumActiveDof);
	if (mUseConstraint)
	{
		for (int i = 0; i < mCharacter->GetMuscleLimitConstraints().size(); i++)
			mWorld->getConstraintSolver()->addConstraint(mCharacter->GetMuscleLimitConstraints()[i]);
	}

	mCharacter->GetSkeleton()->clearConstraintImpulses();
	mCharacter->GetSkeleton()->clearInternalForces();
	mCharacter->GetSkeleton()->clearExternalForces();
	mWorld->reset();
	mCharacter->Reset();

	mTargetPositions = mCharacter->GetSkeleton()->getPositions();

	mNumState = GetState().rows();
	if (mUseMuscle)
		mNumMuscleState = GetMuscleState().rows();
	else
		mNumMuscleState = 0;

	mCharacter->GetSkeleton()->setSelfCollisionCheck(mSelfCollision);
	Reset();

	for (int i = 0; i < mCharacter->GetMuscles().size(); i += 2)
	{
		auto &m1 = mCharacter->GetMuscles()[i];
		auto &m2 = mCharacter->GetMuscles()[i + 1];
		if ((m1->l_m0 != m2->l_m0) ||
			(m1->l_t0 != m2->l_t0))
			std::cout << "[DEBUG] Warning " << m1->name << " " << m2->name << std::endl;
	}
}

void Environment::
	Reset()
{
	double t = 0;
	mIsComplete = false;
	mWorld->reset();

	mCharacter->GetSkeleton()->clearConstraintImpulses();
	mCharacter->GetSkeleton()->clearInternalForces();
	mCharacter->GetSkeleton()->clearExternalForces();

	double phase_state = (dart::math::Random::uniform(0.0, 1.0) < 0.5 ? 0.1 : 0.6) + dart::math::Random::uniform(-0.05, 0.05);

	t = phase_state * mCharacter->GetBVH()->GetMaxTime();
	double m = std::fmod(t, mCharacter->GetBVH()->GetTimeStep());
	t -= m;

	mLocalTime = t;
	mGlobalTime = mLocalTime;
	mStartPhase = GetPhase();

	mWorld->setTime(t);
	mCharacter->Reset();
	mAction.setZero();
	mEoeTime = t;

	mActivationLevels.setZero();
	mExcitationLevels.setZero();

	for (auto muscle : mCharacter->GetMuscles())
	{
		muscle->activation = 0;
		muscle->Update();
	}

	mWorld->getConstraintSolver()->setCollisionDetector(dart::collision::BulletCollisionDetector::create());
	mWorld->getConstraintSolver()->clearLastCollisionResult();
	std::pair<Eigen::VectorXd, Eigen::VectorXd> pv = mCharacter->GetTargetPosAndVel(t, mPhaseRatio * sqrt(1 / mGlobalRatio) / mControlHz);
	mTargetPositions = pv.first;
	mTargetVelocities = pv.second;

	Eigen::VectorXd cur_pos = mTargetPositions;
	Eigen::VectorXd cur_vel = mTargetVelocities;

	double mRefVel = GetTargetVelocity();

	cur_pos.head(24) *= mStrideRatio;
	cur_vel.head(24) *= mStrideRatio;

	cur_vel.head(6).setZero();
	cur_vel[5] = mRefVel;

	if (IsArmExist() && mActionType == 1)
	{
		cur_pos.segment(mCharacter->GetSkeleton()->getJoint("ArmL")->getIndexInSkeleton(0), 3) = Eigen::Vector3d(0, 0, -M_PI * 0.45);
		cur_pos.segment(mCharacter->GetSkeleton()->getJoint("ArmR")->getIndexInSkeleton(0), 3) = Eigen::Vector3d(0, 0, M_PI * 0.45);
	}

	cur_pos[0] = 0;
	cur_pos[1] = 0;
	cur_pos[3] = 0.0;
	cur_pos[4] = -1.0;

	mCharacter->GetSkeleton()->setPositions(cur_pos);
	mCharacter->GetSkeleton()->setVelocities(cur_vel);
	mCharacter->GetSkeleton()->computeForwardKinematics(true, true, false);

	NaivePoseOptimization(100);

	if (mStateType == 4 || mStateType == 6)
	{
		cur_pos = mCharacter->GetSkeleton()->getPositions();
		cur_pos[3] -= mCharacter->GetSkeleton()->getCOM()[0];
		mCharacter->GetSkeleton()->setPositions(cur_pos);
		mCharacter->GetSkeleton()->computeForwardKinematics(true, true, false);
	}

	if (mIsRender)
	{
		std::pair<Eigen::VectorXd, Eigen::VectorXd> bvh_pv = mCharacter->GetTargetPosAndVel(t, mPhaseRatio * sqrt(1 / mGlobalRatio) / mControlHz, mBVHSkeleton);
		mBVHPositions = bvh_pv.first;
		mBVHVelocities = bvh_pv.second;

		mReferenceSkeleton->setPositions(mTargetPositions);
		mReferenceSkeleton->setVelocities(mTargetVelocities);
		mReferenceSkeleton->computeForwardKinematics(true, true, false);

		mBVHSkeleton->setPositions(mBVHPositions);
		mBVHSkeleton->setVelocities(mBVHVelocities);
		mBVHSkeleton->computeForwardKinematics(true, true, false);
	}

	mAverageReward = 0;
	mUsedTorque = 0.0;
	mNetUsedTorque = 0.0;

	mSimUsedTorque = 0.0;
	mNetSimUsedTorque = 0.0;

	mTotalAcc = 0.0;

	mStepCount = 1;
	mCycleCount = 1;

	mActivationBuf.clear();
	mActiveForce = 0;

	mPassiveForce = 0;

	mOverTorque = 0;

	mPrevTime = mWorld->getTime();
	mGlobalPrevIdx = 0;

	mComTrajectory.clear();

	double phase = GetPhase();
	if (phase >= 0.81 || phase < 0.31)
	{
		mCurrentStance = 0;
		if (mCharacter->GetSkeleton()->getBodyNode("HeelL") == NULL)
			mCurrentFoot = mCharacter->GetSkeleton()->getBodyNode("TalusL")->getCOM();
		else
			mCurrentFoot = mCharacter->GetSkeleton()->getBodyNode("HeelL")->getCOM();
		mCurrentFoot[1] = 0;
		mCurrentTargetFoot = mCurrentFoot;
		mNextTargetFoot = mCurrentFoot;
		mNextTargetFoot[2] += (mStrideRatio * mRefStride * 0.5) * mGlobalRatio;
		mNextTargetFoot[0] *= -1;
	}
	else
	{
		mCurrentStance = 1;

		if (mCharacter->GetSkeleton()->getBodyNode("HeelR") == NULL)
			mCurrentFoot = mCharacter->GetSkeleton()->getBodyNode("TalusR")->getCOM();
		else
			mCurrentFoot = mCharacter->GetSkeleton()->getBodyNode("HeelR")->getCOM();
		mCurrentFoot[1] = 0;
		mCurrentTargetFoot = mCurrentFoot;
		mNextTargetFoot = mCurrentFoot;
		mNextTargetFoot[2] += (mStrideRatio * mRefStride * 0.5) * mGlobalRatio;
		mNextTargetFoot[0] *= -1;
	}

	mPhaseDisplacement = 0;
	mStepDisplacement = 0;

	mMetabolicEnergy = 0.0;
	mMass = mCharacter->GetSkeleton()->getMass();
	mPrevCOM = mCharacter->GetSkeleton()->getCOM();
	mCurCOM = mCharacter->GetSkeleton()->getCOM();
	mIsNew = true;
	UpdateHeadInfo();

	mPrevPhase = GetGlobalPhase();
}

void Environment::
	SetMuscleLengthParamState(Eigen::VectorXd ParamState)
{
	int idx = 0;
	for (int i = 0; i < mMuscleLengthParams.size(); i++)
		mMuscleLengthParams[i].current_ratio = ParamState[idx++];
}

void Environment::
	SetMuscleForceParamState(Eigen::VectorXd ParamState)
{
	int idx = 0;
	for (int i = 0; i < mMuscleForceParams.size(); i++)
		mMuscleForceParams[i].current_ratio = ParamState[idx++];
}

void Environment::
	SetSkelLengthParamState(Eigen::VectorXd ParamState)
{
	int idx = 0;
	for (int i = 0; i < mSkelLengthParams.size(); i++)
		mSkelLengthParams[i].current_ratio = ParamState[idx++];
}

void Environment::
	SetParamState(Eigen::VectorXd ParamState)
{
	int idx = 0;

	idx += mMuscleLengthParamNum + mMuscleForceParamNum;

	if (mSkelLengthParamNum > 0)
		SetSkelLengthParamState(ParamState.segment(idx, mSkelLengthParamNum));
	ApplySkelParameter();

	idx = 0;

	if (mMuscleLengthParamNum > 0)
	{
		SetMuscleLengthParamState(ParamState.segment(idx, mMuscleLengthParamNum));
		idx += mMuscleLengthParamNum;
	}

	if (mMuscleForceParamNum > 0)
	{
		SetMuscleForceParamState(ParamState.segment(idx, mMuscleForceParamNum));
		idx += mMuscleForceParamNum;
	}

	ApplyMuscleParameter();

	idx += mSkelLengthParamNum;

	if (mUseVelocity)
		mTargetCOMVelocity = ParamState[idx++];

	if (mUsePhase)
	{
		mPhaseRatio = ParamState[idx++];
		if (mPhaseRatio < 1E-8)
			mPhaseRatio = 1E-8;
	}

	if (mUseStride)
		mStrideRatio = ParamState[idx++];

	mWorld->getConstraintSolver()->setCollisionDetector(dart::collision::BulletCollisionDetector::create());

	if (!mIsConstantPDParameter)
	{
		UpdatePDParameter(mCharacter->GetSkeleton()->getRootJoint());
		mCharacter->SetPDParameters(mKp, mKv);
	}
}

MuscleTuple &
Environment::
	GetMuscleTuple(bool isRender)
{
	GetDesiredTorques();

	Eigen::VectorXd mDesiredTorque_bkup = mDesiredTorque;
	Eigen::VectorXd pos_bkup = mCharacter->GetSkeleton()->getPositions();
	Eigen::VectorXd vel_bkup = mCharacter->GetSkeleton()->getVelocities();

	if (GetPhase() > 0.5 && mIsMuscleSymMode && !isRender)
		mDesiredTorque = mCharacter->GetMirrorPosition(mDesiredTorque_bkup);

	int n = mCharacter->GetSkeleton()->getNumDofs();
	int m = mCharacter->GetMuscles().size();

	mCurrentMuscleTuple.JtA.setZero();
	mCurrentMuscleTuple.Jtp.setZero();
	mCurrentMuscleTuple.L.setZero();
	mCurrentMuscleTuple.b.setZero();
	mCurrentMuscleTuple.tau_des.setZero();

	Eigen::MatrixXd JtA = Eigen::MatrixXd::Zero(n, m);
	Eigen::VectorXd Jtp = Eigen::VectorXd::Zero(n);

	int i = 0;
	int index = 0;
	for (auto &muscle : mCharacter->GetMuscles())
	{
		muscle->Update();
		Eigen::MatrixXd Jt_reduced = muscle->GetReducedJacobianTranspose();
		auto Ap = muscle->GetForceJacobianAndPassive();
		Eigen::VectorXd JtA_reduced = Jt_reduced * Ap.first;
		Eigen::VectorXd JtP_reduced = Jt_reduced * Ap.second;

		for (int j = 0; j < muscle->GetNumRelatedDofs(); j++)
		{
			Jtp[muscle->related_dof_indices[j]] += JtP_reduced[j];
			JtA(muscle->related_dof_indices[j], i) = JtA_reduced[j];
		}
		mCurrentMuscleTuple.JtA.segment(index, JtA_reduced.rows()) = JtA_reduced;
		index += JtA_reduced.rows();
		i++;
	}

	if (GetPhase() > 0.5 && mIsMuscleSymMode && !isRender)
	{
		for (int i = 0; i < mCharacter->GetMuscles().size(); i += 2)
		{
			Eigen::VectorXd tmp = JtA.col(i);
			JtA.col(i) = mCharacter->GetMirrorPosition(JtA.col(i + 1));
			JtA.col(i + 1) = mCharacter->GetMirrorPosition(tmp);
		}
		int i = 0;
		int index = 0;
		for (auto &muscle : mCharacter->GetMuscles())
		{
			Eigen::VectorXd JtA_reduced = Eigen::VectorXd::Ones(muscle->GetNumRelatedDofs());
			for (int j = 0; j < muscle->GetNumRelatedDofs(); j++)
				JtA_reduced[j] = JtA(muscle->related_dof_indices[j], i);

			mCurrentMuscleTuple.JtA.segment(index, JtA_reduced.rows()) = JtA_reduced;
			index += JtA_reduced.rows();
			i++;
		}

		Jtp = mCharacter->GetMirrorPosition(Jtp);
	}

	if (mIsTorqueClip)
	{
		Eigen::VectorXd max_tau = Eigen::VectorXd::Zero(n);
		Eigen::VectorXd min_tau = Eigen::VectorXd::Zero(n);

		for (int i = mRootJointDof; i < n; i++)
		{
			for (int j = 0; j < m; j++)
				if (JtA(i, j) > 0)
					max_tau[i] += JtA(i, j);
				else
					min_tau[i] += JtA(i, j);

			mClipedDesiredTorque[i] = dart::math::clip(mDesiredTorque[i], min_tau[i] + Jtp[i], max_tau[i] + Jtp[i]);
			mOverTorque += pow(std::abs(mDesiredTorque[i] - mClipedDesiredTorque[i]), 2);
			mNetDesiredTorque[i] = mDesiredTorque[i] - Jtp[i];
		}

		mCurrentMuscleTuple.tau_des = mClipedDesiredTorque.tail(mClipedDesiredTorque.rows() - mRootJointDof);
	}
	else
		mCurrentMuscleTuple.tau_des = mDesiredTorque.tail(mDesiredTorque.rows() - mRootJointDof);

	mCurrentMuscleTuple.Jtp = Jtp.segment(mRootJointDof, n - mRootJointDof);

	mCurrentMuscleTuple.L = JtA.block(mRootJointDof, 0, n - mRootJointDof, m);
	mCurrentMuscleTuple.b = Jtp.segment(mRootJointDof, n - mRootJointDof);

	if (GetPhase() > 0.5 && mIsMuscleSymMode && !isRender)
		mDesiredTorque = mDesiredTorque_bkup;

	return mCurrentMuscleTuple;
}

Eigen::VectorXd
Environment::
	GetDerivationOfActivation(Eigen::VectorXd a, Eigen::VectorXd e)
{
	double t_act = 0.01;
	double t_deact = 0.04;

	Eigen::VectorXd result = Eigen::VectorXd::Ones(a.rows());

	if (a.rows() != e.rows())
		exit(-1);

	for (int i = 0; i < a.rows(); i++)
	{
		double tau = 0;
		if (e[i] > a[i])
			tau = t_act * (0.5 + 1.5 * a[i]);
		else
			tau = t_deact / (0.5 + 1.5 * a[i]);

		result[i] = (e[i] - a[i]) / tau;
	}

	return result;
}

void Environment::
	Step()
{
	mIsNew = false;
	if (mIsRender)
	{
		mReferenceSkeleton->setPositions(mTargetPositions);
		mReferenceSkeleton->setVelocities(mTargetVelocities);
		mReferenceSkeleton->computeForwardKinematics(true, true, false);

		mBVHSkeleton->setPositions(mBVHPositions);
		mBVHSkeleton->setVelocities(mBVHVelocities);
		mBVHSkeleton->computeForwardKinematics(true, true, false);
	}

	if (mUseMuscle)
	{
		int count = 0;
		mActivationBuf.push_back(mActivationLevels);

		for (auto muscle : mCharacter->GetMuscles())
		{
			muscle->activation = mActivationLevels[count++];
			muscle->Update();
			muscle->ApplyForceToBody();

			mActiveForce += pow(muscle->GetForce(), 2);
			mPassiveForce += pow(muscle->Getf_p(), 2);
		}
		for (auto j : mCharacter->GetSkeleton()->getJoints())
		{
			if (j->getNumDofs() == 0)
				continue;
			double torque = mDesiredTorque.segment(j->getIndexInSkeleton(0), j->getNumDofs()).norm();
			double net_torque = mNetDesiredTorque.segment(j->getIndexInSkeleton(0), j->getNumDofs()).norm();

			mUsedTorque += torque;
			mNetUsedTorque += net_torque;
		}

		if (mMetabolicType == 5 || mMetabolicType == 6)
		{
			auto mt = GetMuscleTuple();
			Eigen::VectorXd net_sim_t = mt.L * mActivationLevels;
			Eigen::VectorXd sim_t = net_sim_t + mt.b;
			for (auto j : mCharacter->GetSkeleton()->getJoints())
			{
				if (j->getNumDofs() == 0)
					continue;
				double torque = sim_t.segment(j->getIndexInSkeleton(0) - mRootJointDof, j->getNumDofs()).norm();
				double net_torque = net_sim_t.segment(j->getIndexInSkeleton(0) - mRootJointDof, j->getNumDofs()).norm();

				mSimUsedTorque += torque;
				mNetSimUsedTorque += net_torque;
			}
		}
	}
	else
	{
		GetDesiredTorques();
		mDesiredTorque.head(6).setZero();

		for (auto j : mCharacter->GetSkeleton()->getJoints())
		{
			if (j->getNumDofs() == 0)
				continue;
			double torque = mDesiredTorque.segment(j->getIndexInSkeleton(0), j->getNumDofs()).norm();
			mUsedTorque += torque;
			mNetUsedTorque += torque;
		}

		mCharacter->GetSkeleton()->setForces(mDesiredTorque);
	}

	Eigen::VectorXd acc = mCharacter->GetSkeleton()->getAccelerations();
	for (int i = 6; i < acc.size(); i++)
		mTotalAcc += pow(acc[i], 2);

	mWorld->step();

	if (mUseMuscle && mMetabolicType == 3)
	{
		for (auto m : mCharacter->GetMuscles())
		{
			double e = m->GetBHAR04_EnergyRate();
			mMetabolicEnergy += e * 1.0 / mSimulationHz;
		}
	}

	if (mUseExcitation)
		mActivationLevels += GetDerivationOfActivation(mActivationLevels, mExcitationLevels) * (1.0 / mSimulationHz);

	std::vector<dart::dynamics::BodyNode *> left_foot;
	std::vector<dart::dynamics::BodyNode *> right_foot;

	left_foot.push_back(mCharacter->GetSkeleton()->getBodyNode("FootPinkyL"));
	left_foot.push_back(mCharacter->GetSkeleton()->getBodyNode("FootThumbL"));

	right_foot.push_back(mCharacter->GetSkeleton()->getBodyNode("FootPinkyR"));
	right_foot.push_back(mCharacter->GetSkeleton()->getBodyNode("FootThumbR"));

	double phase = GetPhase();

	mComTrajectory.push_back(mCharacter->GetSkeleton()->getCOM());
}

Eigen::VectorXd
Environment::
	GetDesiredTorques()
{

	Eigen::VectorXd p_des = mTargetPositions;

	if (mActionType == 0)
	{
		if (IsArmExist())
		{
		}
	}
	else if (mActionType == 1)
	{
		p_des.setZero();
		if (IsArmExist())
		{
			p_des[mCharacter->GetSkeleton()->getJoint("ArmL")->getIndexInSkeleton(2)] = -M_PI / 2;
			p_des[mCharacter->GetSkeleton()->getJoint("ArmR")->getIndexInSkeleton(2)] = M_PI / 2;
		}
	}
	else if (mActionType == 2)
		p_des = mCharacter->GetSkeleton()->getPositions();

	p_des.tail(mTargetPositions.rows() - mRootJointDof) += mAction;

	if (mActionType == 2)
		mDesiredTorque = mCharacter->GetSPDForces(p_des, false);
	else
		mDesiredTorque = mCharacter->GetSPDForces(p_des);

	return mDesiredTorque.tail(mDesiredTorque.rows() - mRootJointDof);
}

Eigen::VectorXd
Environment::
	GetMuscleTorques()
{
	int index = 0;
	mCurrentMuscleTuple.JtA.setZero();
	for (auto muscle : mCharacter->GetMuscles())
	{
		muscle->Update();
		Eigen::VectorXd JtA_i = muscle->GetRelatedJtA();
		mCurrentMuscleTuple.JtA.segment(index, JtA_i.rows()) = JtA_i;
		index += JtA_i.rows();
	}

	return mCurrentMuscleTuple.JtA;
}

Eigen::VectorXd
Environment::
	GetPassiveMuscleTorques()
{
	int n = mCharacter->GetSkeleton()->getNumDofs();
	Eigen::VectorXd Jtp = Eigen::VectorXd::Zero(n);
	mCurrentMuscleTuple.Jtp.setZero();
	for (auto &muscle : mCharacter->GetMuscles())
	{
		Eigen::MatrixXd Jt = muscle->GetJacobianTranspose();
		auto Ap = muscle->GetForceJacobianAndPassive();
		Jtp += Jt * Ap.second;
	}

	mCurrentMuscleTuple.Jtp = Jtp.segment(mRootJointDof, n - mRootJointDof);
	return mCurrentMuscleTuple.Jtp;
}

double exp_of_squared(const Eigen::VectorXd &vec, double w)
{
	return exp(-w * vec.squaredNorm() / vec.rows());
}
double exp_of_squared(const Eigen::Vector3d &vec, double w)
{
	return exp(-w * vec.squaredNorm() / vec.rows());
}
double exp_of_squared(double val, double w)
{
	return exp(-w * val * val);
}

double exp_of_L1_norm(const Eigen::VectorXd &vec, double w)
{
	return exp(-w * vec.lpNorm<1>());
}
double exp_of_L1_norm(const Eigen::Vector3d &vec, double w)
{
	return exp(-w * vec.lpNorm<1>());
}
double exp_of_L1_norm(double val, double w)
{
	return exp(-w * std::abs(val));
}

int Environment::
	IsEndOfEpisode()
{
	int isTerminal = 0;

	Eigen::VectorXd p = mCharacter->GetSkeleton()->getPositions();
	Eigen::VectorXd v = mCharacter->GetSkeleton()->getVelocities();

	double root_y = mCharacter->GetSkeleton()->getCOM()[1];

	double limit = 0.6 * mGlobalRatio;

	bool isSelfCollision = false;

	if (root_y < limit || isSelfCollision)
		isTerminal = 1;
	else if (dart::math::isNan(p) || dart::math::isNan(v))
		isTerminal = 2;
	else if (mWorld->getTime() > 10.0)
		isTerminal = 3;

	return isTerminal;
}

Eigen::VectorXd
Environment::
	GetMuscleState(bool isRender)
{

	MuscleTuple mt = GetMuscleTuple(isRender);
	Eigen::VectorXd passive_f = mt.Jtp;
	Eigen::VectorXd muscleState;

	if (mUseContractileState)
	{
		muscleState = Eigen::VectorXd::Zero(passive_f.size() * 3);
		Eigen::VectorXd min_tau = Eigen::VectorXd::Zero(passive_f.size());
		Eigen::VectorXd max_tau = Eigen::VectorXd::Zero(passive_f.size());

		for (int i = 0; i < passive_f.size(); i++)
		{
			for (int j = 0; j < mCharacter->GetMuscles().size(); j++)
			{
				if (mt.L(i, j) > 0)
					max_tau[i] += mt.L(i, j);
				else
					min_tau[i] += mt.L(i, j);
			}
		}
		muscleState << passive_f, 0.5 * min_tau, 0.5 * max_tau;
	}
	else
		muscleState = passive_f;

	return muscleState;
}

Eigen::VectorXd
Environment::
	GetProjState(Eigen::VectorXd minv, Eigen::VectorXd maxv)
{

	Eigen::VectorXd full_muscle = getAllMuscleParam();
	Eigen::VectorXd param = GetParamState();
	Eigen::VectorXd v = param;
	int muscle_param_num = mMuscleLengthParams.size() + mMuscleForceParams.size();
	int projected_param_num = 0;
	int state_num = 0;
	for (int i = 0; i < param.rows(); i++)
	{
		v[i] = dart::math::clip(param[i], minv[i], maxv[i]);
		projected_param_num += (abs(maxv[i] - minv[i]) < 1E-6 ? 1 : 0);
	}
	projected_param_num = muscle_param_num - projected_param_num;

	SetParamState(v);
	Eigen::VectorXd state = GetState();
	SetParamState(param);
	setAllMuscleParam(full_muscle);

	if (mStateType == 11)
	{
		state_num = state.rows();
		Eigen::VectorXd proj_state = Eigen::VectorXd::Zero(state_num - muscle_param_num + projected_param_num);
		Eigen::VectorXd proj_muscle_param = Eigen::VectorXd(projected_param_num);
		int idx = 0;
		for (int i = 0; i < muscle_param_num; i += 2)
		{
			if (abs(maxv[i] - minv[i]) > 1E-6)
			{
				if (0.5 < GetPhase() && mIsTorqueSymMode)
				{
					proj_muscle_param[idx++] = param[i + 1];
					proj_muscle_param[idx++] = param[i];
				}
				else
				{
					proj_muscle_param[idx++] = param[i];
					proj_muscle_param[idx++] = param[i + 1];
				}
			}
		}
		if (idx != projected_param_num)
			exit(-1);

		proj_state.head(state_num - muscle_param_num) = state.head(state_num - muscle_param_num);
		proj_state.tail(projected_param_num) = proj_muscle_param;
		return proj_state;
	}
	else
		return state;
}

Eigen::VectorXd
Environment::
	GetState()
{
	Eigen::VectorXd state;
	int idx = 0;
	if (mStateType == 10)
	{
		auto &skel = mCharacter->GetSkeleton();
		dart::dynamics::BodyNode *root = skel->getBodyNode(0);

		double h = skel->getCOM()[1] + (mStateType == 9 ? 1 : -1) * 0.98;

		int num_body_nodes = skel->getNumBodyNodes();

		Eigen::VectorXd p, v;

		p.resize(num_body_nodes * 3 + num_body_nodes * 6);
		v.resize((num_body_nodes + 1) * 3 + num_body_nodes * 3);

		for (int i = 0; i < num_body_nodes; i++)
		{
			p.segment<3>(i * 3) = skel->getBodyNode(i)->getCOM() - skel->getCOM() - Eigen::Vector3d(0.0, -0.98, 0.0);

			Eigen::Isometry3d transform = skel->getBodyNode(i)->getTransform();
			p.segment<6>(num_body_nodes * 3 + 6 * i) << transform.linear()(0, 0), transform.linear()(0, 1), transform.linear()(0, 2),
				transform.linear()(1, 0), transform.linear()(1, 1), transform.linear()(1, 2);

			v.segment<3>(i * 3) = skel->getBodyNode(i)->getCOMLinearVelocity() - skel->getCOMLinearVelocity();

			v.segment<3>((num_body_nodes + 1) * 3 + i * 3) = 0.2 * skel->getBodyNode(i)->getAngularVelocity();
		}
		v.segment<3>(num_body_nodes * 3) = skel->getCOMLinearVelocity();

		double phi = GetPhase();

		Eigen::VectorXd phi_state;

		if (mPhaseType == 0)
		{
			phi_state = Eigen::VectorXd::Zero(2);
			phi_state[0] = cos(phi * 2 * M_PI);
			phi_state[1] = sin(phi * 2 * M_PI);
		}
		else if (mPhaseType == 1)
		{
			phi_state = Eigen::VectorXd::Zero(1);
			phi_state[0] = phi;
		}
		Eigen::VectorXd muscleState;
		if (mUseMuscle)
			muscleState = 0.008 * GetMuscleState();
		state = Eigen::VectorXd::Zero(1 + p.rows() + v.rows() + phi_state.rows() + (mUseMuscle ? muscleState.rows() : 0) + mSkelLengthParams.size() + (mUseVelocity ? 1 : 0) + (mUsePhase ? 1 : 0) + (mUseStride ? 1 + 3 : 0) + (mUseTimeWarping ? 1 : 0));
		int len_tmp = (mUseMuscle ? muscleState.rows() : 0) + mSkelLengthParams.size() + (mUseVelocity ? 1 : 0) + (mUsePhase ? 1 : 0) + (mUseStride ? 1 + 3 : 0) + (mUseTimeWarping ? 1 : 0);
		state << h, p, v, phi_state, Eigen::VectorXd::Zero(len_tmp);
		idx += 1 + p.rows() + v.rows() + phi_state.rows();
		mMuscleStateStartIdx = 1 + p.rows() + v.rows() + phi_state.rows();
		if (mUseMuscle)
		{
			state.segment(idx, muscleState.rows()) = muscleState;
			idx += muscleState.rows();
		}
		for (auto skel : mSkelLengthParams)
			state[idx++] = skel.current_ratio;

		if (mUseVelocity)
			state[idx++] = mTargetCOMVelocity;
		if (mUsePhase)
			state[idx++] = mPhaseRatio;
		if (mUseStride)
			state[idx++] = (mStrideRatio * mRefStride / mCharacter->GetBVH()->GetMaxTime()) * mPhaseRatio;
		if (mUseTimeWarping)
			state[idx++] = GetGlobalPhase();
		if (mUseStride)
		{
			state.segment(idx, 3) = mNextTargetFoot - mCharacter->GetSkeleton()->getRootBodyNode()->getCOM();
			state[idx] = 0.0;
			idx += 3;
		}
	}
	else if (mStateType == 11)
	{
		auto &skel = mCharacter->GetSkeleton();
		dart::dynamics::BodyNode *root = skel->getBodyNode(0);

		double h = skel->getCOM()[1] + (mStateType == 9 ? 1 : -1) * 0.98;

		int num_body_nodes = skel->getNumBodyNodes();

		Eigen::VectorXd p, v;

		p.resize(num_body_nodes * 3 + num_body_nodes * 6);
		v.resize((num_body_nodes + 1) * 3 + num_body_nodes * 3);

		for (int i = 0; i < num_body_nodes; i++)
		{
			p.segment<3>(i * 3) = skel->getBodyNode(i)->getCOM() - skel->getCOM() - Eigen::Vector3d(0.0, -0.98, 0.0);

			Eigen::Isometry3d transform = skel->getBodyNode(i)->getTransform();
			p.segment<6>(num_body_nodes * 3 + 6 * i) << transform.linear()(0, 0), transform.linear()(0, 1), transform.linear()(0, 2),
				transform.linear()(1, 0), transform.linear()(1, 1), transform.linear()(1, 2);

			v.segment<3>(i * 3) = skel->getBodyNode(i)->getCOMLinearVelocity() - skel->getCOMLinearVelocity();

			v.segment<3>((num_body_nodes + 1) * 3 + i * 3) = 0.2 * skel->getBodyNode(i)->getAngularVelocity();
		}
		v.segment<3>(num_body_nodes * 3) = skel->getCOMLinearVelocity();

		double phi = GetPhase();

		Eigen::VectorXd phi_state;

		if (mPhaseType == 0)
		{
			phi_state = Eigen::VectorXd::Zero(2);
			phi_state[0] = cos(phi * 2 * M_PI);
			phi_state[1] = sin(phi * 2 * M_PI);
		}
		else if (mPhaseType == 1)
		{
			phi_state = Eigen::VectorXd::Zero(1);
			phi_state[0] = phi;
		}
		Eigen::VectorXd muscleState;
		if (mUseMuscle)
			muscleState = 0.008 * GetMuscleState();
		state = Eigen::VectorXd::Zero(1 + p.rows() + v.rows() + phi_state.rows() + (mUseMuscle ? muscleState.rows() : 0) + mSkelLengthParams.size() + (mUseVelocity ? 1 : 0) + (mUsePhase ? 1 : 0) + (mUseStride ? 1 + 3 : 0) + (mUseTimeWarping ? 1 : 0) + mMuscleLengthParams.size() + mMuscleForceParams.size());
		int len_tmp = (mUseMuscle ? muscleState.rows() : 0) + mSkelLengthParams.size() + (mUseVelocity ? 1 : 0) + (mUsePhase ? 1 : 0) + (mUseStride ? 1 + 3 : 0) + (mUseTimeWarping ? 1 : 0) + mMuscleLengthParams.size() + mMuscleForceParams.size();
		state << h, p, v, phi_state, Eigen::VectorXd::Zero(len_tmp);
		idx += 1 + p.rows() + v.rows() + phi_state.rows();

		if (mUseMuscle)
		{
			state.segment(idx, muscleState.rows()) = muscleState;
			idx += muscleState.rows();
		}
		for (auto skel : mSkelLengthParams)
			state[idx++] = skel.current_ratio;

		if (mUseVelocity)
			state[idx++] = mTargetCOMVelocity;
		if (mUsePhase)
			state[idx++] = mPhaseRatio;
		if (mUseStride)
			state[idx++] = (mStrideRatio * mRefStride / mCharacter->GetBVH()->GetMaxTime()) * mPhaseRatio;
		if (mUseTimeWarping)
			state[idx++] = GetGlobalPhase();
		if (mUseStride)
		{
			state.segment(idx, 3) = mNextTargetFoot - mCharacter->GetSkeleton()->getRootBodyNode()->getCOM();
			state[idx] = 0.0;
			idx += 3;
		}
		for (int i = 0; i < mMuscleLengthParams.size(); i++)
			state[idx++] = mMuscleLengthParams[i].current_ratio;
		for (int i = 0; i < mMuscleForceParams.size(); i++)
			state[idx++] = mMuscleForceParams[i].current_ratio;
	}

	if (idx != state.rows())
	{
		std::cout << "[Warning] GetState " << std::endl;
		exit(-1);
	}
	if (0.5 < GetPhase() && mIsTorqueSymMode)
		state = GetMirrorState(state);

	return state;
}

Eigen::VectorXd
Environment::
	GetMuscleLengthParamState()
{
	Eigen::VectorXd muscle_length_param(mMuscleLengthParams.size());
	for (int i = 0; i < muscle_length_param.size(); i++)
		muscle_length_param[i] = mMuscleLengthParams[i].current_ratio;
	return muscle_length_param;
}

Eigen::VectorXd
Environment::
	GetMuscleForceParamState()
{
	Eigen::VectorXd muscle_force_param(mMuscleForceParams.size());
	for (int i = 0; i < muscle_force_param.size(); i++)
		muscle_force_param[i] = mMuscleForceParams[i].current_ratio;
	return muscle_force_param;
}

Eigen::VectorXd
Environment::
	GetSkelLengthParamState()
{
	Eigen::VectorXd skel_length_param(mSkelLengthParams.size());
	for (int i = 0; i < skel_length_param.size(); i++)
		skel_length_param[i] = mSkelLengthParams[i].current_ratio;
	return skel_length_param;
}

Eigen::VectorXd
Environment::
	GetParamState()
{
	Eigen::VectorXd muscle_length_param = GetMuscleLengthParamState();
	Eigen::VectorXd muscle_force_param = GetMuscleForceParamState();
	Eigen::VectorXd skel_length_param = GetSkelLengthParamState();
	Eigen::VectorXd result(muscle_length_param.rows() + muscle_force_param.rows() + skel_length_param.rows() + (mUseVelocity ? 1 : 0) + (mUsePhase ? 1 : 0) + (mUseStride ? 1 : 0));
	int idx = 0;

	result << muscle_length_param, muscle_force_param, skel_length_param, Eigen::VectorXd::Zero((mUseVelocity ? 1 : 0) + (mUsePhase ? 1 : 0) + (mUseStride ? 1 : 0));
	idx += muscle_length_param.rows() + muscle_force_param.rows() + skel_length_param.rows();

	if (mUseVelocity)
		result[idx++] = mTargetCOMVelocity;

	if (mUsePhase)
		result[idx++] = mPhaseRatio;

	if (mUseStride)
		result[idx++] = mStrideRatio;

	return result;
}

void Environment::
	CreateTotalParams()
{
	mTotalParams.clear();

	if (mMuscleLengthParams.size() > 0)
	{
		ParamCategory pc;
		pc.name = "Muscle Length";
		pc.params.clear();

		for (auto m : mMuscleLengthParams)
		{
			ParamElem pe;
			pe.name = m.name;
			pe.cur_v = m.current_ratio;
			pe.max_v = m.max_ratio;
			pe.min_v = m.min_ratio;
			pc.params.push_back(pe);
		}
		mTotalParams.push_back(pc);
	}

	if (mMuscleForceParams.size() > 0)
	{
		ParamCategory pc;
		pc.name = "Muscle Force";
		pc.params.clear();

		for (auto m : mMuscleForceParams)
		{
			ParamElem pe;
			pe.name = m.name;
			pe.cur_v = m.current_ratio;
			pe.max_v = m.max_ratio;
			pe.min_v = m.min_ratio;
			pc.params.push_back(pe);
		}
		mTotalParams.push_back(pc);
	}

	if (mSkelLengthParams.size() > 0)
	{
		ParamCategory pc;
		pc.name = "Skel Length";
		pc.params.clear();

		for (auto m : mSkelLengthParams)
		{
			ParamElem pe;
			pe.name = m.name;
			pe.cur_v = m.current_ratio;
			pe.max_v = m.max_ratio;
			pe.min_v = m.min_ratio;
			pc.params.push_back(pe);
		}
		mTotalParams.push_back(pc);
	}

	if (mUseVelocity)
	{
		ParamCategory pc;
		pc.name = "Velocity";
		pc.params.clear();

		ParamElem pe;
		pe.cur_v = mTargetCOMVelocity;
		pe.min_v = mMinCOMVelocity;
		pe.max_v = mMaxCOMVelocity;
		pe.name = "Velocity";

		pc.params.push_back(pe);
		mTotalParams.push_back(pc);
	}

	if (mUsePhase)
	{
		ParamCategory pc;
		pc.name = "Phase";
		pc.params.clear();

		ParamElem pe;
		pe.cur_v = mPhaseRatio;
		pe.min_v = mMinPhase;
		pe.max_v = mMaxPhase;
		pe.name = "Phase";

		pc.params.push_back(pe);
		mTotalParams.push_back(pc);
	}

	if (mUseStride)
	{
		ParamCategory pc;
		pc.name = "Stride";
		pc.params.clear();

		ParamElem pe;
		pe.cur_v = mStrideRatio;
		pe.min_v = mMinStride;
		pe.max_v = mMaxStride;
		pe.name = "Stride";

		pc.params.push_back(pe);
		mTotalParams.push_back(pc);
	}
}

void Environment::
	UpdateHeadInfo()
{
	mHeadPrevLinearVel = mCharacter->GetSkeleton()->getBodyNode("Head")->getCOMLinearVelocity();
	mHeadPrevAngularVel = mCharacter->GetSkeleton()->getBodyNode("Head")->getAngularVelocity();
}

void Environment::
	SetAction(const Eigen::VectorXd &a)
{

	mAction = a.head(mNumActiveDof) * mActionScale;

	if (0.5 < GetPhase() && mIsTorqueSymMode)
		mAction = GetMirrorAction(mAction);

	mPhaseDisplacement = 0.0;
	mStepDisplacement = 0.0;

	int idx = mNumActiveDof;

	if (mUseTimeWarping)
		mPhaseDisplacement = a[idx++] * mPhaseScale;

	if (mUseVelocity)
	{
		mPhaseRatio = 1.0 + a[idx++] * 0.005;
		if (mPhaseRatio < 0)
			mPhaseRatio = 1E-6;
	}
	if (mUseStepWarping)
		mStepDisplacement = a[idx++] * 0.1;

	if (mPhaseDisplacement < -(1.0 / mControlHz))
		mPhaseDisplacement = -(1.0 / mControlHz);

	mLocalTime += ((mPhaseDisplacement + (1.0 / mControlHz)) * mPhaseRatio * sqrt(1 / mGlobalRatio));

	mGlobalTime += (1.0 / mControlHz) * mPhaseRatio * sqrt(1 / mGlobalRatio);
	double current_gp = GetGlobalPhase();
	double start_time = 0;
	int current_gs = (mGlobalTime - start_time) / mCharacter->GetBVH()->GetMaxTime();

	if (mUseTimeWarping && mStateType != 6)
	{

		double localtime_min = (current_gs + 0) * mCharacter->GetBVH()->GetMaxTime();
		double localtime_max = (current_gs + (1.0)) * mCharacter->GetBVH()->GetMaxTime();

		if ((mLocalTime - start_time) < localtime_min + 1E-6)
			mLocalTime = localtime_min + start_time + 1E-6;
		if ((mLocalTime - start_time) > localtime_max - 1E-6)
			mLocalTime = localtime_max + start_time - 1E-6;
	}

	std::pair<Eigen::VectorXd, Eigen::VectorXd> pv = mCharacter->GetTargetPosAndVel(mLocalTime, mPhaseRatio * sqrt(1 / mGlobalRatio) / mControlHz);
	mTargetPositions = pv.first;
	mTargetVelocities = pv.second;

	if (mIsRender)
	{
		std::pair<Eigen::VectorXd, Eigen::VectorXd> bvh_pv = mCharacter->GetTargetPosAndVel(mGlobalTime, mPhaseRatio * sqrt(1 / mGlobalRatio) / mControlHz, mBVHSkeleton);
		mBVHPositions = bvh_pv.first;
		mBVHVelocities = bvh_pv.second;
	}
	mAverageActivationLevels.setZero();
}

double
Environment::
	GetPhase()
{
	double t_phase = mCharacter->GetBVH()->GetMaxTime();

	return std::fmod(mLocalTime, t_phase) / t_phase;
}

double
Environment::
	GetGlobalPhase()
{
	double t_phase = mCharacter->GetBVH()->GetMaxTime();
	return std::fmod(mGlobalTime, t_phase) / t_phase;
}

void Environment::
	ApplyMuscleParameter()
{

	for (auto &muscleparam_elem : mMuscleLengthParams)
		for (auto m : muscleparam_elem.muscle)
		{
			m->l_min = muscleparam_elem.min_ratio;
			m->l_max = muscleparam_elem.max_ratio;
			m->change_l(muscleparam_elem.current_ratio);
		}

	for (auto &muscleparam_elem : mMuscleForceParams)
		for (auto m : muscleparam_elem.muscle)
		{
			m->f_min = muscleparam_elem.min_ratio;
			m->f_max = muscleparam_elem.max_ratio;
			m->change_f(muscleparam_elem.current_ratio);
		}
}

void Environment::
	ApplySkelParameter()
{
	mGlobalRatio = 1.0;
	for (auto &skel_elem : mSkelLengthParams)
	{
		if (skel_elem.name == "global")
		{
			mGlobalRatio = skel_elem.current_ratio;
			if (mUseAdaptiveKp)
				mCharacter->mGlobalRatio = pow(mGlobalRatio, 3);
		}
	}

	for (auto &s_info : mSkelInfos)
	{
		if (std::get<0>(s_info) != "Head")
		{
			std::get<1>(s_info).value[0] = mGlobalRatio;
			std::get<1>(s_info).value[1] = mGlobalRatio;
			std::get<1>(s_info).value[2] = mGlobalRatio;
		}
	}

	for (auto &skel_elem : mSkelLengthParams)
		for (auto &s_info : mSkelInfos)
			if (skel_elem.Compare(std::get<0>(s_info)))
			{
				if (skel_elem.name == "Head")
				{
					std::get<1>(s_info).value[0] = skel_elem.current_ratio;
					std::get<1>(s_info).value[1] = skel_elem.current_ratio;
					std::get<1>(s_info).value[2] = skel_elem.current_ratio;
				}
				else
				{
					if (skel_elem.name.find("Arm") != std::string::npos)
						std::get<1>(s_info).value[0] = skel_elem.current_ratio * mGlobalRatio;
					else
						std::get<1>(s_info).value[1] = skel_elem.current_ratio * mGlobalRatio;
				}
			}

	mCharacter->ModifySkeletonLength(mSkelInfos);
}

std::map<std::string, double>
Environment::
	GetRewardMap()
{

	return mValues;
}

Eigen::VectorXd
Environment::
	GetMinV()
{
	int idx = 0;
	Eigen::VectorXd min_v(GetParamState().size());

	for (int i = 0; i < mMuscleLengthParams.size(); i++)
		min_v[idx++] = mMuscleLengthParams[i].min_ratio;

	for (int i = 0; i < mMuscleForceParams.size(); i++)
		min_v[idx++] = mMuscleForceParams[i].min_ratio;

	for (int i = 0; i < mSkelLengthParams.size(); i++)
		min_v[idx++] = mSkelLengthParams[i].min_ratio;

	if (mUseVelocity)
		min_v[idx++] = mMinCOMVelocity;

	if (mUsePhase)
		min_v[idx++] = mMinPhase;

	if (mUseStride)
		min_v[idx++] = mMinStride;

	assert(idx == GetParamState().size());
	return min_v;
}

Eigen::VectorXd
Environment::
	GetNormalV()
{
	int idx = 0;
	Eigen::VectorXd normal_v(GetParamState().size());

	for (int i = 0; i < mMuscleLengthParams.size(); i++)
		normal_v[idx++] = 1.0;
	for (int i = 0; i < mMuscleForceParams.size(); i++)
		normal_v[idx++] = 1.0;
	for (int i = 0; i < mSkelLengthParams.size(); i++)
		normal_v[idx++] = 1.0;
	if (mUseVelocity)
		normal_v[idx++] = mPhaseRatio * 1.34 / mCharacter->GetBVH()->GetMaxTime();
	;
	if (mUsePhase)
		normal_v[idx++] = 1.0;
	if (mUseStride)
		normal_v[idx++] = 1.0;

	assert(idx == GetParamState().size());
	return normal_v;
}

Eigen::VectorXd
Environment::
	GetMaxV()
{
	int idx = 0;
	Eigen::VectorXd max_v(GetParamState().size());

	for (int i = 0; i < mMuscleLengthParams.size(); i++)
		max_v[idx++] = mMuscleLengthParams[i].max_ratio;

	for (int i = 0; i < mMuscleForceParams.size(); i++)
		max_v[idx++] = mMuscleForceParams[i].max_ratio;

	for (int i = 0; i < mSkelLengthParams.size(); i++)
		max_v[idx++] = mSkelLengthParams[i].max_ratio;

	if (mUseVelocity)
		max_v[idx++] = mMaxCOMVelocity;

	if (mUsePhase)
		max_v[idx++] = mMaxPhase;

	if (mUseStride)
		max_v[idx++] = mMaxStride;

	assert(idx == GetParamState().size());
	return max_v;
}
std::vector<std::string>
Environment::
	GetMuscleLengthParamName()
{
	std::vector<std::string> param_name;
	for (auto m : mMuscleLengthParams)
		param_name.push_back(m.name);
	return param_name;
}

std::vector<std::string>
Environment::
	GetMuscleForceParamName()
{
	std::vector<std::string> param_name;
	for (auto m : mMuscleForceParams)
		param_name.push_back(m.name);
	return param_name;
}

std::vector<std::string>
Environment::
	GetSkelLengthParamName()
{
	std::vector<std::string> param_name;
	for (auto m : mSkelLengthParams)
		param_name.push_back(m.name);
	return param_name;
}

std::vector<std::string>
Environment::
	GetParamName()
{

	std::vector<std::string> param_name;

	for (auto m : mMuscleLengthParams)
		param_name.push_back(m.name);

	for (auto m : mMuscleForceParams)
		param_name.push_back(m.name);

	for (auto m : mSkelLengthParams)
		param_name.push_back(m.name);

	if (mUseVelocity)
		param_name.push_back("COM Velocity");

	if (mUsePhase)
		param_name.push_back("Phase");

	if (mUseStride)
		param_name.push_back("Stride");

	return param_name;
}

static double sq(double x)
{
	return x * x;
}
static double vectorAngle(const Eigen::Vector3d &a, const Eigen::Vector3d &b)
{
	return atan2(a.cross(b).norm(), a.dot(b));
}
double RadianClamp(double input)
{
	return std::fmod(input + M_PI, 2 * M_PI) - M_PI;
}

Eigen::Vector3d
Environment::
	GetAngularMomentum()
{
	Eigen::Vector3d I = Eigen::Vector3d::Zero();
	auto &skel = mCharacter->GetSkeleton();
	for (auto bn : mCharacter->GetSkeleton()->getBodyNodes())
	{
		I += bn->getAngularMomentum();
		I += bn->getMass() * (bn->getCOM() - skel->getCOM()).cross(bn->getCOMLinearVelocity() - skel->getCOMLinearVelocity());
	}
	return I;
}

double
Environment::
	GetLocoPrinReward()
{
	auto *root = mCharacter->GetSkeleton()->getRootBodyNode();
	auto *head = mCharacter->GetSkeleton()->getBodyNode("Head");

	double mRefVel = GetTargetVelocity();
	double head_rot_diff = mRotDiffWeight * (Eigen::AngleAxisd(head->getTransform().linear()).angle());

	Eigen::Vector3d head_linear_diff = (head->getCOMLinearVelocity() - mHeadPrevLinearVel);

	head_linear_diff[1] *= 0.5;

	double head_linear_acc = mLinearAccWeight * (head_linear_diff * mControlHz).norm();

	double head_angular_acc = mAngularAccWeight * ((head->getAngularVelocity() - mHeadPrevAngularVel) * mControlHz).norm();
	double head_acc_diff = head_linear_acc + head_angular_acc;
	double rot_diff = head_rot_diff;

	double w_alive = 0.1;
	double r_linear_acc = w_alive + (1 - w_alive) * exp_of_squared(head_linear_acc, 4E-3);
	double r_angular_acc = w_alive + (1 - w_alive) * exp_of_squared(head_angular_acc, 4E-3);
	double r_rot = w_alive + (1 - w_alive) * exp_of_squared(rot_diff, 4.0);

	if (mIsRender)
	{
		if (mLinearAccWeight > 1E-6)
			mValues.insert(std::make_pair("head_linear_acc", r_linear_acc));
		if (mAngularAccWeight > 1E-6)
			mValues.insert(std::make_pair("head_angular_acc", r_angular_acc));
		if (mRotDiffWeight > 1E-6)
			mValues.insert(std::make_pair("rot_diff", r_rot));
	}
	return (r_rot * r_linear_acc * r_angular_acc);
}

double
Environment::
	GetImitationReward()
{
	auto *root = mCharacter->GetSkeleton()->getRootBodyNode();

	std::vector<Eigen::Vector3d> cur_ee_pos;
	for (auto &ee : mCharacter->GetEndEffectors())
		cur_ee_pos.push_back(ee->getCOM() - root->getCOM());
	Eigen::VectorXd cur_q = mCharacter->GetSkeleton()->getPositions();
	Eigen::VectorXd cur_dq = mCharacter->GetSkeleton()->getVelocities();

	mCharacter->GetSkeleton()->setPositions(mTargetPositions);
	mCharacter->GetSkeleton()->setVelocities(mTargetVelocities);
	mCharacter->GetSkeleton()->computeForwardKinematics(true, true, false);
	std::vector<Eigen::Vector3d> ref_ee_pos;
	for (auto &ee : mCharacter->GetEndEffectors())
		ref_ee_pos.push_back(ee->getCOM() - root->getCOM());
	Eigen::VectorXd ref_q = mTargetPositions;
	Eigen::VectorXd ref_dq = mTargetVelocities;

	mCharacter->GetSkeleton()->setPositions(cur_q);
	mCharacter->GetSkeleton()->setVelocities(cur_dq);
	mCharacter->GetSkeleton()->computeForwardKinematics(true, true, false);

	Eigen::VectorXd q_diff = 2 * mCharacter->GetSkeleton()->getPositionDifferences(cur_q, ref_q);
	q_diff.head(6).setZero();
	q_diff.segment(13, 2).setZero();
	q_diff.segment(22, 2).setZero();

	if (IsArmExist())
	{
	}

	Eigen::VectorXd ee_diff(mCharacter->GetEndEffectors().size() * 3);
	for (int i = 0; i < mCharacter->GetEndEffectors().size(); i++)
		ee_diff.segment(i * 3, 3) = 4 * (cur_ee_pos[i] - ref_ee_pos[i]);

	double w_alive = 0.0;
	double r_q = w_alive + (1 - w_alive) * exp_of_squared(q_diff, 8.0);

	if (mIsRender)
	{
		mValues.insert(std::make_pair("q", r_q));
	}

	return r_q;
}

double Environment::
	GetAvgVelocityReward()
{

	int horizon = mCharacter->GetBVH()->GetMaxTime() * mSimulationHz / (mPhaseRatio * sqrt(1 / mGlobalRatio));

	double mRefVel = GetTargetVelocity();

	Eigen::Vector3d avg_vel;
	if (mComTrajectory.size() > 1.5 * horizon)
		avg_vel = ((mComTrajectory.back() - mComTrajectory[mComTrajectory.size() - horizon]) / horizon) * mSimulationHz;

	else
		avg_vel = Eigen::Vector3d(0, 0, mRefVel);

	Eigen::Vector3d vel_diff = (avg_vel - Eigen::Vector3d(0, 0, mRefVel));

	double inst_vel_diff = mCharacter->GetSkeleton()->getCOMLinearVelocity()[2] - mRefVel;

	double r_instvel = 1.0;
	double r_avgvel = exp_of_squared(vel_diff, 16.0);

	if (mIsRender)
	{
		mValues.insert(std::make_pair("avg_velocity", r_avgvel));
	}
	return r_avgvel * r_instvel;
}

double
Environment::
	GetOverTorqueReward()
{
	mOverTorque /= (mSimulationHz / mControlHz);
	double r_overtorque = 0.0;
	r_overtorque = exp_of_squared(mOverTorque, 0.01);

	mOverTorque = 0;

	return r_overtorque;
}

double
Environment::
	GetCOMReward()
{
	Eigen::VectorXd cur_pos = mCharacter->GetSkeleton()->getPositions();
	Eigen::Vector3d cur_com = mCharacter->GetSkeleton()->getCOM();
	mCharacter->GetSkeleton()->setPositions(mTargetPositions);
	Eigen::Vector3d tar_com = mCharacter->GetSkeleton()->getCOM();
	mCharacter->GetSkeleton()->setPositions(cur_pos);

	Eigen::Vector3d com_diff = tar_com - cur_com;
	com_diff[1] = 0;

	double r_com = exp_of_squared(com_diff.norm(), 10.0);

	if (mIsRender)
		mValues.insert(std::make_pair("com", r_com));

	return r_com;
}

double
Environment::
	GetPassiveForceReward()
{
	mPassiveForce /= (mSimulationHz / mControlHz);
	mPassiveForce *= 1E-6;

	double r_passive = 0.0;
	r_passive = exp_of_squared(mPassiveForce, 5E-1);
	if (mIsRender)
		mValues.insert(std::make_pair("passive_force", r_passive));
	return r_passive;
}

double
Environment::
	GetMetabolicReward()
{
	double r_metabolic = 1.0;
	double w_alive = 0.0;

	if (mMetabolicType < 3 || (mMetabolicType == 5) || (mMetabolicType == 6))
	{

		mUsedTorque /= (mSimulationHz / mControlHz);
		mUsedTorque *= 5E-4;

		mNetUsedTorque /= (mSimulationHz / mControlHz);
		mNetUsedTorque *= 5E-4;

		mSimUsedTorque /= (mSimulationHz / mControlHz);
		mSimUsedTorque *= 5E-4;

		mNetSimUsedTorque /= (mSimulationHz / mControlHz);
		mNetSimUsedTorque *= 5E-4;

		double mass_ratio = mCharacter->GetSkeleton()->getMass() / 65.0;
		mass_ratio *= mass_ratio;

		mActiveForce /= (mSimulationHz / mControlHz);
		mActiveForce *= 1E-6;

		mTotalAcc /= (mSimulationHz / mControlHz);
		mTotalAcc *= 1E-5;

		if (mMetabolicType == 0)
			r_metabolic = w_alive + (1 - w_alive) * exp(-mUsedTorque / mass_ratio);
		else if (mMetabolicType == 1)
		{
			if (mUseMuscle)
				r_metabolic = w_alive + (1 - w_alive) * exp_of_squared(mActiveForce / mass_ratio, 5E-1);
			else
			{
				std::cout << "[Error ] Invlid Metabolic Type " << std::endl;
				exit(-1);
			}
		}
		else if (mMetabolicType == 2)
		{
			r_metabolic = w_alive + (1 - w_alive) * exp(-mNetUsedTorque / mass_ratio);
		}
		else if (mMetabolicType == 5)
		{
			r_metabolic = w_alive + (1 - w_alive) * exp_of_squared(mSimUsedTorque / mass_ratio, 1);
		}
		else if (mMetabolicType == 6)
		{
			r_metabolic = w_alive + (1 - w_alive) * exp_of_squared(mNetSimUsedTorque / mass_ratio, 1);
		}

		mUsedTorque = 0;
		mNetUsedTorque = 0;

		mSimUsedTorque = 0;
		mNetSimUsedTorque = 0;

		mActiveForce = 0;
		mTotalAcc = 0;
	}
	else if (mMetabolicType == 3)
	{

		mMetabolicEnergy = mMetabolicEnergy / mMass;
		r_metabolic = -mMetabolicEnergy;
	}
	else if (mMetabolicType == 4)
	{
		double act = 0;
		for (auto a : mActivationBuf)
			act += (a.squaredNorm() / a.rows());

		act /= pow(mGlobalRatio, 0.5);
		r_metabolic = exp(-act);
		mActivationBuf.clear();
	}

	if (mIsRender)
		mValues.insert(std::make_pair("metabolic", r_metabolic));
	return r_metabolic;
}

double
Environment::
	GetStepReward()
{
	const auto result = mWorld->getConstraintSolver()->getLastCollisionResult();
	double phase = GetPhase();
	Eigen::Vector3d foot_diff;
	foot_diff.setZero();
	Eigen::Vector2i isContact = GetIsContact();
	if (mCurrentStance == 0)
	{
		if (mCharacter->GetSkeleton()->getBodyNode("HeelL") == NULL)
			mCurrentFoot = mCharacter->GetSkeleton()->getBodyNode("TalusL")->getCOM();
		else
			mCurrentFoot = mCharacter->GetSkeleton()->getBodyNode("HeelL")->getCOM();

		mCurrentFoot[1] -= 0.0249;
		mCurrentFoot[2] -= mStepDisplacement;

		foot_diff = mCurrentFoot - mCurrentTargetFoot;

		if (isContact[0] == 1)
			foot_diff[1] = 0;

		if (phase >= 0.31 && phase < 0.81)
		{
			{
				mCurrentStance = 1;
				mCurrentTargetFoot = mNextTargetFoot;
				mNextTargetFoot = mCurrentTargetFoot;

				if (mUseAbsStep)
					mNextTargetFoot[2] += 0.5 * mStrideRatio * mRefStride * mGlobalRatio;
				else
					mNextTargetFoot[2] = mCurrentFoot[2] + mStrideRatio * mRefStride * mGlobalRatio;

				mNextTargetFoot[0] *= -1;
			}
		}
	}
	else
	{
		if (mCharacter->GetSkeleton()->getBodyNode("HeelR") == NULL)
			mCurrentFoot = mCharacter->GetSkeleton()->getBodyNode("TalusR")->getCOM();
		else
			mCurrentFoot = mCharacter->GetSkeleton()->getBodyNode("HeelR")->getCOM();

		mCurrentFoot[1] -= 0.0249;
		mCurrentFoot[2] -= mStepDisplacement;

		foot_diff = mCurrentFoot - mCurrentTargetFoot;

		if (isContact[1] == 1)
			foot_diff[1] = 0;

		if (phase >= 0.81 || phase < 0.31)
		{
			{
				mCurrentStance = 0;
				mCurrentTargetFoot = mNextTargetFoot;
				mNextTargetFoot = mCurrentTargetFoot;

				if (mUseAbsStep)
					mNextTargetFoot[2] += 0.5 * mStrideRatio * mRefStride * mGlobalRatio;
				else
					mNextTargetFoot[2] = mCurrentFoot[2] + mStrideRatio * mRefStride * mGlobalRatio;

				mNextTargetFoot[0] *= -1;
			}
		}
	}
	foot_diff[0] = 0.0;

	if (abs(foot_diff[0]) < 0.075 && abs(foot_diff[1]) < 0.075 && abs(foot_diff[2]) < 0.075)
		foot_diff.setZero();
	else
	{
		Eigen::Vector3d clipped_foot_diff = foot_diff;
		clipped_foot_diff[0] = dart::math::clip(clipped_foot_diff[0], -0.075, 0.075);
		clipped_foot_diff[1] = dart::math::clip(clipped_foot_diff[1], -0.075, 0.075);
		clipped_foot_diff[2] = dart::math::clip(clipped_foot_diff[2], -0.075, 0.075);

		foot_diff -= clipped_foot_diff;
	}
	foot_diff *= 8;

	double r = exp_of_squared(foot_diff, 2.0);
	if (mIsRender)
		mValues.insert(std::make_pair("foot_diff", r));

	return r;
}

double
Environment::
	GetReward()
{
	if (mIsRender)
		mValues.clear();

	max_r = 1.0;

	double r_total =
		0.0 + mMetabolicWeight * GetMetabolicReward() + (mUseImitation ? 0.2 * GetImitationReward() : 0) + 2.0 * (mUseLocoPrinReward ? GetLocoPrinReward() : 1) * GetAvgVelocityReward() * (mUseVelocity ? 1 : GetStepReward());

	GetOverTorqueReward();

	if (dart::math::isNan(r_total))
		return 0;

	if (mIsRender)
		mValues.insert(std::make_pair("total", r_total));

	return r_total;
}

Eigen::VectorXd
Environment::
	getPositionDifferences(Eigen::VectorXd v1, Eigen::VectorXd v2)
{
	auto &skel = mCharacter->GetSkeleton();
	Eigen::VectorXd cur_pos = skel->getPositions();
	auto joints = skel->getJoints();
	Eigen::VectorXd result = Eigen::VectorXd::Zero(joints.size());

	for (int i = 0; i < joints.size(); i++)
	{
		auto jn = joints[i];
		if (jn->getNumDofs() == 0)
			continue;

		int idx = jn->getIndexInSkeleton(0);

		double diff = 0;
		if (jn->getNumDofs() == 6)
		{
			Eigen::AngleAxisd a1 = Eigen::AngleAxisd(BallJoint::convertToRotation(v1.segment<3>(idx)));
			Eigen::AngleAxisd a2 = Eigen::AngleAxisd(BallJoint::convertToRotation(v2.segment<3>(idx)));
			double axis_diff = atan2(a1.axis().cross(a2.axis()).norm(), a1.axis().dot(a2.axis()));
			double angle_diff = (a1.angle() - a2.angle());
			double dist_diff = (v1.segment<3>(idx + 3) - v2.segment<3>(idx + 3)).norm();

			diff = abs(axis_diff) + abs(angle_diff) + abs(dist_diff);
		}
		else if (jn->getNumDofs() == 3)
		{

			Eigen::AngleAxisd a1 = Eigen::AngleAxisd(BallJoint::convertToRotation(v1.segment<3>(idx)));
			Eigen::AngleAxisd a2 = Eigen::AngleAxisd(BallJoint::convertToRotation(v2.segment<3>(idx)));
			double axis_diff = atan2(a1.axis().cross(a2.axis()).norm(), a1.axis().dot(a2.axis()));
			double angle_diff = (a1.angle() - a2.angle());

			diff = abs(axis_diff) + abs(angle_diff);
		}
		else if (jn->getNumDofs() == 1)
		{

			double angle_diff = (v1[idx] - v2[idx]);

			diff = abs(angle_diff);
		}
		result[i] = 0.8 * diff;
	}
	return result;
}

int Environment::
	GetCascadingType(std::string metadata)
{
	std::stringstream ifs;
	ifs.str(metadata);
	std::string str;
	std::string index;
	std::stringstream ss;
	int result = 0;
	while (!ifs.eof())
	{
		str.clear();
		index.clear();
		ss.clear();

		std::getline(ifs, str);
		ss.str(str);
		ss >> index;
		if (!index.compare("cascading_type"))
		{
			ss >> result;
		}
	}
	return result;
}

std::pair<Eigen::VectorXd, Eigen::VectorXd>
Environment::
	GetSpace(std::string metadata)
{
	if (metadata == "None")
		return std::make_pair(GetMinV(), GetMaxV());

	std::stringstream ifs;
	ifs.str(metadata);
	std::string str;
	std::string index;
	std::stringstream ss;

	Eigen::VectorXd minv = Eigen::VectorXd::Ones(mNumParamState);
	Eigen::VectorXd maxv = Eigen::VectorXd::Ones(mNumParamState);

	while (!ifs.eof())
	{
		str.clear();
		index.clear();
		ss.clear();

		std::getline(ifs, str);
		ss.str(str);
		ss >> index;
		if (!index.compare("muscle_param") || !index.compare("muscle_length_param"))
		{
			while (!ifs.eof())
			{
				std::string muscle_name;
				double min_ratio;
				double max_ratio;

				str.clear();
				ss.clear();

				std::getline(ifs, str);
				ss.str(str);

				ss >> muscle_name;
				if (!muscle_name.compare("group"))
					ss >> muscle_name;
				if (!muscle_name.compare("muscle_end") || !muscle_name.compare("muscle_length_end"))
					break;

				ss >> min_ratio;
				ss >> max_ratio;

				int idx = 0;
				for (auto m : mMuscleLengthParams)
				{
					if (m.name == muscle_name)
					{
						minv[idx] = min_ratio;
						maxv[idx] = max_ratio;
						break;
					}
					idx++;
				}
			}
		}
		else if (!index.compare("muscle_force_param"))
		{
			while (!ifs.eof())
			{
				std::string muscle_name;
				double min_ratio;
				double max_ratio;

				str.clear();
				ss.clear();

				std::getline(ifs, str);
				ss.str(str);

				ss >> muscle_name;
				if (!muscle_name.compare("group"))
					ss >> muscle_name;
				if (!muscle_name.compare("muscle_force_end"))
					break;

				ss >> min_ratio;
				ss >> max_ratio;

				int idx = mMuscleLengthParamNum;
				for (auto m : mMuscleForceParams)
				{
					if (m.name == muscle_name)
					{
						minv[idx] = 0.05;
						maxv[idx] = max_ratio;
						break;
					}
					idx++;
				}
			}
		}
		else if (!index.compare("skel_length_param"))
		{
			while (!ifs.eof())
			{
				SkelParam skelparam_elem;

				std::string skel_name;
				double min_ratio;
				double max_ratio;

				str.clear();
				ss.clear();

				std::getline(ifs, str);
				ss.str(str);

				ss >> skel_name;

				if (!skel_name.compare("skel_length_end"))
					break;

				if (!skel_name.compare("#"))
					continue;

				skelparam_elem.name = skel_name;

				ss >> min_ratio;
				ss >> max_ratio;

				int idx = mMuscleLengthParamNum + mMuscleForceParamNum;
				for (auto m : mSkelLengthParams)
				{
					if (m.name == skel_name)
					{
						minv[idx] = min_ratio;
						maxv[idx] = max_ratio;
						break;
					}
					idx++;
				}
			}
		}

		else if (!index.compare("use_velocity"))
		{
			int idx = mMuscleLengthParamNum + mMuscleForceParamNum + mSkelLengthParamNum;

			minv[idx] = mMinCOMVelocity;
			maxv[idx] = mMaxCOMVelocity;
		}
		else if (!index.compare("use_phase") && !mUseVelocity)
		{
			int idx = mMuscleLengthParamNum + mMuscleForceParamNum + mSkelLengthParamNum + (mUseVelocity ? 1 : 0);

			minv[idx] = mMinPhase;
			maxv[idx] = mMaxPhase;
		}
		else if (!index.compare("use_stride") && !mUseVelocity)
		{
			int idx = mMuscleLengthParamNum + mMuscleForceParamNum + mSkelLengthParamNum + (mUseVelocity ? 1 : 0) + (mUsePhase ? 1 : 0);

			minv[idx] = mMinStride;
			maxv[idx] = mMaxStride;
		}
	}
	return std::make_pair(minv, maxv);
}

void Environment::
	NaivePoseOptimization(int max_iter)
{
	int iter = 0;
	bool isDone = false;

	if (mUseMuscle)
	{
		double min_passive_norm = 1E5;
		Eigen::VectorXd min_pos;
		while (iter < max_iter)
		{
			isDone = true;
			Eigen::VectorXd pos = mCharacter->GetSkeleton()->getPositions();
			Eigen::VectorXd _m_state = GetMuscleState(true) * 0.008;
			Eigen::VectorXd m_state = _m_state.head(_m_state.rows() / 3);

			if (min_passive_norm > m_state.norm())
			{
				min_passive_norm = m_state.norm();
				min_pos = pos;
				if (min_passive_norm < 0.5)
					break;
			}

			for (int i = 0; i < m_state.size(); i++)
			{
				if (abs(m_state[i]) < 0.5)
					m_state[i] = 0.0;
				else
				{
					m_state[i] *= 0.005;
					if (m_state[i] > 0.1)
						m_state *= (0.1 / m_state[i]);
				}
				m_state[i] *= mOptWeight[i + mRootJointDof];
			}

			if ((pos.rows() - 6) != m_state.rows())
			{
				std::cout << "[DEBUG] " << pos.rows() - 6 << "  ==  " << m_state.rows() << std::endl;
				exit(-1);
			}
			pos.tail(pos.rows() - 6) += m_state;
			mCharacter->GetSkeleton()->setPositions(pos);
			updateCharacterInLimit();
			iter++;
		}

		mCharacter->GetSkeleton()->setPositions(min_pos);

		Eigen::Vector3d mid_foots;
		double phase = GetPhase();
		mid_foots = 0.5 * (mCharacter->GetSkeleton()->getBodyNode("TalusL")->getCOM() + mCharacter->GetSkeleton()->getBodyNode("TalusR")->getCOM());
		// if(phase >= 0.81 || phase < 0.31)
		// 	mid_foots = (mCharacter->GetSkeleton()->getBodyNode("TalusL")->getCOM() * mCharacter->GetSkeleton()->getBodyNode("TalusL")->getMass()
		// 	+ mCharacter->GetSkeleton()->getBodyNode("FootPinkyL")->getCOM() * mCharacter->GetSkeleton()->getBodyNode("FootPinkyL")->getMass()
		// 	+ mCharacter->GetSkeleton()->getBodyNode("FootThumbL")->getCOM() * mCharacter->GetSkeleton()->getBodyNode("FootThumbL")->getMass()
		// 	)
		// 	/(mCharacter->GetSkeleton()->getBodyNode("TalusL")->getMass()  + mCharacter->GetSkeleton()->getBodyNode("FootPinkyL")->getMass()  +  mCharacter->GetSkeleton()->getBodyNode("FootThumbL")->getMass() );
		// else
		// 	mid_foots = (mCharacter->GetSkeleton()->getBodyNode("TalusR")->getCOM() * mCharacter->GetSkeleton()->getBodyNode("TalusR")->getMass()
		// 	+ mCharacter->GetSkeleton()->getBodyNode("FootPinkyR")->getCOM() * mCharacter->GetSkeleton()->getBodyNode("FootPinkyR")->getMass()
		// 	+ mCharacter->GetSkeleton()->getBodyNode("FootThumbR")->getCOM() * mCharacter->GetSkeleton()->getBodyNode("FootThumbR")->getMass()
		// 	)
		// 	/(mCharacter->GetSkeleton()->getBodyNode("TalusR")->getMass()  + mCharacter->GetSkeleton()->getBodyNode("FootPinkyR")->getMass()  +  mCharacter->GetSkeleton()->getBodyNode("FootThumbR")->getMass() );

		Eigen::Vector3d com = mCharacter->GetSkeleton()->getCOM();

		Eigen::Vector3d vec1 = Eigen::Vector3d(0, -1, 0);
		Eigen::Vector3d vec2 = mid_foots - com;
		vec2[0] = 0;
		vec2 = vec2.normalized();
		double theta = atan2((vec1.cross(vec2)).norm(), vec1.dot(vec2));

		Eigen::VectorXd pos = mCharacter->GetSkeleton()->getPositions();
		Eigen::Matrix3d rot = BallJoint::convertToRotation(pos.head(3)) * Eigen::AngleAxisd(-theta, vec1.cross(vec2).normalized());
		Eigen::AngleAxisd a_rot = Eigen::AngleAxisd(rot);
		pos.head(3) = a_rot.axis() * a_rot.angle();
		mCharacter->GetSkeleton()->setPositions(pos);
	}

	auto collisionEngine = mWorld->getConstraintSolver()->getCollisionDetector();
	auto collisionGroup = mWorld->getConstraintSolver()->getCollisionGroup();

	dart::collision::CollisionOption option;
	dart::collision::CollisionResult results;

	bool collision = collisionGroup->collide(option, &results);
	iter = 0;
	while (collision)
	{

		bool isGround = false;
		for (auto bn : results.getCollidingBodyNodes())
			if (bn->getName() == "ground")
			{
				isGround = true;
				break;
			}
		if (!isGround)
			break;
		Eigen::VectorXd pos = mCharacter->GetSkeleton()->getPositions();

		pos[4] += 0.001;
		mCharacter->GetSkeleton()->setPositions(pos);
		iter++;
		collision = collisionGroup->collide(option, &results);
	}
}

void Environment::
	NaiveMotionOptimization()
{
	mCharacter->GetBVH()->ResetModifiedMotions();
	auto &motions = mCharacter->GetBVH()->GetModifiedMotions();
	for (auto &m : motions)
	{
		mCharacter->GetSkeleton()->setPositions(m);
		mCharacter->GetSkeleton()->computeForwardKinematics(true, false, false);
		NaivePoseOptimization();
		m = mCharacter->GetSkeleton()->getPositions();
	}
}

Eigen::VectorXd
Environment::
	GetMirrorState(Eigen::VectorXd state)
{
	Eigen::VectorXd mirror_state = state;
	int idx = 0;
	auto &skel = mCharacter->GetSkeleton();
	int gap = 3 * (skel->getBodyNode("FemurL")->getIndexInTree() - skel->getBodyNode("FemurR")->getIndexInTree());
	gap = abs(gap);
	int arm_gap = 0;

	if (IsArmExist() || skel->getBodyNode("ShoulderL") != NULL)
	{
		arm_gap = 3 * (skel->getBodyNode("ShoulderL")->getIndexInTree() - skel->getBodyNode("ShoulderR")->getIndexInTree());
		arm_gap = abs(arm_gap);
	}

	if (state.rows() != mNumState)
		exit(-1);

	if (mStateType == 8 || mStateType == 9 || mStateType == 10 || mStateType == 11)
	{
		int num_body_nodes = skel->getNumBodyNodes();

		idx = 1;

		Eigen::VectorXd p_bkup = skel->getPositions();
		Eigen::VectorXd v_bkup = skel->getVelocities();
		skel->setPositions(mCharacter->GetMirrorPosition(p_bkup));
		skel->setVelocities(mCharacter->GetMirrorPosition(v_bkup));

		Eigen::VectorXd p_mirror, v_mirror;
		p_mirror.resize(num_body_nodes * 3 + num_body_nodes * 6);
		v_mirror.resize((num_body_nodes + 1) * 3 + num_body_nodes * 3);

		for (int i = 0; i < num_body_nodes; i++)
		{
			p_mirror.segment<3>(i * 3) = skel->getBodyNode(i)->getCOM() - skel->getCOM() - Eigen::Vector3d(0.0, -0.98, 0.0);
			Eigen::Isometry3d transform = skel->getBodyNode(i)->getTransform();
			p_mirror.segment<6>(num_body_nodes * 3 + 6 * i) << transform.linear()(0, 0), transform.linear()(0, 1), transform.linear()(0, 2),
				transform.linear()(1, 0), transform.linear()(1, 1), transform.linear()(1, 2);

			v_mirror.segment<3>(i * 3) = skel->getBodyNode(i)->getCOMLinearVelocity() - skel->getCOMLinearVelocity();

			v_mirror.segment<3>((num_body_nodes + 1) * 3 + i * 3) = 0.2 * skel->getBodyNode(i)->getAngularVelocity();
		}
		v_mirror.segment<3>(num_body_nodes * 3) = skel->getCOMLinearVelocity();

		mirror_state.segment(idx, p_mirror.rows()) = p_mirror;
		idx += p_mirror.rows();
		mirror_state.segment(idx, v_mirror.rows()) = v_mirror;
		idx += v_mirror.rows();

		skel->setPositions(p_bkup);
		skel->setVelocities(v_bkup);
		skel->computeForwardKinematics(true, true, false);
	}

	if (mPhaseType == 0)
	{
		mirror_state[idx++] *= -1;
		mirror_state[idx++] *= -1;
	}
	else if (mPhaseType == 1 || mStateType == 2)
	{
		if (mirror_state[idx] > 0)
			mirror_state[idx] = fmod(mirror_state[idx] + 0.5, 1.0);
		idx++;
	}

	if (mUseMuscle && mStateType != 5)
	{
		idx += mNumMuscleState;
	}

	if (mStateType == 10 || mStateType == 11)
	{
		if (mSkelLengthParams.size() >= 2)
			idx += 2;

		for (int i = 2; i < mSkelLengthParams.size(); i += 2)
		{
			mirror_state[idx] = state[idx + 1];
			mirror_state[idx + 1] = state[idx];
			idx += 2;
		}
	}
	idx += (mStateType == 6 ? 1 : (mUseVelocity ? 1 : 0)) + (mUsePhase ? 1 : 0) + (mUseStride ? 1 : 0);

	if (mUseTimeWarping)
	{
		mirror_state[idx] = std::fmod(state[idx] + 0.5, 1.0);
		idx++;
	}
	if (mUseStride)
	{
		mirror_state[idx++] *= -1;
		idx++;
		idx++;
	}

	if (mStateType == 11)
	{
		for (int i = 0; i < mMuscleLengthParams.size(); i += 2)
		{
			mirror_state[idx] = state[idx + 1];
			mirror_state[idx + 1] = state[idx];
			idx += 2;
		}
		for (int i = 0; i < mMuscleForceParams.size(); i += 2)
		{
			mirror_state[idx] = state[idx + 1];
			mirror_state[idx + 1] = state[idx];
			idx += 2;
		}
	}

	if (idx != mNumState)
	{
		std::cout << "[Warning] GetMirrorState " << idx << "  " << mNumState << std::endl;
		exit(-1);
	}

	return mirror_state;
}

Eigen::VectorXd
Environment::
	GetMirrorAction(Eigen::VectorXd action)
{
	Eigen::VectorXd full_action = Eigen::VectorXd(mCharacter->GetSkeleton()->getPositions().rows());
	full_action.setZero();
	full_action.tail(action.rows()) = action;
	return mCharacter->GetMirrorPosition(full_action).tail(action.rows());
}

Eigen::VectorXd
Environment::
	GetMirrorActivation(Eigen::VectorXd activation)
{
	Eigen::VectorXd mirror_activation = activation;
	double tmp = 0.0;
	for (int i = 0; i < mirror_activation.rows(); i += 2)
	{
		tmp = mirror_activation[i];
		mirror_activation[i] = mirror_activation[i + 1];
		mirror_activation[i + 1] = tmp;
	}
	return mirror_activation;
}

void Environment::
	updateCharacterInLimit()
{
	Eigen::VectorXd cur_q = mCharacter->GetSkeleton()->getPositions();
	for (auto jn : mCharacter->GetSkeleton()->getJoints())
	{
		int jn_dof = jn->getNumDofs();
		if (jn_dof == 6 || jn_dof == 0)
			continue;
		int jn_idx = jn->getIndexInSkeleton(0);
		if (jn_dof == 3)
			for (int i = 0; i < 3; i++)
				cur_q[jn_idx + i] = clamp(cur_q[jn_idx + i], jn->getPositionLowerLimit(i), jn->getPositionUpperLimit(i));
		if (jn_dof == 1)
			cur_q[jn_idx] = clamp(cur_q[jn_idx], jn->getPositionLowerLimit(0), jn->getPositionUpperLimit(0));
	}
	mCharacter->GetSkeleton()->setPositions(cur_q);
	mCharacter->GetSkeleton()->computeForwardKinematics(true, false, false);
}

Eigen::VectorXd
Environment::
	GetDisplacement()
{
	Eigen::VectorXd res = mCharacter->GetSkeleton()->getPositions() - mTargetPositions;

	return res.tail(res.rows() - mRootJointDof);
}

Eigen::VectorXd
Environment::
	GetParamSamplingPolicy()
{
	Eigen::VectorXd res = Eigen::VectorXd::Zero(mNumParamState);

	for (int i = 0; i < mMuscleForceParamNum + mMuscleLengthParamNum; i++)
		res[i] = 1;

	return res;
}

bool Environment::
	IsArmExist()
{
	bool isExist = false;
	for (auto bn : mCharacter->GetSkeleton()->getBodyNodes())
		if (bn->getName().find("Hand") != std::string::npos)
			isExist = true;
	return isExist;
}

double
Environment::
	GetTargetVelocity()
{
	return (mUseVelocity ? mTargetCOMVelocity : mPhaseRatio * sqrt(1 / mGlobalRatio) * (mStrideRatio * mRefStride * mGlobalRatio / mCharacter->GetBVH()->GetMaxTime()));
}

void Environment::
	SetMuscleAction(const Eigen::VectorXd &a)
{
	if (GetPhase() > 0.5 && mIsMuscleSymMode)
	{
		for (int i = 0; i < a.rows(); i += 2)
		{
			if (mUseExcitation && !mIsNew)
			{
				mExcitationLevels[i] = a[i + 1];
				mExcitationLevels[i + 1] = a[i];
			}
			else
			{
				mActivationLevels[i] = a[i + 1];
				mActivationLevels[i + 1] = a[i];
			}
		}
	}
	else
	{
		if (mUseExcitation && !mIsNew)
			mExcitationLevels = a;
		else
			mActivationLevels = a;
	}
}

Eigen::Vector2i
Environment::
	GetIsContact()
{
	Eigen::Vector2i result = Eigen::Vector2i(0, 0);
	const auto results = mWorld->getConstraintSolver()->getLastCollisionResult();
	for (auto bn : results.getCollidingBodyNodes())
	{
		if (bn->getName() == "HeelL" || bn->getName() == "TalusL" || ((bn->getName() == "FootPinkyL" || bn->getName() == "FootThumbL") && !mIsRender))
			result[0] = 1;

		if (bn->getName() == "HeelR" || bn->getName() == "TalusR" || ((bn->getName() == "FootPinkyR" || bn->getName() == "FootThumbR") && !mIsRender))
			result[1] = 1;
	}

	return result;
}
