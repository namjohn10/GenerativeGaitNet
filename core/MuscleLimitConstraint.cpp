#include "MuscleLimitConstraint.h"
#include "dart/external/odelcpsolver/lcp.h"
#include "Muscle.h"

#define DART_ERROR_ALLOWANCE 0.0
#define DART_ERP 0.01
#define DART_MAX_ERV 1e+1
#define DART_CFM 1e-9

double MuscleLimitConstraint::mErrorAllowance = DART_ERROR_ALLOWANCE;
double MuscleLimitConstraint::mErrorReductionParameter = DART_ERP;
double MuscleLimitConstraint::mMaxErrorReductionVelocity = DART_MAX_ERV;
double MuscleLimitConstraint::mConstraintForceMixing = DART_CFM;
using namespace dart::constraint;
using namespace MASS;

MuscleLimitConstraint::
	MuscleLimitConstraint(Muscle *muscle)
	: ConstraintBase(),
	  mMuscle(muscle),
	  mJoints(muscle->GetRelatedJoints()),
	  mSkeleton(muscle->mAnchors[0]->bodynodes[0]->getSkeleton()),
	  mBodyNodes(muscle->GetRelatedBodyNodes()),
	  mNumRelatedDofs(0)
{
	for (int i = 0; i < mJoints.size(); i++)
		mNumRelatedDofs += mJoints[i]->getNumDofs();
	mActive = false;
	mViolation = 0;
	mNegativeVel = 0;
	mUpperBound = -dInfinity;
	mLowerBound = dInfinity;
	mdCdTheta.resize(mSkeleton->getNumDofs());

	setConstraintForceMixing(1E-2);
	mOldX = 0;
	mLifeTime = 0;
}

MuscleLimitConstraint::~MuscleLimitConstraint()
{
}
//==============================================================================
void MuscleLimitConstraint::setErrorAllowance(double _allowance)
{
	// Clamp error reduction parameter if it is out of the range
	if (_allowance < 0.0)
	{
		dtwarn << "Error reduction parameter[" << _allowance
			   << "] is lower than 0.0. "
			   << "It is set to 0.0." << std::endl;
		mErrorAllowance = 0.0;
	}

	mErrorAllowance = _allowance;
}

//==============================================================================
double MuscleLimitConstraint::getErrorAllowance()
{
	return mErrorAllowance;
}

//==============================================================================
void MuscleLimitConstraint::setErrorReductionParameter(double _erp)
{
	// Clamp error reduction parameter if it is out of the range [0, 1]
	if (_erp < 0.0)
	{
		dtwarn << "Error reduction parameter[" << _erp << "] is lower than 0.0. "
			   << "It is set to 0.0." << std::endl;
		mErrorReductionParameter = 0.0;
	}
	if (_erp > 1.0)
	{
		dtwarn << "Error reduction parameter[" << _erp << "] is greater than 1.0. "
			   << "It is set to 1.0." << std::endl;
		mErrorReductionParameter = 1.0;
	}

	mErrorReductionParameter = _erp;
}

//==============================================================================
double MuscleLimitConstraint::getErrorReductionParameter()
{
	return mErrorReductionParameter;
}

//==============================================================================
void MuscleLimitConstraint::setMaxErrorReductionVelocity(double _erv)
{
	// Clamp maximum error reduction velocity if it is out of the range
	if (_erv < 0.0)
	{
		dtwarn << "Maximum error reduction velocity[" << _erv
			   << "] is lower than 0.0. "
			   << "It is set to 0.0." << std::endl;
		mMaxErrorReductionVelocity = 0.0;
	}

	mMaxErrorReductionVelocity = _erv;
}

//==============================================================================
double MuscleLimitConstraint::getMaxErrorReductionVelocity()
{
	return mMaxErrorReductionVelocity;
}

//==============================================================================
void MuscleLimitConstraint::setConstraintForceMixing(double _cfm)
{
	// Clamp constraint force mixing parameter if it is out of the range
	if (_cfm < 1e-9)
	{
		dtwarn << "Constraint force mixing parameter[" << _cfm
			   << "] is lower than 1e-9. "
			   << "It is set to 1e-9." << std::endl;
		mConstraintForceMixing = 1e-9;
	}
	if (_cfm > 1.0)
	{
		dtwarn << "Constraint force mixing parameter[" << _cfm
			   << "] is greater than 1.0. "
			   << "It is set to 1.0." << std::endl;
		mConstraintForceMixing = 1.0;
	}

	mConstraintForceMixing = _cfm;
}

//==============================================================================
double MuscleLimitConstraint::getConstraintForceMixing()
{
	return mConstraintForceMixing;
}
double MuscleLimitConstraint::EvalObjective()
{
	mMuscle->Update();
	update();
	return -mViolation;
}
Eigen::VectorXd MuscleLimitConstraint::EvalGradient()
{
	mMuscle->Update();
	update();
	return -mdCdTheta;
}
void MuscleLimitConstraint::update()
{
	double l_mt = mMuscle->Getl_mt();
	mViolation = (l_mt - mMuscle->l_mt_max);
	if (mViolation >= 0.0) // Violate!
	{
		mDim = 1;

		mdCdTheta = mMuscle->Getdl_dtheta();
		mNegativeVel = -mdCdTheta.dot(mSkeleton->getVelocities());

		mLowerBound = -dInfinity;
		mUpperBound = 0.0;
		if (mActive)
			mLifeTime++;
		else
		{
			mActive = true;
			mLifeTime = 0;
		}
	}
	else
	{
		mDim = 0;
		mActive = false;
	}
}

void MuscleLimitConstraint::getInformation(ConstraintInfo *_lcp)
{
	assert(isActive());

	assert(_lcp->w[0] == 0.0);
	// assert(_lcp->findex[index] == -1);

	double bouncingVel = -mViolation;

	bouncingVel *= _lcp->invTimeStep * mErrorReductionParameter;

	if (bouncingVel > mMaxErrorReductionVelocity)
		bouncingVel = mMaxErrorReductionVelocity;

	_lcp->b[0] = mNegativeVel + bouncingVel;
	_lcp->lo[0] = mLowerBound;
	_lcp->hi[0] = mUpperBound;

	if (mLifeTime)
		_lcp->x[0] = mOldX;
	else
		_lcp->x[0] = 0.0;
}
void MuscleLimitConstraint::applyUnitImpulse(std::size_t _index)
{
	assert(_index == 0 && "Invalid Index.");

	mSkeleton->clearConstraintImpulses();
	for (auto joint : mJoints)
	{
		std::size_t dof = joint->getNumDofs();
		for (int i = 0; i < dof; i++)
			joint->setConstraintImpulse(i, mdCdTheta[joint->getIndexInSkeleton(i)]);
	}
	for (auto bn : mBodyNodes)
		mSkeleton->updateBiasImpulse(bn);

	mSkeleton->updateVelocityChange();
	for (auto joint : mJoints)
	{
		std::size_t dof = joint->getNumDofs();
		for (int i = 0; i < dof; i++)
			joint->setConstraintImpulse(i, 0.0);
	}
}
void MuscleLimitConstraint::
	getVelocityChange(double *_delVel, bool _withCfm)
{
	assert(_delVel != nullptr && "Null pointer is not allowed.");

	Eigen::VectorXd vel_change(mSkeleton->getNumDofs());
	vel_change.setZero();
	if (mSkeleton->isImpulseApplied())
	{
		for (auto joint : mJoints)
		{
			std::size_t dof = joint->getNumDofs();
			for (int i = 0; i < dof; i++)
				vel_change[joint->getIndexInSkeleton(i)] = joint->getVelocityChange(i);
		}

		_delVel[0] = mdCdTheta.dot(vel_change);
	}
	else
		_delVel[0] = 0.0;
	// std::cout<<_delVel[0]<<std::endl;
	if (_withCfm)
	{
		_delVel[0] += _delVel[0] * mConstraintForceMixing;
	}
}
void MuscleLimitConstraint::
	excite()
{
	mSkeleton->setImpulseApplied(true);
}
void MuscleLimitConstraint::
	unexcite()
{
	mSkeleton->setImpulseApplied(false);
}
void MuscleLimitConstraint::
	applyImpulse(double *_lambda)
{

	Eigen::VectorXd constraint_impulse = _lambda[0] * mdCdTheta;
	for (auto joint : mJoints)
	{
		std::size_t dof = joint->getNumDofs();

		for (int i = 0; i < dof; i++)
		{
			// std::cout<<constraint_impulse[joint->getIndexInSkeleton(i)]<<"\t";
			joint->setConstraintImpulse(i, joint->getConstraintImpulse(i) + constraint_impulse[joint->getIndexInSkeleton(i)]);
		}
	}
	// std::cout<<std::endl;
	mOldX = _lambda[0];
}
dart::dynamics::SkeletonPtr
MuscleLimitConstraint::
	getRootSkeleton() const
{
	return mSkeleton->mUnionRootSkeleton.lock();
}
bool MuscleLimitConstraint::
	isActive() const
{
	return mActive;
}
