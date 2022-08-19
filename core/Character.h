#ifndef __MASS_CHARACTER_H__
#define __MASS_CHARACTER_H__
#include "dart/dart.hpp"
#include "MuscleLimitConstraint.h"
#include "SimpleMotion.h"
#include <utility>

namespace MASS
{
	struct MirrorPair
	{
		int origin_idx;
		int opposite_idx;
		int dof;
		bool isAlign;
	};

	class BVH;
	class Muscle;

	struct ModifyInfo
	{
		ModifyInfo() : ModifyInfo(1.0, 1.0, 1.0, 1.0, 0.0) {}
		ModifyInfo(double lx, double ly, double lz, double sc, double to)
		{
			value[0] = lx;
			value[1] = ly;
			value[2] = lz;
			value[3] = sc;
			value[4] = to;
		};
		double value[5];
		double &operator[](int idx) { return value[idx]; }
		double operator[](int idx) const { return value[idx]; }
	};
	using BoneInfo = std::tuple<std::string, ModifyInfo>;

	class Character
	{
	public:
		Character();

		void LoadSkeleton(const std::string &path, double kp, bool create_obj = false);
		void LoadMuscles(const std::string &path);
		void LoadBVH(const std::string &path, bool cyclic = true);

		void Reset();
		void SetPDParameters(double kp, double kv);
		void SetPDParameters(Eigen::VectorXd kp, Eigen::VectorXd kv);
		void AddEndEffector(const std::string &body_name) { mEndEffectors.push_back(mSkeleton->getBodyNode(body_name)); }

		Eigen::VectorXd GetSPDForces(const Eigen::VectorXd &p_desired, bool isIncludeVel = true);

		Eigen::VectorXd GetTargetPositions(double t, double dt, dart::dynamics::SkeletonPtr skeleton);
		std::pair<Eigen::VectorXd, Eigen::VectorXd> GetTargetPosAndVel(double t, double dt, dart::dynamics::SkeletonPtr skeleton);

		Eigen::VectorXd GetTargetPositions(double t, double dt);
		std::pair<Eigen::VectorXd, Eigen::VectorXd> GetTargetPosAndVel(double t, double dt);

		const dart::dynamics::SkeletonPtr &GetSkeleton() { return mSkeleton; }
		const std::vector<Muscle *> &GetMuscles() { return mMuscles; }
		const std::vector<dart::dynamics::BodyNode *> &GetEndEffectors() { return mEndEffectors; }
		BVH *GetBVH() { return mBVH; }

		void AddMuscleLimitConstraint(const std::shared_ptr<MuscleLimitConstraint> &cst) { mMuscleLimitConstraints.push_back(cst); }
		const std::vector<std::shared_ptr<MuscleLimitConstraint>> &GetMuscleLimitConstraints() { return mMuscleLimitConstraints; };

		void AddChangedMuscleIdx(int idx) { mChangedMuscleIdx.push_back(idx); }
		std::vector<int> GetChangedMuscleIdx() { return mChangedMuscleIdx; }

		void AddChangedMuscle(Muscle *muscle_elem) { mChangedMuscles.push_back(muscle_elem); }

		std::vector<Muscle *> GetChangedMuscles() { return mChangedMuscles; }

		void CheckMirrorPair(dart::dynamics::SkeletonPtr skeleton);

		Eigen::VectorXd GetMirrorPosition(Eigen::VectorXd Position);
		Eigen::VectorXd GetMirrorActRange(Eigen::VectorXd ActRange);

		std::vector<MirrorPair> mMirrorPairs;

		void SetMirrorMotion();
		void SetUseNewSPD(bool usenewspd) { mUseNewSPD = usenewspd; }

	public:
		dart::dynamics::SkeletonPtr mSkeleton, mStdSkeleton;
		BVH *mBVH;
		Eigen::Isometry3d mTc;

		std::vector<Muscle *> mMuscles, mStdMuscles;
		std::vector<dart::dynamics::BodyNode *> mEndEffectors;

		std::vector<std::shared_ptr<MuscleLimitConstraint>> mMuscleLimitConstraints;

		std::vector<Muscle *> mChangedMuscles;

		Eigen::VectorXd mKp, mKv;

		std::vector<int> mChangedMuscleIdx;

		static std::vector<BoneInfo> LoadSkelParamFile(const std::string &filename);
		void ModifySkeletonBodyNode(const std::vector<BoneInfo> &info, dart::dynamics::SkeletonPtr skel);
		void ModifySkeletonLength(const std::vector<BoneInfo> &info);
		void SetSimpleMotion(const std::string &simplemotion, const std::string &jointmap);

		double mGlobalRatio;

	private:
		bool mUseOBJ;
		double motionScale, yOffset, rootDefaultHeight, footDifference;
		std::map<std::string, std::vector<SimpleMotion *>> muscleToSimpleMotions;
		std::map<dart::dynamics::BodyNode *, ModifyInfo> modifyLog;

		double calculateMetric(Muscle *stdMuscle, Muscle *rtgMuscle, const std::vector<SimpleMotion *> &simpleMotions, const Eigen::EIGEN_VV_VEC3D &x0);
		double fShape(Muscle *stdMuscle, Muscle *rtgMuscle);
		double fLengthCurve(double minPhaseDiff, double maxPhaseDiff, double lengthDiff);
		double fRegularizer(Muscle *rtgMuscle, const Eigen::EIGEN_VV_VEC3D &x0);

		double mUseNewSPD;
	};
};

#endif