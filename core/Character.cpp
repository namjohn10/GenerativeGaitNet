#include "Character.h"
#include "BVH.h"
#include "DARTHelper.h"
#include "Muscle.h"
#include <tinyxml2.h>
using namespace dart;
using namespace dart::dynamics;
using namespace MASS;

typedef tinyxml2::XMLElement TiXmlElement;
typedef tinyxml2::XMLDocument TiXmlDocument;

Character::
	Character()
	: mSkeleton(nullptr), mBVH(nullptr), mTc(Eigen::Isometry3d::Identity()), motionScale(1.0), mUseNewSPD(false), mGlobalRatio(1.0)
{
}

void Character::
	LoadSkeleton(const std::string &path, double kp, bool create_obj)
{

	mUseOBJ = create_obj;
	mSkeleton = BuildFromFile(path, create_obj);
	mStdSkeleton = BuildFromFile(path, create_obj);

	mKp = Eigen::VectorXd::Ones(mSkeleton->getNumDofs());
	mKv = Eigen::VectorXd::Ones(mSkeleton->getNumDofs());

	for (BodyNode *bodynode : mSkeleton->getBodyNodes())
		modifyLog[bodynode] = ModifyInfo();
	rootDefaultHeight = mSkeleton->getRootBodyNode()->getTransform().translation()[1];
	yOffset = -1e18;
	yOffset = std::max(yOffset, -mSkeleton->getBodyNode("TalusR")->getCOM()[1] + 0.04);

	std::map<std::string, std::string> bvh_map;
	TiXmlDocument doc;
	doc.LoadFile(path.c_str());
	TiXmlElement *skel_elem = doc.FirstChildElement("Skeleton");

	for (TiXmlElement *node = skel_elem->FirstChildElement("Node"); node != nullptr; node = node->NextSiblingElement("Node"))
	{
		if (node->Attribute("endeffector") != nullptr)
		{
			std::string ee = node->Attribute("endeffector");
			if (ee == "True")
			{
				mEndEffectors.push_back(mSkeleton->getBodyNode(std::string(node->Attribute("name"))));
			}
		}

		TiXmlElement *joint_elem = node->FirstChildElement("Joint");
		int dof = mSkeleton->getJoint(node->Attribute("name"))->getNumDofs();

		if (joint_elem->Attribute("bvh") != nullptr)
			bvh_map.insert(std::make_pair(node->Attribute("name"), joint_elem->Attribute("bvh")));

		if (dof == 0)
			continue;

		int idx = mSkeleton->getJoint(node->Attribute("name"))->getIndexInSkeleton(0);

		if (joint_elem->Attribute("kp") != nullptr)
		{
			if (dof == 1)
			{
				mKp[idx] = std::stod(joint_elem->Attribute("kp"));
				if (joint_elem->Attribute("kv") != nullptr)
					mKv[idx] = std::stod(joint_elem->Attribute("kv"));
				else
					mKv[idx] = sqrt(2 * mKp[idx]);
			}
			else if (dof == 3)
			{
				mKp.segment(idx, dof) = string_to_vector3d(joint_elem->Attribute("kp"));
				if (joint_elem->Attribute("kv") != nullptr)
					mKv.segment(idx, dof) = string_to_vector3d(joint_elem->Attribute("kv"));
				else
				{
					mKv[idx] = sqrt(2 * mKp[idx]);
					mKv[idx + 1] = sqrt(2 * mKp[idx + 1]);
					mKv[idx + 2] = sqrt(2 * mKp[idx + 2]);
				}
			}
		}
		else
		{
			for (int i = idx; i < idx + dof; i++)
			{
				mKp[i] = kp;
				mKv[i] = sqrt(2 * kp);
			}
		}
	}

	mKp.head(6).setZero();
	mKv.head(6).setZero();

	mBVH = new BVH(mSkeleton, bvh_map);
}

void Character::
	LoadMuscles(const std::string &path)
{
	TiXmlDocument doc;
	if (doc.LoadFile(path.c_str()))
	{
		std::cout << "Can't open file : " << path << std::endl;
		return;
	}

	TiXmlElement *muscledoc = doc.FirstChildElement("Muscle");
	for (TiXmlElement *unit = muscledoc->FirstChildElement("Unit"); unit != nullptr; unit = unit->NextSiblingElement("Unit"))
	{
		std::string name = unit->Attribute("name");
		double f0 = std::stod(unit->Attribute("f0"));
		double lm = std::stod(unit->Attribute("lm"));
		double lt = std::stod(unit->Attribute("lt"));
		double pa = std::stod(unit->Attribute("pen_angle"));
		double lmax = std::stod(unit->Attribute("lmax"));
		double type1_fraction = 0.5;
		if (unit->Attribute("type1_fraction") != nullptr)
			type1_fraction = std::stod(unit->Attribute("type1_fraction"));

		Muscle *muscle_elem = new Muscle(name, f0, lm, lt, pa, lmax, type1_fraction);
		Muscle *stdmuscle_elem = new Muscle(name, f0, lm, lt, pa, lmax, type1_fraction);
		bool isValid = true;
		int num_waypoints = 0;
		for (TiXmlElement *waypoint = unit->FirstChildElement("Waypoint"); waypoint != nullptr; waypoint = waypoint->NextSiblingElement("Waypoint"))
			num_waypoints++;
		int i = 0;
		for (TiXmlElement *waypoint = unit->FirstChildElement("Waypoint"); waypoint != nullptr; waypoint = waypoint->NextSiblingElement("Waypoint"))
		{
			std::string body = waypoint->Attribute("body");
			Eigen::Vector3d glob_pos = string_to_vector3d(waypoint->Attribute("p"));
			if (mSkeleton->getBodyNode(body) == NULL)
			{
				isValid = false;
				break;
			}

			if (i == 0 || i == num_waypoints - 1)
			{
				muscle_elem->AddAnchor(mSkeleton->getBodyNode(body), glob_pos);
				stdmuscle_elem->AddAnchor(mStdSkeleton->getBodyNode(body), glob_pos);
			}
			else
			{
				muscle_elem->AddAnchor(mSkeleton, mSkeleton->getBodyNode(body), glob_pos, 2);
				stdmuscle_elem->AddAnchor(mStdSkeleton, mStdSkeleton->getBodyNode(body), glob_pos, 2);
			}

			i++;
		}
		if (isValid)
		{
			muscle_elem->SetMuscle();
			if (muscle_elem->GetNumRelatedDofs() > 0)
			{
				stdmuscle_elem->SetMuscle();
				mMuscles.push_back(muscle_elem);
				mStdMuscles.push_back(stdmuscle_elem);
			}
		}
	}
}
void Character::
	LoadBVH(const std::string &path, bool cyclic)
{
	if (mBVH == nullptr)
	{
		std::cout << "Initialize BVH class first" << std::endl;
		return;
	}
	mBVH->Parse(path, cyclic);
}
void Character::
	Reset()
{
	mTc = mBVH->GetT0();
	mTc.translation()[1] = 0.0;
}
void Character::
	SetPDParameters(double kp, double kv)
{
	int dof = mSkeleton->getNumDofs();
	mKp = Eigen::VectorXd::Constant(dof, kp);
	mKv = Eigen::VectorXd::Constant(dof, kv);
}

void Character::
	SetPDParameters(Eigen::VectorXd kp, Eigen::VectorXd kv)
{
	if ((mKp.rows() != kp.rows()) || (mKv.rows() != kv.rows()))
	{

		exit(-1);
	}

	mKp = kp;
	mKv = kv;
}

Eigen::VectorXd
Character::
	GetSPDForces(const Eigen::VectorXd &p_desired, bool isIncludeVel)
{
	Eigen::VectorXd q = mSkeleton->getPositions();
	Eigen::VectorXd dq = mSkeleton->getVelocities();
	double dt = mSkeleton->getTimeStep();
	Eigen::MatrixXd M_inv = (mSkeleton->getMassMatrix() + Eigen::MatrixXd(dt * sqrt(mGlobalRatio) * mKv.asDiagonal())).inverse();
	Eigen::VectorXd qdqdt = q + dq * dt;

	Eigen::VectorXd p_diff = -mGlobalRatio * mKp.cwiseProduct(mSkeleton->getPositionDifferences(qdqdt, p_desired));
	if (mUseNewSPD)
		p_diff.head(6).setZero();

	Eigen::VectorXd v_diff = -sqrt(mGlobalRatio) * mKv.cwiseProduct(dq);

	Eigen::VectorXd ddq = M_inv * (-mSkeleton->getCoriolisAndGravityForces() + p_diff + v_diff + mSkeleton->getConstraintForces());
	Eigen::VectorXd tau = p_diff + v_diff - sqrt(mGlobalRatio) * dt * mKv.cwiseProduct(ddq);

	tau.head<6>().setZero();

	return tau;
}

Eigen::VectorXd Character::GetTargetPositions(double t, double dt)
{

	Eigen::VectorXd p = mBVH->GetModifiedMotion(t);
	if (mBVH->IsCyclic())
	{
		int k = (int)(t / mBVH->GetMaxTime());
		Eigen::Vector3d bvh_vec = mBVH->GetT1().translation() - mBVH->GetT0().translation();
		bvh_vec[1] = 0.;
		p.segment<3>(3) += k * bvh_vec;
	}

	p.segment<3>(3) -= mSkeleton->getRootJoint()->getTransformFromParentBodyNode().translation();
	p[3] *= motionScale;
	p[4] += yOffset;
	p[5] *= motionScale;
	return p;
}

std::pair<Eigen::VectorXd, Eigen::VectorXd>
Character::GetTargetPosAndVel(double t, double dt)
{

	Eigen::VectorXd p = this->GetTargetPositions(t, dt);

	Eigen::VectorXd p1 = this->GetTargetPositions(t + dt, dt);

	p[4] += 0.035;

	return std::make_pair(p, mSkeleton->getPositionDifferences(p1, p) / dt);
}

Eigen::VectorXd Character::GetTargetPositions(double t, double dt, dart::dynamics::SkeletonPtr skeleton)
{
	Eigen::VectorXd p = mBVH->GetMotion(t, skeleton);
	if (mBVH->IsCyclic())
	{
		int k = (int)(t / mBVH->GetMaxTime());
		Eigen::Vector3d bvh_vec = mBVH->GetT1().translation() - mBVH->GetT0().translation();
		bvh_vec[1] = 0.;
		p.segment<3>(3) += k * bvh_vec;
	}
	p.segment<3>(3) -= mSkeleton->getRootJoint()->getTransformFromParentBodyNode().translation();

	return p;
}

std::pair<Eigen::VectorXd, Eigen::VectorXd>
Character::GetTargetPosAndVel(double t, double dt, dart::dynamics::SkeletonPtr skeleton)
{
	Eigen::VectorXd p = this->GetTargetPositions(t, dt, skeleton);
	Eigen::VectorXd p1 = this->GetTargetPositions(t + dt, dt, skeleton);

	return std::make_pair(p, skeleton->getPositionDifferences(p1, p) / dt);
}

Eigen::VectorXd
Character::
	GetMirrorActRange(Eigen::VectorXd ActRange)
{
	int m_num = ActRange.rows() / 2;
	Eigen::VectorXd ActMin(m_num);
	Eigen::VectorXd ActMax(m_num);

	ActMin.setZero();
	ActMax.setZero();

	ActMin = ActRange.head(m_num);
	ActMax = ActRange.tail(m_num);

	Eigen::VectorXd ActMin_res = ActMin;
	Eigen::VectorXd ActMax_res = ActMax;
	Eigen::VectorXd ActTotal_res(m_num * 2);

	int rootDof = 6;
	for (auto elem : mMirrorPairs)
	{
		if (elem.dof == 3)
		{
			ActMin_res[elem.opposite_idx - rootDof] = ActMin[elem.origin_idx - rootDof];
			ActMax_res[elem.opposite_idx + 1 - rootDof] = -ActMin[elem.origin_idx + 1 - rootDof];
			ActMax_res[elem.opposite_idx + 2 - rootDof] = -ActMin[elem.origin_idx + 2 - rootDof];

			ActMax_res[elem.opposite_idx - rootDof] = ActMax[elem.origin_idx - rootDof];
			ActMin_res[elem.opposite_idx + 1 - rootDof] = -ActMax[elem.origin_idx + 1 - rootDof];
			ActMin_res[elem.opposite_idx + 2 - rootDof] = -ActMax[elem.origin_idx + 2 - rootDof];
		}
		else if (elem.dof == 1)
		{
			if (!elem.isAlign)
			{
				ActMax_res[elem.opposite_idx - rootDof] = -ActMin[elem.origin_idx - rootDof];
				ActMin_res[elem.opposite_idx - rootDof] = -ActMax[elem.origin_idx - rootDof];
			}
			else
			{
				ActMax_res[elem.opposite_idx - rootDof] = ActMax[elem.origin_idx - rootDof];
				ActMin_res[elem.opposite_idx - rootDof] = ActMin[elem.origin_idx - rootDof];
			}
		}
	}

	ActTotal_res << ActMin_res, ActMax_res;
	return ActTotal_res;
}

Eigen::VectorXd
Character::
	GetMirrorPosition(const Eigen::VectorXd Position)
{
	Eigen::VectorXd mirrorPosition = Position;

	for (auto elem : mMirrorPairs)
	{
		if (elem.dof == 6)
		{

			Eigen::Vector3d des = Position.segment<3>(elem.origin_idx);
			des[1] *= -1;
			des[2] *= -1;
			mirrorPosition.segment<3>(elem.opposite_idx) = des;
			mirrorPosition[3] = -Position[3];
		}
		if (elem.dof == 3)
		{

			Eigen::Vector3d des = Position.segment<3>(elem.origin_idx);
			des[1] *= -1;
			des[2] *= -1;
			mirrorPosition.segment<3>(elem.opposite_idx) = des;
		}
		else if (elem.dof == 1)
		{
			if (!elem.isAlign)
				mirrorPosition[elem.opposite_idx] = -Position[elem.origin_idx];
			else
				mirrorPosition[elem.opposite_idx] = Position[elem.origin_idx];
		}
	}
	return mirrorPosition;
}

void Character::
	CheckMirrorPair(dart::dynamics::SkeletonPtr skeleton)
{
	auto joints = skeleton->getJoints();

	for (int i = 0; i < joints.size(); i++)
	{
		if (joints[i]->getNumDofs() == 0)
			continue;
		if (joints[i]->getName().back() == 'R' || joints[i]->getName().back() == 'L')
		{
			for (int j = i + 1; j < joints.size(); j++)
			{
				if (joints[i]->getName().substr(0, joints[i]->getName().length() - 1) == joints[j]->getName().substr(0, joints[j]->getName().length() - 1))
				{
					MirrorPair elem_i;

					elem_i.origin_idx = joints[i]->getIndexInSkeleton(0);
					elem_i.opposite_idx = joints[j]->getIndexInSkeleton(0);
					elem_i.dof = joints[i]->getNumDofs();
					elem_i.isAlign = true;
					if (joints[i]->getName().find("ForeArm") != std::string::npos)
						elem_i.isAlign = false;

					MirrorPair elem_j;
					elem_j.origin_idx = joints[j]->getIndexInSkeleton(0);
					elem_j.opposite_idx = joints[i]->getIndexInSkeleton(0);
					elem_j.dof = joints[i]->getNumDofs();

					elem_j.isAlign = true;
					if (joints[i]->getName().find("ForeArm") != std::string::npos)
						elem_j.isAlign = false;

					mMirrorPairs.push_back(elem_i);
					mMirrorPairs.push_back(elem_j);

					break;
				}
			}
		}
		else
		{
			MirrorPair elem;

			elem.origin_idx = joints[i]->getIndexInSkeleton(0);
			elem.opposite_idx = joints[i]->getIndexInSkeleton(0);
			elem.dof = joints[i]->getNumDofs();
			elem.isAlign = true;
			mMirrorPairs.push_back(elem);
		}
	}
}

void Character::
	SetMirrorMotion()
{
	std::vector<Eigen::VectorXd> pureMotions = mBVH->mPureMotions;
	std::vector<Eigen::VectorXd> mirrorMotions;
	int n = pureMotions.size();
	Eigen::Vector3d pos(0, 0, 0);
	for (int i = 0; i < n; i++)
	{
		int idx = (i + n / 2) % n;
		Eigen::VectorXd m = GetMirrorPosition(pureMotions[idx]);
		m.segment<3>(3) += pos;
		if (idx == (n - 1))
		{
			pos = m.segment<3>(3);
			pos[1] = 0;
		}
		mirrorMotions.push_back(m);
	}
	Eigen::Vector3d p0 = mirrorMotions[0].segment<3>(3);
	for (auto &m : mirrorMotions)
		m.segment<3>(3) -= Eigen::Vector3d(p0[0], 0, p0[2]);

	Eigen::Vector3d d(0, 0, 0);
	d[0] = (pureMotions.back()[3] - pureMotions[0][3]) + (mirrorMotions.back()[3] - mirrorMotions[0][3]);
	d[2] = -(pureMotions.back()[5] - pureMotions[0][5]) + (mirrorMotions.back()[5] - mirrorMotions[0][5]);

	for (int i = n / 2; i < n; i++)
		mirrorMotions[i].segment<3>(3) -= d;

	mBVH->setMirrorMotions(mirrorMotions);
	mBVH->blendMirrorMotion();
}

static std::tuple<Eigen::Vector3d, double, double> UnfoldModifyInfo(const MASS::ModifyInfo &info)
{
	return std::make_tuple(Eigen::Vector3d(info[0], info[1], info[2]), info[3], info[4]);
}

static Eigen::Isometry3d modifyIsometry3d(const Eigen::Isometry3d &iso, const MASS::ModifyInfo &info, int axis, bool rotate = true)
{
	Eigen::Vector3d l;
	double s, t;
	std::tie(l, s, t) = UnfoldModifyInfo(info);
	Eigen::Vector3d translation = iso.translation();
	translation = translation.cwiseProduct(l);
	translation *= s;
	auto tmp = Eigen::Isometry3d(Eigen::Translation3d(translation));
	tmp.linear() = iso.linear();

	return tmp;
}

static void modifyShapeNode(BodyNode *rtgBody, BodyNode *stdBody, const MASS::ModifyInfo &info, int axis)
{
	Eigen::Vector3d l;
	double s, t;
	std::tie(l, s, t) = UnfoldModifyInfo(info);
	double la = l[axis], lb = l[(axis + 1) % 3], lc = l[(axis + 2) % 3];

	for (int i = 0; i < rtgBody->getNumShapeNodes(); i++)
	{
		ShapeNode *rtgShape = rtgBody->getShapeNode(i), *stdShape = stdBody->getShapeNode(i);
		ShapePtr newShape;
		if (auto rtg = std::dynamic_pointer_cast<CapsuleShape>(rtgShape->getShape()))
		{
			auto std = std::dynamic_pointer_cast<CapsuleShape>(stdShape->getShape());
			double radius = std->getRadius() * s * (lb + lc) / 2, height = std->getHeight() * s * la;
			newShape = ShapePtr(new CapsuleShape(radius, height));
		}
		else if (auto rtg = std::dynamic_pointer_cast<SphereShape>(rtgShape->getShape()))
		{
			auto std = std::dynamic_pointer_cast<SphereShape>(stdShape->getShape());
			double radius = std->getRadius() * s * (la + lb + lc) / 3;
			newShape = ShapePtr(new SphereShape(radius));
		}
		else if (auto rtg = std::dynamic_pointer_cast<CylinderShape>(rtgShape->getShape()))
		{
			auto std = std::dynamic_pointer_cast<CylinderShape>(stdShape->getShape());
			double radius = std->getRadius() * s * (lb + lc) / 2, height = std->getHeight() * s * la;
			newShape = ShapePtr(new CylinderShape(radius, height));
		}
		else if (std::dynamic_pointer_cast<BoxShape>(rtgShape->getShape()))
		{
			auto std = std::dynamic_pointer_cast<BoxShape>(stdShape->getShape());
			Eigen::Vector3d size = std->getSize() * s;
			size = size.cwiseProduct(l);
			newShape = ShapePtr(new BoxShape(size));
		}
		else if (auto rtg = std::dynamic_pointer_cast<MeshShape>(rtgShape->getShape()))
		{
			auto std = std::dynamic_pointer_cast<MeshShape>(stdShape->getShape());
			Eigen::Vector3d scale = std->getScale();
			scale *= s;
			scale = scale.cwiseProduct(l);
			rtg->setScale(scale);
			Eigen::Isometry3d s = stdShape->getRelativeTransform(), r = modifyIsometry3d(s.inverse(), info, axis).inverse();
			rtgShape->setRelativeTransform(r);
			newShape = rtg;
		}
		rtgShape->setShape(newShape);
	}
	ShapePtr shape = rtgBody->getShapeNodesWith<DynamicsAspect>()[0]->getShape();
	double mass = stdBody->getMass() * l[0] * l[1] * l[2] * s * s * s;
	dart::dynamics::Inertia inertia;
	inertia.setMass(mass);
	inertia.setMoment(shape->computeInertia(mass));
	rtgBody->setInertia(inertia);
}

static std::map<std::string, int> skeletonAxis = {
	{"Pelvis", 1},
	{"FemurR", 1},
	{"TibiaR", 1},
	{"TalusR", 2},

	{"FootThumbR", 2},
	{"FootPinkyR", 2},
	{"FemurL", 1},
	{"TibiaL", 1},
	{"TalusL", 2},

	{"FootThumbL", 2},
	{"FootPinkyL", 2},
	{"Spine", 1},
	{"Torso", 1},
	{"Neck", 1},
	{"Head", 1},
	{"ShoulderR", 0},
	{"ArmR", 0},
	{"ForeArmR", 0},
	{"HandR", 0},
	{"ShoulderL", 0},
	{"ArmL", 0},
	{"ForeArmL", 0},
	{"HandL", 0},
};

void MASS::Character::ModifySkeletonBodyNode(const std::vector<BoneInfo> &info, dart::dynamics::SkeletonPtr skel)
{
	for (auto bone : info)
	{
		std::string name;
		ModifyInfo info;
		std::tie(name, info) = bone;
		int axis = skeletonAxis[name];
		BodyNode *rtgBody = skel->getBodyNode(name);
		BodyNode *stdBody = mStdSkeleton->getBodyNode(name);
		if (rtgBody == NULL)
			continue;
		if (mUseOBJ)
		{
			modifyShapeNode(rtgBody, stdBody, info, axis);
		}

		if (Joint *rtgParent = rtgBody->getParentJoint())
		{
			Joint *stdParent = stdBody->getParentJoint();
			Eigen::Isometry3d up = stdParent->getTransformFromChildBodyNode();
			rtgParent->setTransformFromChildBodyNode(modifyIsometry3d(up, info, axis));
		}

		for (int i = 0; i < rtgBody->getNumChildJoints(); i++)
		{
			Joint *rtgJoint = rtgBody->getChildJoint(i);
			Joint *stdJoint = stdBody->getChildJoint(i);
			Eigen::Isometry3d down = stdJoint->getTransformFromParentBodyNode();
			rtgJoint->setTransformFromParentBodyNode(modifyIsometry3d(down, info, axis, false));
		}
	}
}

void MASS::Character::ModifySkeletonLength(const std::vector<BoneInfo> &info)
{
	for (auto bone : info)
	{
		std::string name;
		ModifyInfo info;
		std::tie(name, info) = bone;
		modifyLog[mSkeleton->getBodyNode(name)] = info;
	}
	ModifySkeletonBodyNode(info, mSkeleton);

	Eigen::VectorXd positions = mSkeleton->getPositions();
	mSkeleton->setPositions(Eigen::VectorXd::Zero(mSkeleton->getNumDofs()));
	mSkeleton->computeForwardKinematics(true, false, false);
	mStdSkeleton->setPositions(Eigen::VectorXd::Zero(mSkeleton->getNumDofs()));
	mStdSkeleton->computeForwardKinematics(true, false, false);

	double currentLegLength = mSkeleton->getBodyNode("Pelvis")->getCOM()[1] - mSkeleton->getBodyNode("TalusL")->getCOM()[1];
	double originalLegLength = mStdSkeleton->getBodyNode("Pelvis")->getCOM()[1] - mStdSkeleton->getBodyNode("TalusL")->getCOM()[1];
	motionScale = currentLegLength / originalLegLength;

	double prevOffset = yOffset;
	yOffset = -1e18;
	for (const auto &foot : {"TalusR", "TalusL"})
	{
		yOffset = std::max(yOffset, -mSkeleton->getBodyNode(foot)->getCOM()[1] + 0.04);
	}
	positions[4] += yOffset - prevOffset;
	footDifference = 0;
	for (const auto &foot : {"TalusR", "TalusL"})
	{
		footDifference = std::max(footDifference, yOffset - (-mSkeleton->getBodyNode(foot)->getCOM()[1] + 0.04));
	}
	rootDefaultHeight = mSkeleton->getRootBodyNode()->getTransform().translation()[1] + yOffset;

	for (int i = 0; i < mMuscles.size(); i++)
	{
		Muscle *mMuscle = mMuscles[i], *mStdMuscle = mStdMuscles[i];
		for (int j = 0; j < mMuscle->GetAnchors().size(); j++)
		{
			Anchor *mAnchor = mMuscle->GetAnchors()[j], *mStdAnchor = mStdMuscle->GetAnchors()[j];
			for (int k = 0; k < mAnchor->bodynodes.size(); k++)
			{
				BodyNode *mBody = mAnchor->bodynodes[k];
				int axis = skeletonAxis[mBody->getName()];
				auto cur = Eigen::Isometry3d(Eigen::Translation3d(mStdAnchor->local_positions[k]));
				Eigen::Isometry3d tmp = modifyIsometry3d(cur, modifyLog[mBody], axis);
				mAnchor->local_positions[k] = tmp.translation();
			}
		}
		mMuscle->SetMuscle();
		mMuscle->f0_original = mMuscle->f0 = mStdMuscle->f0 * pow(mMuscle->l_mt0 / mStdMuscle->l_mt0, 1.5);
	}

	double eps = 5e-5;
	std::vector<double> derivative;
	for (int muscleIdx = 0; muscleIdx < mMuscles.size(); muscleIdx++)
	{
		auto stdMuscle = mStdMuscles[muscleIdx];
		auto rtgMuscle = mMuscles[muscleIdx];
		int numAnchors = rtgMuscle->mAnchors.size();
		if (muscleToSimpleMotions.find(rtgMuscle->name) == muscleToSimpleMotions.end())
			continue;
		const std::vector<SimpleMotion *> &simpleMotions = muscleToSimpleMotions.find(rtgMuscle->name)->second;
		if (simpleMotions.size() == 0 || numAnchors == 2)
			continue;

		std::vector<std::vector<Eigen::Vector3d>> x0(numAnchors - 2);
		for (int i = 1; i + 1 < numAnchors; i++)
		{
			Anchor *anchor = rtgMuscle->mAnchors[i];
			for (int j = 0; j < anchor->local_positions.size(); j++)
			{
				x0[i - 1].push_back(anchor->local_positions[j]);
			}
		}

		int rep;
		for (rep = 0; rep < 20; rep++)
		{
			double currentDifference = calculateMetric(stdMuscle, rtgMuscle, simpleMotions, x0);

			derivative.clear();
			for (int i = 1; i + 1 < rtgMuscle->mAnchors.size(); i++)
			{
				Anchor *anchor = rtgMuscle->mAnchors[i];
				for (int j = 0; j < anchor->local_positions.size(); j++)
				{
					for (int dir = 0; dir < 3; dir++)
					{
						double dx = 0;
						anchor->local_positions[j][dir] += eps;
						rtgMuscle->SetMuscle();
						dx += calculateMetric(stdMuscle, rtgMuscle, simpleMotions, x0);
						anchor->local_positions[j][dir] -= eps * 2;
						rtgMuscle->SetMuscle();
						dx -= calculateMetric(stdMuscle, rtgMuscle, simpleMotions, x0);
						anchor->local_positions[j][dir] += eps;
						rtgMuscle->SetMuscle();
						derivative.push_back(dx / (eps * 2));
					}
				}
			}

			double alpha = 0.1;

			int lineStep;
			for (lineStep = 0; lineStep < 32; lineStep++)
			{
				for (int i = 1, derivativeIdx = 0; i + 1 < rtgMuscle->mAnchors.size(); i++)
				{
					Anchor *anchor = rtgMuscle->mAnchors[i];
					for (int j = 0; j < anchor->local_positions.size(); j++)
					{
						for (int dir = 0; dir < 3; dir++, derivativeIdx++)
						{
							anchor->local_positions[j][dir] -= alpha * derivative[derivativeIdx];
						}
					}
				}
				rtgMuscle->SetMuscle();

				double nextDifference = calculateMetric(stdMuscle, rtgMuscle, simpleMotions, x0);
				if (nextDifference < currentDifference * 0.999)
					break;

				for (int i = 1, derivativeIdx = 0; i + 1 < rtgMuscle->mAnchors.size(); i++)
				{
					Anchor *anchor = rtgMuscle->mAnchors[i];
					for (int j = 0; j < anchor->local_positions.size(); j++)
					{
						for (int dir = 0; dir < 3; dir++, derivativeIdx++)
						{
							anchor->local_positions[j][dir] += alpha * derivative[derivativeIdx];
						}
					}
				}
				rtgMuscle->SetMuscle();
				alpha *= 0.5;
			}
			if (lineStep == 32)
				break;
		}
	}
	for (int i = 0; i < mMuscles.size(); i++)
	{
		Muscle *mMuscle = mMuscles[i], *mStdMuscle = mStdMuscles[i];
		mMuscle->SetMuscle();
		mMuscle->f0_original = mMuscle->f0 = mStdMuscle->f0 * pow(mMuscle->l_mt0 / mStdMuscle->l_mt0, 1.5);
	}

	mSkeleton->setPositions(positions);
	mSkeleton->computeForwardKinematics(true, false, false);
	mStdSkeleton->setPositions(Eigen::VectorXd::Zero(mSkeleton->getNumDofs()));
}

std::vector<BoneInfo> MASS::Character::LoadSkelParamFile(const std::string &filename)
{
	std::vector<BoneInfo> infos;
	tinyxml2::XMLDocument doc;
	if (doc.LoadFile(filename.c_str()))
	{
		std::cout << "Can't open/parse file : " << filename << std::endl;
		throw std::invalid_argument("In func ModifyInfo");
	}
	tinyxml2::XMLElement *infoxml = doc.FirstChildElement("ModifyInfo");

	for (tinyxml2::XMLElement *boneXML = infoxml->FirstChildElement("Bone"); boneXML; boneXML = boneXML->NextSiblingElement("Bone"))
	{
		std::string body = std::string(boneXML->Attribute("body"));
		std::stringstream ss(boneXML->Attribute("info"));
		ModifyInfo info;
		for (int i = 0; i < 5; i++)
			ss >> info[i];
		infos.emplace_back(body, info);
	}
	return infos;
}

static std::map<std::string, int> readJointMap(const std::string &filename, dart::dynamics::SkeletonPtr skel)
{
	FILE *in = fopen(filename.c_str(), "r");
	std::map<std::string, int> jointMap;
	char line[1005];
	while (fgets(line, 100, in) != NULL)
	{
		std::stringstream linestream(line);
		std::string name, bnName;
		int idx;
		linestream >> name >> bnName >> idx;

		dart::dynamics::BodyNode *bn = skel->getBodyNode(bnName);
		if (bn == NULL)
			continue;

		if (bn->getParentJoint()->getNumDofs() == 0)
			continue;
		int offset = bn->getParentJoint()->getDof(0)->getIndexInSkeleton();
		jointMap[name] = offset + idx;
	}
	return jointMap;
}

static const double PI = acos(-1);
void MASS::Character::SetSimpleMotion(const std::string &simplemotion, const std::string &jointmap)
{
	std::map<std::string, int> jointMap;
	jointMap = readJointMap(jointmap, mSkeleton);
	FILE *in = fopen(simplemotion.c_str(), "r");
	char line[1005];
	MASS::SimpleMotion *currentMotion = NULL;
	std::set<std::string> validMotion;
	while (fgets(line, 100, in) != NULL)
	{
		std::stringstream linestream(line);
		std::string type = "#";
		linestream >> type;
		if (type == "n")
		{
			std::string motion;
			linestream >> motion;
			if (validMotion.find(motion) == validMotion.end())
				currentMotion = NULL;
			else
			{
				currentMotion = new MASS::SimpleMotion();

				currentMotion->motionName = motion;

				if (motion == "T_Pose" || motion == "Stand_Pose")
				{
					for (MASS::Muscle *muscle : mMuscles)
					{
						muscleToSimpleMotions[muscle->name].push_back(currentMotion);
					}
				}
			}
		}
		else if (type == "c" && currentMotion)
		{
			std::string joint;
			double s, e;
			linestream >> joint >> s >> e;

			s *= PI / 180;
			e *= PI / 180;
			int idx = jointMap[joint];

			currentMotion->idx.push_back(idx);
			currentMotion->start.push_back(s);
			currentMotion->end.push_back(e);
		}
		else if (type == "m" && currentMotion)
		{
			std::string muscleName;
			linestream >> muscleName;

			muscleToSimpleMotions[muscleName].push_back(currentMotion);
		}
		else if (type == "p")
		{
			std::string motionName;
			linestream >> motionName;
			validMotion.insert(motionName);
		}
	}
}

double MASS::Character::calculateMetric(Muscle *stdMuscle, Muscle *rtgMuscle, const std::vector<SimpleMotion *> &simpleMotions, const Eigen::EIGEN_VV_VEC3D &x0)
{
	double lambdaShape = 0.1;
	double lambdaLengthCurve = 1.0;
	double lambdaRegularizer = 0.1;

	double shapeTerm = 0;
	double lengthCurveTerm = 0;
	double regularizerTerm = 0;

	double ret = 0.0;
	int numSampling = 50;
	for (SimpleMotion *sm : simpleMotions)
	{
		std::pair<double, double> stdMin, stdMax, rtgMin, rtgMax;
		stdMin = rtgMin = std::pair<double, double>(1e10, 0);
		stdMax = rtgMax = std::pair<double, double>(-1e10, 0);

		for (int rep = 0; rep <= numSampling; rep++)
		{
			double phase = 1.0 * rep / numSampling;
			for (auto [idx, pose] : sm->getPose(phase))
			{
				mStdSkeleton->setPosition(idx, pose);
				mSkeleton->setPosition(idx, pose);
			}

			stdMuscle->Update();
			rtgMuscle->Update();
			shapeTerm += (rep == 0 || rep == numSampling ? 0.5 : 1) * fShape(stdMuscle, rtgMuscle);

			std::pair<double, double> stdLength = std::make_pair(stdMuscle->length / stdMuscle->l_mt0, phase);
			std::pair<double, double> rtgLength = std::make_pair(rtgMuscle->length / rtgMuscle->l_mt0, phase);
			stdMin = std::min(stdMin, stdLength);
			stdMax = std::max(stdMax, stdLength);
			rtgMin = std::min(rtgMin, rtgLength);
			rtgMax = std::max(rtgMax, rtgLength);
		}
		lengthCurveTerm += fLengthCurve(stdMin.second - rtgMin.second, stdMax.second - rtgMax.second,
										(stdMax.first - stdMin.first) - (rtgMax.first - rtgMin.first));

		for (auto [idx, pose] : sm->getPose(0))
		{
			mStdSkeleton->setPosition(idx, 0);
			mSkeleton->setPosition(idx, 0);
		}
	}
	regularizerTerm += fRegularizer(rtgMuscle, x0);

	int dof = mStdSkeleton->getNumDofs();
	mStdSkeleton->setPositions(Eigen::VectorXd::Zero(dof));
	mSkeleton->setPositions(Eigen::VectorXd::Zero(dof));

	return lambdaShape * shapeTerm / numSampling / simpleMotions.size() + lambdaLengthCurve * lengthCurveTerm / simpleMotions.size() + lambdaRegularizer * regularizerTerm;
}

double MASS::Character::fShape(Muscle *stdMuscle, Muscle *rtgMuscle)
{
	double ret = 0;
	int cnt = 0;
	for (int i = 0; i + 1 < stdMuscle->mAnchors.size(); i++)
	{
		Eigen::Vector3d stdVector, rtgVector;
		stdVector = stdMuscle->mCachedAnchorPositions[i + 1] -
					stdMuscle->mCachedAnchorPositions[i];
		rtgVector = rtgMuscle->mCachedAnchorPositions[i + 1] -
					rtgMuscle->mCachedAnchorPositions[i];
		stdVector.normalize();
		rtgVector.normalize();
		ret += (stdVector.cross(rtgVector)).norm();
		cnt += 1;
	}
	return cnt ? ret / cnt : 0;
}

double MASS::Character::fLengthCurve(double minPhaseDiff, double maxPhaseDiff, double lengthDiff)
{
	return 0.007 * pow(minPhaseDiff, 2) + 0.007 * pow(maxPhaseDiff, 2) + 0.5 * pow(lengthDiff, 2);
}

double MASS::Character::fRegularizer(Muscle *rtgMuscle, const Eigen::EIGEN_VV_VEC3D &x0)
{
	double total = 0;
	for (int i = 1; i + 1 < rtgMuscle->mAnchors.size(); i++)
	{
		Anchor *anchor = rtgMuscle->mAnchors[i];
		for (int j = 0; j < anchor->local_positions.size(); j++)
		{
			total += (anchor->local_positions[j] - x0[i - 1][j]).norm() * anchor->weights[j];
		}
	}
	return total;
}