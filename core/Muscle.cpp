#include "Muscle.h"

using namespace MASS;
using namespace dart::dynamics;
std::vector<int> sort_indices(const std::vector<double> &val)
{
	std::vector<int> idx(val.size());
	std::iota(idx.begin(), idx.end(), 0);

	std::sort(idx.begin(), idx.end(), [&val](int i1, int i2)
			  { return val[i1] < val[i2]; });

	return idx;
}
Anchor::
	Anchor(std::vector<BodyNode *> bns, std::vector<Eigen::Vector3d> lps, std::vector<double> ws)
	: bodynodes(bns), local_positions(lps), weights(ws), num_related_bodies(bns.size())
{
}

Eigen::Vector3d
Anchor::
	GetPoint()
{
	Eigen::Vector3d p;
	p.setZero();
	for (int i = 0; i < num_related_bodies; i++)
		p += weights[i] * (bodynodes[i]->getTransform() * local_positions[i]);
	return p;
}

Muscle::
	Muscle(std::string _name, double _f0, double _lm0, double _lt0, double _pen_angle, double lmax, double _type1_fraction)
	: selected(false), name(_name), f0_original(_f0), f0(_f0), l_m0(_lm0), l_m(l_mt - l_t0), l_t0(_lt0), l_mt0(0.0), l_mt(1.0), activation(0.0), f_toe(0.33), k_toe(3.0), k_lin(51.878788), e_toe(0.02), e_t0(0.033), k_pe(5.5), e_mo(0.3), gamma(0.45), l_mt_max(lmax), type1_fraction(_type1_fraction)
{
	l_min = 0.99;
	l_max = 1.00;
	f_min = 0.99;
	f_max = 1.00;
}

void Muscle::
	AddAnchor(const dart::dynamics::SkeletonPtr &skel, dart::dynamics::BodyNode *bn, const Eigen::Vector3d &glob_pos, int num_related_bodies)
{
	std::vector<double> distance;
	std::vector<Eigen::Vector3d> local_positions;
	distance.resize(skel->getNumBodyNodes(), 0.0);
	local_positions.resize(skel->getNumBodyNodes());
	for (int i = 0; i < skel->getNumBodyNodes(); i++)
	{
		Eigen::Isometry3d T;
		T = skel->getBodyNode(i)->getTransform() * skel->getBodyNode(i)->getParentJoint()->getTransformFromChildBodyNode();
		local_positions[i] = skel->getBodyNode(i)->getTransform().inverse() * glob_pos;
		distance[i] = (glob_pos - T.translation()).norm();
	}

	std::vector<int> index_sort_by_distance = sort_indices(distance);
	std::vector<dart::dynamics::BodyNode *> lbs_body_nodes;
	std::vector<Eigen::Vector3d> lbs_local_positions;
	std::vector<double> lbs_weights;

	double total_weight = 0.0;

	if (distance[index_sort_by_distance[0]] < 0.08)
	{
		lbs_weights.push_back(1.0 / sqrt(distance[index_sort_by_distance[0]]));
		total_weight += lbs_weights[0];
		lbs_body_nodes.push_back(skel->getBodyNode(index_sort_by_distance[0]));
		lbs_local_positions.push_back(local_positions[index_sort_by_distance[0]]);

		if (lbs_body_nodes[0]->getParentBodyNode() != nullptr)
		{
			auto bn_parent = lbs_body_nodes[0]->getParentBodyNode();
			lbs_weights.push_back(1.0 / sqrt(distance[bn_parent->getIndexInSkeleton()]));
			total_weight += lbs_weights[1];
			lbs_body_nodes.push_back(bn_parent);
			lbs_local_positions.push_back(local_positions[bn_parent->getIndexInSkeleton()]);
		}
	}
	else
	{
		total_weight = 1.0;
		lbs_weights.push_back(1.0);
		lbs_body_nodes.push_back(bn);
		lbs_local_positions.push_back(bn->getTransform().inverse() * glob_pos);
	}

	for (int i = 0; i < lbs_body_nodes.size(); i++)
	{
		lbs_weights[i] /= total_weight;
	}
	mAnchors.push_back(new Anchor(lbs_body_nodes, lbs_local_positions, lbs_weights));

	int n = mAnchors.size();

	if (n > 1)
		l_mt0 += (mAnchors[n - 1]->GetPoint() - mAnchors[n - 2]->GetPoint()).norm();
	l_mt0_original = l_mt0;

	mCachedAnchorPositions.resize(n);
}
void Muscle::
	AddAnchor(dart::dynamics::BodyNode *bn, const Eigen::Vector3d &glob_pos)
{
	std::vector<dart::dynamics::BodyNode *> lbs_body_nodes;
	std::vector<Eigen::Vector3d> lbs_local_positions;
	std::vector<double> lbs_weights;

	lbs_body_nodes.push_back(bn);
	lbs_local_positions.push_back(bn->getTransform().inverse() * glob_pos);
	lbs_weights.push_back(1.0);

	mAnchors.push_back(new Anchor(lbs_body_nodes, lbs_local_positions, lbs_weights));

	int n = mAnchors.size();
	if (n > 1)
		l_mt0 += (mAnchors[n - 1]->GetPoint() - mAnchors[n - 2]->GetPoint()).norm();

	l_mt0_original = l_mt0;

	mCachedAnchorPositions.resize(n);
}
void Muscle::
	SetMuscle()
{
	int n = mAnchors.size();
	l_mt0 = 0;
	for (int i = 1; i < n; i++)
		l_mt0 += (mAnchors[i]->GetPoint() - mAnchors[i - 1]->GetPoint()).norm();
	l_mt0_original = l_mt0;

	Update();
	Eigen::MatrixXd Jt = GetJacobianTranspose();
	auto Ap = GetForceJacobianAndPassive();
	Eigen::VectorXd JtA = Jt * Ap.first;
	num_related_dofs = 0;
	related_dof_indices.clear();

	related_vec = Eigen::VectorXd::Zero(JtA.rows());

	for (int i = 0; i < JtA.rows(); i++)
	{
		if (std::abs(JtA[i]) > 1E-10)
		{
			num_related_dofs++;
			related_dof_indices.push_back(i);
		}
		if (JtA[i] > 1E-10)
			related_vec[i] = 1;
		else if (JtA[i] < -1E-10)
			related_vec[i] = -1;
		else
			related_vec[i] = 0;
	}
}

void Muscle::
	ApplyForceToBody()
{
	double f = GetForce();

	for (int i = 0; i < mAnchors.size() - 1; i++)
	{
		Eigen::Vector3d dir = mCachedAnchorPositions[i + 1] - mCachedAnchorPositions[i];
		dir.normalize();
		dir = f * dir;
		mAnchors[i]->bodynodes[0]->addExtForce(dir, mCachedAnchorPositions[i], false, false);
	}

	for (int i = 1; i < mAnchors.size(); i++)
	{
		Eigen::Vector3d dir = mCachedAnchorPositions[i - 1] - mCachedAnchorPositions[i];
		dir.normalize();
		dir = f * dir;
		mAnchors[i]->bodynodes[0]->addExtForce(dir, mCachedAnchorPositions[i], false, false);
	}
}
double Muscle::
	GetNormalizedLength()
{
	return l_m / l_m0;
}
void Muscle::
	Update()
{
	for (int i = 0; i < mAnchors.size(); i++)
		mCachedAnchorPositions[i] = mAnchors[i]->GetPoint();
	l_mt = Getl_mt();

	l_m = l_mt - l_t0;
}
double
Muscle::
	GetForce()
{
	return Getf_A() * activation + Getf_p();
}
double
Muscle::
	Getf_A()
{
	return f0 * g_al(l_m / l_m0);
}
double
Muscle::
	Getf_p()
{
	return f0 * g_pl(l_m / l_m0);
}
double
Muscle::
	Getl_mt()
{
	length = 0.0;
	for (int i = 1; i < mAnchors.size(); i++)
		length += (mCachedAnchorPositions[i] - mCachedAnchorPositions[i - 1]).norm();

	return length / l_mt0;
}
Eigen::VectorXd
Muscle::
	GetRelatedJtA()
{
	Eigen::MatrixXd Jt_reduced = GetReducedJacobianTranspose();
	Eigen::VectorXd A = GetForceJacobianAndPassive().first;
	Eigen::VectorXd JtA_reduced = Jt_reduced * A;
	return JtA_reduced;
}

Eigen::MatrixXd
Muscle::
	GetReducedJacobianTranspose()
{
	const auto &skel = mAnchors[0]->bodynodes[0]->getSkeleton();
	Eigen::MatrixXd Jt(num_related_dofs, 3 * mAnchors.size());

	Jt.setZero();
	for (int i = 0; i < mAnchors.size(); i++)
	{
		auto bn = mAnchors[i]->bodynodes[0];
		dart::math::Jacobian J = dart::math::Jacobian::Zero(6, num_related_dofs);
		for (int j = 0; j < num_related_dofs; j++)
		{
			auto &indices = bn->getDependentGenCoordIndices();
			int idx = std::find(indices.begin(), indices.end(), related_dof_indices[j]) - indices.begin();
			if (idx != indices.size())
				J.col(j) = bn->getJacobian().col(idx);
		}
		Eigen::Vector3d offset = mAnchors[i]->bodynodes[0]->getTransform().inverse() * mCachedAnchorPositions[i];
		dart::math::LinearJacobian JLinear = J.bottomRows<3>() + J.topRows<3>().colwise().cross(offset);
		Jt.block(0, i * 3, num_related_dofs, 3) = (bn->getTransform().linear() * JLinear).transpose();
	}
	return Jt;
}

Eigen::VectorXd
Muscle::
	GetRelatedJtp()
{
	Eigen::MatrixXd Jt_reduced = GetReducedJacobianTranspose();
	Eigen::VectorXd P = GetForceJacobianAndPassive().second;
	Eigen::VectorXd JtP_reduced = Jt_reduced * P;
	return JtP_reduced;
}

Eigen::MatrixXd
Muscle::
	GetJacobianTranspose()
{
	const auto &skel = mAnchors[0]->bodynodes[0]->getSkeleton();
	int dof = skel->getNumDofs();

	Eigen::MatrixXd Jt(dof, 3 * mAnchors.size());

	Jt.setZero();
	for (int i = 0; i < mAnchors.size(); i++)
		Jt.block(0, i * 3, dof, 3) = skel->getLinearJacobian(mAnchors[i]->bodynodes[0], mAnchors[i]->bodynodes[0]->getTransform().inverse() * mCachedAnchorPositions[i]).transpose();

	return Jt;
}

std::pair<Eigen::VectorXd, Eigen::VectorXd>
Muscle::
	GetForceJacobianAndPassive()
{
	double f_a = Getf_A();
	double f_p = Getf_p();

	std::vector<Eigen::Vector3d> force_dir;
	for (int i = 0; i < mAnchors.size(); i++)
	{
		force_dir.push_back(Eigen::Vector3d::Zero());
	}

	for (int i = 0; i < mAnchors.size() - 1; i++)
	{
		Eigen::Vector3d dir = mCachedAnchorPositions[i + 1] - mCachedAnchorPositions[i];
		dir.normalize();
		force_dir[i] += dir;
	}

	for (int i = 1; i < mAnchors.size(); i++)
	{
		Eigen::Vector3d dir = mCachedAnchorPositions[i - 1] - mCachedAnchorPositions[i];
		dir.normalize();
		force_dir[i] += dir;
	}

	Eigen::VectorXd A(3 * mAnchors.size());
	Eigen::VectorXd p(3 * mAnchors.size());
	A.setZero();
	p.setZero();

	for (int i = 0; i < mAnchors.size(); i++)
	{
		A.segment<3>(i * 3) = force_dir[i] * f_a;
		p.segment<3>(i * 3) = force_dir[i] * f_p;
	}
	return std::make_pair(A, p);
}

std::vector<dart::dynamics::Joint *>
Muscle::
	GetRelatedJoints()
{
	auto skel = mAnchors[0]->bodynodes[0]->getSkeleton();
	std::map<dart::dynamics::Joint *, int> jns;
	std::vector<dart::dynamics::Joint *> jns_related;
	for (int i = 0; i < skel->getNumJoints(); i++)
		jns.insert(std::make_pair(skel->getJoint(i), 0));

	Eigen::VectorXd dl_dtheta = Getdl_dtheta();

	for (int i = 0; i < dl_dtheta.rows(); i++)
		if (std::abs(dl_dtheta[i]) > 1E-6)
			jns[skel->getDof(i)->getJoint()] += 1;

	for (auto jn : jns)
		if (jn.second > 0)
			jns_related.push_back(jn.first);
	return jns_related;
}
std::vector<dart::dynamics::BodyNode *>
Muscle::
	GetRelatedBodyNodes()
{
	std::vector<dart::dynamics::BodyNode *> bns_related;
	auto rjs = GetRelatedJoints();
	for (auto joint : rjs)
	{
		bns_related.push_back(joint->getChildBodyNode());
	}

	return bns_related;
}
void Muscle::
	ComputeJacobians()
{
	const auto &skel = mAnchors[0]->bodynodes[0]->getSkeleton();
	int dof = skel->getNumDofs();
	mCachedJs.resize(mAnchors.size());
	for (int i = 0; i < mAnchors.size(); i++)
	{
		mCachedJs[i].resize(3, skel->getNumDofs());
		mCachedJs[i].setZero();

		for (int j = 0; j < mAnchors[i]->num_related_bodies; j++)
		{
			mCachedJs[i] += mAnchors[i]->weights[j] * skel->getLinearJacobian(mAnchors[i]->bodynodes[j], mAnchors[i]->local_positions[j]);
		}
	}
}
Eigen::VectorXd
Muscle::
	Getdl_dtheta()
{
	ComputeJacobians();
	const auto &skel = mAnchors[0]->bodynodes[0]->getSkeleton();
	Eigen::VectorXd dl_dtheta(skel->getNumDofs());
	dl_dtheta.setZero();
	for (int i = 0; i < mAnchors.size() - 1; i++)
	{
		Eigen::Vector3d pi = mCachedAnchorPositions[i + 1] - mCachedAnchorPositions[i];
		Eigen::MatrixXd dpi_dtheta = mCachedJs[i + 1] - mCachedJs[i];
		Eigen::VectorXd dli_d_theta = (dpi_dtheta.transpose() * pi) / (l_mt0 * pi.norm());
		dl_dtheta += dli_d_theta;
	}

	for (int i = 0; i < dl_dtheta.rows(); i++)
		if (std::abs(dl_dtheta[i]) < 1E-6)
			dl_dtheta[i] = 0.0;

	return dl_dtheta;
}

double
Muscle::
	g(double _l_m)
{
	double e_t = (l_mt - _l_m - l_t0) / l_t0;
	_l_m = _l_m / l_m0;
	double f = g_t(e_t) - (g_pl(_l_m) + activation * g_al(_l_m));
	return f;
}
double
Muscle::
	g_t(double e_t)
{
	double f_t;
	if (e_t <= e_t0)
		f_t = f_toe / (exp(k_toe) - 1) * (exp(k_toe * e_t / e_toe) - 1);
	else
		f_t = k_lin * (e_t - e_toe) + f_toe;

	return f_t;
}
double
Muscle::
	g_pl(double _l_m)
{
	double f_pl = (exp(k_pe * (_l_m - 1) / e_mo) - 1.0) / (exp(k_pe) - 1.0);
	if (_l_m < 1.0)
		return 0.0;
	else
		return f_pl;
}
double
Muscle::
	g_al(double _l_m)
{
	return exp(-(_l_m - 1.0) * (_l_m - 1.0) / gamma);
}

double
Muscle::
	GetMass()
{
	return 1059.7 * f0 / 250000 * l_mt0;
}

double
Muscle::
	Getdl_velocity()
{
	ComputeJacobians();
	const auto &skel = mAnchors[0]->bodynodes[0]->getSkeleton();
	double dl_velocity = 0.0;
	for (int i = 0; i < mAnchors.size() - 1; i++)
	{
		Eigen::Vector3d dist = mCachedAnchorPositions[i + 1] - mCachedAnchorPositions[i];
		Eigen::Vector3d d_dist = (mCachedJs[i + 1] - mCachedJs[i]) * skel->getVelocities();
		dl_velocity += dist.dot(d_dist) / dist.norm();
	}
	return dl_velocity;
}

std::vector<std::vector<double>>
Muscle::
	GetGraphData()
{
	std::vector<std::vector<double>> result;
	std::vector<double> x;
	std::vector<double> a;
	std::vector<double> a_f;
	std::vector<double> p;
	std::vector<double> current;

	Update();

	result.clear();
	x.clear();
	a.clear();
	a_f.clear();
	p.clear();
	current.clear();

	for (int i = 0; i < 250; i++)
	{
		x.push_back(i * 0.01);
		a.push_back(f0 * g_al(i * 0.01));
		a_f.push_back(f0 * g_al(i * 0.01) * activation);
		p.push_back(f0 * g_pl(i * 0.01));
	}
	current.push_back(GetNormalizedLength());
	result.push_back(current);
	result.push_back(x);
	result.push_back(a);
	result.push_back(a_f);
	result.push_back(p);

	return result;
}

double
Muscle::
	GetRecommendedMinLength()
{
	double r_l =
		1.0 / ((1 / (Getl_mt()) * l_m0 / f0 * ((-e_mo) * log((2000.0 / f0) * (exp(k_pe) - 1) + 1) / k_pe - 1) + l_t0));
	return r_l;
}

double
Muscle::
	GetBHAR04_EnergyRate()
{
	double e_dot = 0.0;
	double a_dot = 0.0;
	double m_dot = 0.0;
	double s_dot = 0.0;
	double w_dot = 0.0;

	double mass = GetMass();
	double f_a_u = 40 * type1_fraction * sin(M_PI * 0.5 * activation) + 133 * (1 - type1_fraction) * (1 - cos(M_PI * 0.5 * activation));
	double f_m_a = 74 * type1_fraction * sin(M_PI * 0.5 * activation) + 111 * (1 - type1_fraction) * (1 - cos(M_PI * 0.5 * activation));
	double g_l = 0.0;
	double l_m_ratio = l_m / l_m0;

	if (l_m_ratio < 0.5)
		g_l = 0.5;
	else if (l_m_ratio < 1.0)
		g_l = l_m_ratio;
	else if (l_m_ratio < 1.5)
		g_l = -2 * l_m_ratio + 3;
	else
		g_l = 0;

	a_dot = mass * f_a_u;
	m_dot = mass * g_l * f_m_a;

	double alpha = 0.0;
	double dl_velocity = Getdl_velocity();
	if (dl_velocity <= 0)
		alpha = 0.16 * Getf_A() + 0.18 * GetForce();
	else
		alpha = 0.157 * GetForce();

	s_dot = -alpha * dl_velocity;

	w_dot = -dl_velocity * GetForce();

	e_dot = a_dot + m_dot + s_dot + w_dot;

	return e_dot;
}