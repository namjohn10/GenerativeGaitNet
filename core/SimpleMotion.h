//
// Created by hoseok on 11/3/18.
//

#ifndef EXAM_SIMPLEMOTION_H
#define EXAM_SIMPLEMOTION_H

#include <string>
#include <vector>
#include "Eigen/Core"

namespace MASS
{
	class SimpleMotion
	{
	public:
		SimpleMotion();

		std::vector<std::pair<int, double>> getPose(double ratio);

		std::vector<int> idx;
		std::vector<double> start, end;
		std::string motionName; // ex) "knee flexion", "hip flexion"
	};
}

#endif // EXAM_SIMPLEMOTION_H
