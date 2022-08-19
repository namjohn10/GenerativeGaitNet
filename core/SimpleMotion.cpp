#include "SimpleMotion.h"

MASS::SimpleMotion::SimpleMotion() {}

std::vector<std::pair<int, double>> MASS::SimpleMotion::getPose(double ratio)
{
	std::vector<std::pair<int, double>> tmp;
	for (int i = 0; i < start.size(); i++)
		tmp.emplace_back(idx[i], start[i] * (1.0 - ratio) + end[i] * ratio);
	return tmp;
}
