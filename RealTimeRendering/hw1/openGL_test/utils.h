#pragma once

#include <iostream>
#include <random>
#include <GLM/glm.hpp>

struct vec_cmp
{
	bool operator()(glm::vec<3, unsigned int> const& a, glm::vec<3, unsigned int> const& b) const
	{
		return (a.x == b.x && (a.y == b.y && (a.z < b.z) || a.y < b.y)) || a.x < b.x;
	}
};

inline std::ostream& operator<<(std::ostream& s, glm::vec2 v) {
	s << "(" << v[0] << ", " << v[1] << ")";
	return s;
}

inline std::ostream& operator<<(std::ostream& s, glm::vec3 v) {
	s << "(" << v[0] << ", " << v[1] << ", " << v[2] << ")";
	return s;
}

inline std::ostream& operator<<(std::ostream& s, glm::vec4 v) {
	s << "(" << v[0] << ", " << v[1] << ", " << v[2] << ", " << v[3] << ")";
	return s;
}

inline double randomFloat() {
	static std::default_random_engine generator;
	static std::uniform_real_distribution<double> distribution(-1, 1); //doubles from -1 to 1
	return distribution(generator);
}
