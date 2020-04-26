#ifndef IMPL_CENTERS_H__
#define IMPL_CENTERS_H__

#include <array>
#include <map>

struct Center {

	Center() {
		z = 0.0;
		y = 0.0;
		x = 0.0;
		n = 0;
	}

	double z, y, x;
	size_t n;
};

template <typename T>
std::map<T, Center>
centers(
		size_t size_z,
		size_t size_y,
		size_t size_x,
		const T* labels){

	std::map<T, Center> centers;

	size_t n = size_z*size_y*size_x;

	std::array<int, 3> pos({0, 0, 0});
	for (std::ptrdiff_t i = 0; i < n; ++i) {

		T l = labels[i];
		if (l > 0) {

			auto& c = centers[l];
			c.z += pos[0];
			c.y += pos[1];
			c.x += pos[2];
			c.n++;
		}

		pos[2]++;
		if (pos[2] >= size_x) {
			pos[2] = 0;
			pos[1]++;
			if (pos[1] >= size_y) {
				pos[1] = 0;
				pos[0]++;
			}
		}
	}

	for (auto& p : centers) {
		p.second.z /= p.second.n;
		p.second.y /= p.second.n;
		p.second.x /= p.second.n;
	}

	return centers;
}

#endif // IMPL_CENTERS_H__
