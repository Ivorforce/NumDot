#ifndef VATENSOR_UTIL_HPP
#define VATENSOR_UTIL_HPP

namespace va::util {
	static std::size_t normalize_axis(const std::ptrdiff_t axis, const std::size_t dimension) {
		if (axis >= 0 && axis >= dimension) {
			throw std::runtime_error("axis out of range");
		}
  		if (axis < 0 && -axis > dimension) {
			throw std::runtime_error("axis out of range");
		}

		return xt::normalize_axis(dimension, axis);
    };
}

#endif //VATENSOR_UTIL_HPP
