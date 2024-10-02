#ifndef VRANDOM_HPP
#define VRANDOM_HPP

#include "varray.hpp"
#include "vatensor/auto_defines.hpp"

#include "xtensor/xrandom.hpp"                                    // for random engine

namespace va::random {
	struct VRandomEngine {
		xt::random::default_engine_type engine;

		VRandomEngine();
		explicit VRandomEngine(std::size_t seed);

		VRandomEngine spawn();

		std::shared_ptr<VArray> random_floats(shape_type shape, DType dtype);
		std::shared_ptr<VArray> random_integers(long long low, long long high, shape_type shape, DType dtype, bool endpoint);
	};
}

#endif //VRANDOM_HPP
