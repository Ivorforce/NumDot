#ifndef VRANDOM_HPP
#define VRANDOM_HPP

#include "varray.hpp"

#include "xtensor/xrandom.hpp"                                    // for random engine

namespace va::random {
	struct VRandomEngine {
		xt::random::default_engine_type engine;

		VRandomEngine();
		explicit VRandomEngine(std::size_t seed);

		VRandomEngine spawn();

		std::shared_ptr<VArray> random_floats(VStoreAllocator& allocator, const shape_type& shape, DType dtype);
		std::shared_ptr<VArray> random_integers(VStoreAllocator& allocator, long long low, long long high, const shape_type& shape, DType dtype, bool endpoint);
		std::shared_ptr<VArray> random_normal(VStoreAllocator& allocator, const shape_type& shape, DType dtype);
	};
}

#endif //VRANDOM_HPP
