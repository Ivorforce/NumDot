#include "vrandom.hpp"

#include "varray.hpp"

using namespace va;
using namespace va::random;

VRandomEngine::VRandomEngine() : engine(std::random_device()()) {
}

VRandomEngine::VRandomEngine(const std::size_t seed) : engine(xt::random::default_engine_type(seed)) {}

VRandomEngine VRandomEngine::spawn() {
	return VRandomEngine(xt::random::randint<std::size_t>({1}, 0, 0, this->engine)[0]);
}

std::shared_ptr<va::VArray> VRandomEngine::random_floats(shape_type shape, const DType dtype) {
	return std::visit([shape, this](auto t) -> std::shared_ptr<va::VArray> {
		using T = decltype(t);

		if constexpr (!std::is_floating_point_v<T>) {
			throw std::runtime_error("This function can only generate floating point types.");
		}
		else {
			return from_store(make_store<T>(xt::random::rand<T>(shape, 0, 1, this->engine)));
		}
	}, dtype_to_variant(dtype));
}

std::shared_ptr<VArray> VRandomEngine::random_integers(long long low, long long high, shape_type shape, const DType dtype, bool endpoint) {
	return std::visit([low, high, shape, this, endpoint](auto t) -> std::shared_ptr<VArray> {
		using T = decltype(t);

		if constexpr (!std::is_integral_v<T>) {
			throw std::runtime_error("This function can only generate integer types.");
		}
		else {
		// TODO Should automatically figure out somehow which are supported, not hardcode it...
#ifdef _WIN32
		// Windows supports no 8 bit random
		using TRandom = std::conditional_t<
			std::is_same_v<T, int8_t>,
			int16_t,
			std::conditional_t<
				std::is_same_v<T, bool> || std::is_same_v<T, uint8_t>,
				uint16_t,
				T
			>
		>;
#else
		// Unix supports all integrals except bool
		using TRandom = std::conditional_t<std::is_same_v<T, bool>, uint8_t, T>;
#endif
			// FIXME + 1 can cause problems if INT_MAX, but xt does not support an endpoint parameter
			return from_store(make_store<T>(xt::random::randint<TRandom>(shape, low, high + (endpoint ? 1 : 0), this->engine)));
		}
	}, dtype_to_variant(dtype));
}
