#include "vrandom.hpp"

#include "varray.hpp"
#include "vcall.hpp"
#include "vcompute.hpp"

using namespace va;
using namespace va::random;

VRandomEngine::VRandomEngine() : engine(std::random_device()()) {}

VRandomEngine::VRandomEngine(const std::size_t seed) : engine(xt::random::default_engine_type(seed)) {}

VRandomEngine VRandomEngine::spawn() {
	return VRandomEngine(xt::random::randint<std::size_t>({ 1 }, 0, 0, this->engine)[0]);
}

std::shared_ptr<va::VArray> VRandomEngine::random_floats(VStoreAllocator& allocator, const shape_type& shape, const DType dtype) {
	auto array = va::empty(allocator, dtype, shape);
	va::_call_vfunc_inplace(va::vfunc::tables::fill_random_float, array->data, engine);
	return array;
}

std::shared_ptr<VArray> VRandomEngine::random_integers(VStoreAllocator& allocator, long long low, long long high, const shape_type& shape, const DType dtype, bool endpoint) {
	auto array = va::empty(allocator, dtype, shape);
	va::_call_vfunc_inplace(va::vfunc::tables::fill_random_int, array->data, engine, std::move(low), std::move(high));
	return array;
}

std::shared_ptr<va::VArray> VRandomEngine::random_normal(VStoreAllocator& allocator, const shape_type& shape, const DType dtype) {
	auto array = va::empty(allocator, dtype, shape);
	va::_call_vfunc_inplace(va::vfunc::tables::fill_random_normal, array->data, engine);
	return array;
}