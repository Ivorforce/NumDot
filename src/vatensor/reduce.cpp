#include "reduce.h"

#include "vcompute.h"

using namespace va;

#define Reducer(Name, fun_name)\
	Name() = default;\
	Name(const Name&) = default;\
	Name(Name&&) noexcept = default;\
	Name& operator=(const Name&) = default;\
	Name& operator=(Name&&) noexcept = default;\
	~Name() = default;\
\
	template <typename GivenAxes, typename A>\
	auto operator()(GivenAxes&& axes, A&& a) const {\
		return xt::fun_name(std::forward<A>(a), std::forward<GivenAxes>(axes));\
	}\
\
	template <typename A>\
	auto operator()(A&& a) const {\
		return xt::fun_name(std::forward<A>(a));\
	}

struct Sum { Reducer(Sum, sum) };

struct Prod { Reducer(Prod, prod) };

struct Mean { Reducer(Mean, mean) };

struct Variance { Reducer(Variance, variance) };

struct Std { Reducer(Std, stddev) };

struct Amax { Reducer(Amax, amax) };

struct Amin { Reducer(Amin, amin) };

VArray va::sum(const VArray &array, const Axes &axes) {
	return va::xreduction<promote::common_type>(
		Sum{}, axes, array.to_compute_variant()
	);
}

VArray va::prod(const VArray &array, const Axes &axes) {
	return va::xreduction<promote::at_least_int32>(
		Prod{}, axes, array.to_compute_variant()
	);
}

VArray va::mean(const VArray &array, const Axes &axes) {
	return va::xreduction<promote::matching_float_or_default<double_t>>(
		Mean{}, axes, array.to_compute_variant()
	);
}

VArray va::var(const VArray &array, const Axes &axes) {
	return va::xreduction<promote::matching_float_or_default<double_t>>(
		Variance{}, axes, array.to_compute_variant()
	);
}

VArray va::std(const VArray &array, const Axes &axes) {
	return va::xreduction<promote::matching_float_or_default<double_t>>(
		Std{}, axes, array.to_compute_variant()
	);
}

VArray va::max(const VArray &array, const Axes &axes) {
	return va::xreduction<promote::common_type>(
		Amax{}, axes, array.to_compute_variant()
	);
}

VArray va::min(const VArray &array, const Axes &axes) {
	return va::xreduction<promote::common_type>(
		Amin{}, axes, array.to_compute_variant()
	);
}
