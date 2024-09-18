#include "reduce.h"

#include <cmath>                                       // for double_t
#include <utility>                                      // for forward
#include "vatensor/varray.h"                            // for VArray, Axes
#include "vcompute.h"
#include "vpromote.h"                                    // for promote
#include "xtensor/xiterator.hpp"                        // for operator==
#include "xtensor/xlayout.hpp"                          // for layout_type
#include "xtensor/xmath.hpp"                            // for amax, amin, mean
#include "xtl/xiterator_base.hpp"                       // for operator!=

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

void va::sum(VArrayTarget target, const VArray& array, const Axes &axes) {
	va::xreduction_inplace<promote::num_common_type>(
		Sum{}, axes, target, array.to_compute_variant()
	);
}

void va::prod(VArrayTarget target, const VArray& array, const Axes &axes) {
	va::xreduction_inplace<promote::num_at_least_int32>(
		Prod{}, axes, target, array.to_compute_variant()
	);
}

void va::mean(VArrayTarget target, const VArray& array, const Axes &axes) {
	va::xreduction_inplace<promote::num_matching_float_or_default<double_t>>(
		Mean{}, axes, target, array.to_compute_variant()
	);
}

void va::var(VArrayTarget target, const VArray& array, const Axes &axes) {
	va::xreduction_inplace<promote::num_matching_float_or_default<double_t>>(
		Variance{}, axes, target, array.to_compute_variant()
	);
}

void va::std(VArrayTarget target, const VArray& array, const Axes &axes) {
	va::xreduction_inplace<promote::num_matching_float_or_default<double_t>>(
		Std{}, axes, target, array.to_compute_variant()
	);
}

void va::max(VArrayTarget target, const VArray& array, const Axes &axes) {
	va::xreduction_inplace<promote::num_common_type>(
		Amax{}, axes, target, array.to_compute_variant()
	);
}

void va::min(VArrayTarget target, const VArray& array, const Axes &axes) {
	va::xreduction_inplace<promote::num_common_type>(
		Amin{}, axes, target, array.to_compute_variant()
	);
}
