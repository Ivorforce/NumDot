#include "reduce.h"

#include <cmath>                                       // for double_t
#include <utility>                                      // for forward
#include "vatensor/varray.h"                            // for VArray, Axes
#include "vcompute.h"
#include "vpromote.h"                                    // for promote
#include "xtensor/xiterator.hpp"                        // for operator==
#include "xtensor/xlayout.hpp"                          // for layout_type
#include "xtensor/xmath.hpp"                            // for amax, amin, mean
#include "xtensor/xnorm.hpp"                            // for norms
#include "xtl/xiterator_base.hpp"                       // for operator!=

using namespace va;

// TODO Passing EVS is required because norms don't support it without it, we should make a PR (though it's not bad to explicitly make it lazy).
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
		return xt::fun_name(std::forward<A>(a), std::forward<GivenAxes>(axes), std::tuple<xt::evaluation_strategy::lazy_type>());\
	}\
\
	template <typename A>\
	auto operator()(A&& a) const {\
		return xt::fun_name(std::forward<A>(a), std::tuple<xt::evaluation_strategy::lazy_type>());\
	}

struct Sum { Reducer(Sum, sum) };

struct Prod { Reducer(Prod, prod) };

struct Mean { Reducer(Mean, mean) };

struct Variance { Reducer(Variance, variance) };

struct Std { Reducer(Std, stddev) };

struct Amax { Reducer(Amax, amax) };

struct Amin { Reducer(Amin, amin) };

struct NormL0 { Reducer(NormL0, norm_l0) };
struct NormL1 { Reducer(NormL1, norm_l1) };
struct NormL2 { Reducer(NormL2, norm_l2) };
struct NormLInf { Reducer(NormLInf, norm_linf) };

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

void va::norm_l0(VArrayTarget target, const VArray &array, const Axes &axes) {
	va::xreduction_inplace<promote::num_matching_float_or_default<double_t>>(
		NormL0{}, axes, target, array.to_compute_variant()
	);
}

void va::norm_l1(VArrayTarget target, const VArray &array, const Axes &axes) {
	va::xreduction_inplace<promote::num_matching_float_or_default<double_t>>(
		NormL1{}, axes, target, array.to_compute_variant()
	);
}

void va::norm_l2(VArrayTarget target, const VArray &array, const Axes &axes) {
	va::xreduction_inplace<promote::num_matching_float_or_default<double_t>>(
		NormL2{}, axes, target, array.to_compute_variant()
	);
}

void va::norm_linf(VArrayTarget target, const VArray &array, const Axes &axes) {
	va::xreduction_inplace<promote::num_matching_float_or_default<double_t>>(
		NormLInf{}, axes, target, array.to_compute_variant()
	);
}
