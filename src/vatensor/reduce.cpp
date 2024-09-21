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
#include "xtensor/xtensor_forward.hpp"                            // for xtensor_fixed
#include "xtl/xiterator_base.hpp"                       // for operator!=

using namespace va;

// TODO Passing EVS is required because norms don't support it without it, we should make a PR (though it's not bad to explicitly make it lazy).
#define Reducer(Name, fun_name_axes, fun_name_no_axes)\
	template <typename GivenAxes, typename A>\
	auto operator()(GivenAxes&& axes, A&& a) const {\
		return fun_name_axes(std::forward<A>(a), std::forward<GivenAxes>(axes), std::tuple<xt::evaluation_strategy::lazy_type>());\
	}\
\
	template <typename A>\
	auto operator()(A&& a) const {\
		return fun_name_no_axes(std::forward<A>(a), std::tuple<xt::evaluation_strategy::lazy_type>());\
	}

// To be able to use xt::all and xt::any, the second function needs special treatment.
// For one, it can't get EVS as parameter.
// Second, the return type is bool, so it needs to be converted to a temp tensor.
#define ReducerAnyAll(Name, fun_name_axes, fun_name_no_axes)\
	template <typename GivenAxes, typename A>\
	auto operator()(GivenAxes&& axes, A&& a) const {\
		return fun_name_axes(std::forward<A>(a), std::forward<GivenAxes>(axes), std::tuple<xt::evaluation_strategy::lazy_type>());\
	}\
\
	template <typename A>\
	auto operator()(A&& a) const {\
		return xt::xtensor_fixed<bool, xshape<>>(fun_name_no_axes(std::forward<A>(a)));\
	}

struct Sum { Reducer(Sum, xt::sum, xt::sum) };
struct Prod { Reducer(Prod, xt::prod, xt::prod) };
struct Mean { Reducer(Mean, xt::mean, xt::mean) };
struct Variance { Reducer(Variance, xt::variance, xt::variance) };
struct Std { Reducer(Std, xt::stddev, xt::stddev) };

struct Amax { Reducer(Amax, xt::amax, xt::amax) };
struct Amin { Reducer(Amin, xt::amin, xt::amin) };

struct NormL0 { Reducer(NormL0, xt::norm_l0, xt::norm_l0) };
struct NormL1 { Reducer(NormL1, xt::norm_l1, xt::norm_l1) };
struct NormL2 { Reducer(NormL2, xt::norm_l2, xt::norm_l2) };
struct NormLInf { Reducer(NormLInf, xt::norm_linf, xt::norm_linf) };

// FIXME These don't support axes yet, see https://github.com/xtensor-stack/xtensor/issues/1555
using namespace xt;
XTENSOR_REDUCER_FUNCTION(va_any, xt::detail::logical_or, bool, true)
XTENSOR_REDUCER_FUNCTION(va_all, xt::detail::logical_and, bool, false)

struct All { ReducerAnyAll(All, va_all, xt::all) };
struct Any { ReducerAnyAll(Any, va_any, xt::any) };

void va::sum(VArrayTarget target, const VArray& array, const Axes &axes) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	va::xreduction_inplace<promote::num_common_type>(
		Sum{}, axes, target, array.to_compute_variant()
	);
#endif
}

void va::prod(VArrayTarget target, const VArray& array, const Axes &axes) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	va::xreduction_inplace<promote::num_at_least_int32>(
		Prod{}, axes, target, array.to_compute_variant()
	);
#endif
}

void va::mean(VArrayTarget target, const VArray& array, const Axes &axes) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	va::xreduction_inplace<promote::num_matching_float_or_default<double_t>>(
		Mean{}, axes, target, array.to_compute_variant()
	);
#endif
}

void va::var(VArrayTarget target, const VArray& array, const Axes &axes) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	va::xreduction_inplace<promote::num_matching_float_or_default<double_t>>(
		Variance{}, axes, target, array.to_compute_variant()
	);
#endif
}

void va::std(VArrayTarget target, const VArray& array, const Axes &axes) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	va::xreduction_inplace<promote::num_matching_float_or_default<double_t>>(
		Std{}, axes, target, array.to_compute_variant()
	);
#endif
}

void va::max(VArrayTarget target, const VArray& array, const Axes &axes) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	va::xreduction_inplace<promote::num_common_type>(
		Amax{}, axes, target, array.to_compute_variant()
	);
#endif
}

void va::min(VArrayTarget target, const VArray& array, const Axes &axes) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	va::xreduction_inplace<promote::num_common_type>(
		Amin{}, axes, target, array.to_compute_variant()
	);
#endif
}

void va::norm_l0(VArrayTarget target, const VArray &array, const Axes &axes) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	va::xreduction_inplace<promote::num_matching_float_or_default<double_t>>(
		NormL0{}, axes, target, array.to_compute_variant()
	);
#endif
}

void va::norm_l1(VArrayTarget target, const VArray &array, const Axes &axes) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	va::xreduction_inplace<promote::num_matching_float_or_default<double_t>>(
		NormL1{}, axes, target, array.to_compute_variant()
	);
#endif
}

void va::norm_l2(VArrayTarget target, const VArray &array, const Axes &axes) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	va::xreduction_inplace<promote::num_matching_float_or_default<double_t>>(
		NormL2{}, axes, target, array.to_compute_variant()
	);
#endif
}

void va::norm_linf(VArrayTarget target, const VArray &array, const Axes &axes) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	va::xreduction_inplace<promote::num_matching_float_or_default<double_t>>(
		NormLInf{}, axes, target, array.to_compute_variant()
	);
#endif
}

void va::all(VArrayTarget target, const VArray &array, const Axes &axes) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	va::xreduction_inplace<promote::bool_in_bool_out>(
		All{}, axes, target, array.to_compute_variant()
	);
#endif
}

void va::any(VArrayTarget target, const VArray &array, const Axes &axes) {
#ifdef NUMDOT_DISABLE_REDUCTION_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_REDUCTION_FUNCTIONS to enable it.");
#else
	va::xreduction_inplace<promote::bool_in_bool_out>(
		Any{}, axes, target, array.to_compute_variant()
	);
#endif
}
